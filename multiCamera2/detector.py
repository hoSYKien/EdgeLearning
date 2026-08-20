# -*- coding: utf-8 -*-
"""
THUẬT TOÁN PHÁT HIỆN VẬT THỂ & KHỬ BÓNG CHUYÊN SÂU (SHADOW REMOVAL) & ĐỌC BARCODE
================================================================================
Tham chiếu từ thuật toán chuẩn trong live_crop_detect_v2.py:
  1. Mô hình hóa nền BackgroundModel (Vector hóa 3D RGB/BGR).
  2. Khử bóng bằng độ bất biến màu (Normalized Chromaticity Invariance):
     - So sánh sắc độ giữa ảnh thực và ảnh nền để phát hiện bóng râm có cùng màu với mặt bàn.
     - Kiểm tra dải độ sáng (Shadow Dimming Range) và triệt tiêu hoàn toàn bóng.
  3. Ngưỡng kép Hysteresis (THRESH_STRONG & THRESH_WEAK) kết hợp Connected Components:
     - Giữ trọn vẹn đường biên vật thể mềm/phản quang mà không bị nhiễu hạt lốm đốm.
  4. Lấp đầy lỗ trống (FloodFill Holes) & Biến đổi hình thái học (Morphology Open/Close):
     - Không bị thủng lỗ mask do tem nhãn/chữ in trên bao bì.
  5. Đọc mã vạch tốc độ cao (zxing-cpp / pyzbar) với CLAHE tiền xử lý.
"""

import time
import cv2
import numpy as np

from config import (
    RESIZE_DIV, THRESH_STRONG, THRESH_WEAK, CHROMA_TOL,
    SHADOW_DIM, SHADOW_MAX_DIFF, FILL_HOLES, CROP_PADDING,
    MIN_AREA_RATIO, MAX_AREA_RATIO
)

# Nạp các engine barcode
_HAS_ZXING = False
try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:
    pass

_HAS_PYZBAR = False
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    pass


class BackgroundModel:
    """Mô hình nền lưu thông tin màu sắc và độ sáng chuẩn bị cho phép tính khử bóng."""
    def __init__(self, bg_bgr_float):
        self.bg = bg_bgr_float.astype(np.float32)
        b, g, r = cv2.split(self.bg)
        self.bg_sum = b + g + r + 3.0
        self.bg_sum_3d = cv2.merge([self.bg_sum, self.bg_sum, self.bg_sum])
        self.bg_plus_1 = self.bg + 1.0
        self.shadow_dim_low = SHADOW_DIM[0] * self.bg_sum
        self.shadow_dim_high = SHADOW_DIM[1] * self.bg_sum


class ObjectDetector:
    """Bộ phát hiện vật thể tích hợp thuật toán khử bóng và lọc mask chuẩn xác."""

    def __init__(self, resize_div=RESIZE_DIV,
                 thresh_strong=THRESH_STRONG, thresh_weak=THRESH_WEAK,
                 chroma_tol=CHROMA_TOL, shadow_max_diff=SHADOW_MAX_DIFF,
                 min_area_ratio=MIN_AREA_RATIO, max_area_ratio=MAX_AREA_RATIO,
                 crop_padding=CROP_PADDING):
        self.resize_div = resize_div
        self.thresh_strong = thresh_strong
        self.thresh_weak = thresh_weak
        self.chroma_tol = chroma_tol
        self.shadow_max_diff = shadow_max_diff
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.crop_padding = crop_padding
        
        self.bg_model = None
        self.last_mask = None  # Lưu mask debug cho giao diện hiển thị

    def learn_background(self, cam, num_frames=15, timeout=25.0):
        """Học nền từ camera (khung nhìn mặt bàn TRỐNG không có sản phẩm)."""
        frames = []
        t0 = time.time()
        fps = getattr(cam, 'target_fps', 10.0) or 10.0
        calculated_timeout = max(timeout, (num_frames / fps) + 10.0)

        while len(frames) < num_frames:
            f = cam.read()
            if f is None:
                if time.time() - t0 > calculated_timeout:
                    print(f"[{cam.name}] Timeout khi học nền!")
                    return False
                time.sleep(0.02)
                continue
            h, w = f.shape[:2]
            small = cv2.resize(f, (w // self.resize_div, h // self.resize_div))
            blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
            frames.append(blur)
            time.sleep(0.02)

        bg_mean = np.mean(frames, axis=0)
        self.bg_model = BackgroundModel(bg_mean)
        print(f"[{cam.name}] Đã học nền thành công với mô hình khử bóng ({len(frames)} frames).")
        return True

    def _mask_from_background(self, img_small):
        """Tạo mask nhị phân loại bỏ bóng bằng phân tích sắc độ (Chromaticity Invariance)."""
        blur = cv2.GaussianBlur(img_small, (7, 7), 0)
        img_f = blur.astype(np.float32)
        
        # 1. Sai khác màu sắc tuyệt đối
        diff_c = cv2.absdiff(img_f, self.bg_model.bg)
        b, g, r = cv2.split(diff_c)
        diff = cv2.max(cv2.max(b, g), r)

        # 2. Phân tích sắc độ & Loại trừ vùng bóng đổ
        ib, ig, ir = cv2.split(img_f)
        S_img = ib + ig + ir + 3.0
        S_img_3d = cv2.merge([S_img, S_img, S_img])

        term_img = (img_f + 1.0) * self.bg_model.bg_sum_3d
        term_bg = self.bg_model.bg_plus_1 * S_img_3d
        chroma_diff = cv2.absdiff(term_img, term_bg)
        cb, cg, cr = cv2.split(chroma_diff)
        max_chroma = cv2.max(cv2.max(cb, cg), cr)

        # Điều kiện điểm ảnh là bóng:
        # - Cùng sắc độ màu với nền mặt bàn
        # - Độ sáng bị tối đi trong dải bóng (30% - 95% nền)
        # - Sai khác không vượt quá mức tối đa của bóng
        same_color = max_chroma < (self.chroma_tol * self.bg_model.bg_sum) * S_img
        dim_moderate = (S_img > self.bg_model.shadow_dim_low) & (S_img < self.bg_model.shadow_dim_high)
        is_shadow = same_color & dim_moderate & (diff < self.shadow_max_diff)
        
        # Triệt tiêu bóng hoàn toàn khỏi mask chênh lệch
        diff[is_shadow] = 0.0

        # 3. Ngưỡng kép Hysteresis (Mạnh / Yếu)
        strong = (diff > self.thresh_strong).astype(np.uint8)
        weak = (diff > self.thresh_weak).astype(np.uint8)

        if cv2.countNonZero(strong) == 0:
            return np.zeros(img_small.shape[:2], dtype=np.uint8)

        # 4. Giữ các thành phần liên thông có chứa ít nhất 1 pixel 'strong'
        n_lbl, lbl = cv2.connectedComponents(weak, connectivity=8)
        strong_labels = np.unique(lbl[strong > 0])
        strong_labels = strong_labels[strong_labels != 0]

        keep = np.zeros(n_lbl, dtype=bool)
        keep[strong_labels] = True
        return (keep[lbl].astype(np.uint8) * 255)

    def _fill_holes(self, mask):
        """Lấp đầy các lỗ trống/chữ in bên trong vật thể bằng FloodFill."""
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:-1, 1:-1] = mask
        ff = padded.copy()
        ff_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        filled = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
        return filled[1:-1, 1:-1]

    def _refine_mask(self, mask):
        """Làm mịn mask, loại bỏ nhiễu biên và làm liền mạch vật thể."""
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        if FILL_HOLES:
            mask = self._fill_holes(mask)
        return mask

    def detect_objects(self, full_bgr):
        """
        Phát hiện toàn bộ vật thể trên khung hình với thuật toán khử bóng.
        Trả về: list các tuple ((cx, cy), (x1, y1, x2, y2)) trên ảnh gốc.
        """
        if self.bg_model is None or full_bgr is None:
            return []

        h_full, w_full = full_bgr.shape[:2]
        small_w = w_full // self.resize_div
        small_h = h_full // self.resize_div
        small = cv2.resize(full_bgr, (small_w, small_h))

        # Tạo mask đã khử sạch bóng và lấp đầy lỗ
        raw_mask = self._mask_from_background(small)
        clean_mask = self._refine_mask(raw_mask)
        self.last_mask = clean_mask  # Lưu lại để xem debug

        area_img = small_w * small_h
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area_ratio * area_img <= a <= self.max_area_ratio * area_img:
                x, y, ww, hh = cv2.boundingRect(c)
                
                # Tỉ lệ phóng to lại theo ảnh gốc
                x1 = x * self.resize_div - self.crop_padding
                y1 = y * self.resize_div - self.crop_padding
                x2 = (x + ww) * self.resize_div + self.crop_padding
                y2 = (y + hh) * self.resize_div + self.crop_padding

                # Giới hạn tọa độ trong ảnh gốc
                x1 = max(0, min(w_full - 1, x1))
                y1 = max(0, min(h_full - 1, y1))
                x2 = max(0, min(w_full, x2))
                y2 = max(0, min(h_full, y2))

                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    objects.append(((cx, cy), (x1, y1, x2, y2)))

        return objects

    @staticmethod
    def decode_barcode(crop_img):
        """
        Giải mã barcode từ vùng crop của vật.
        Hỗ trợ toàn diện cả ảnh Màu (Color 3-channel) lẫn ảnh Đen Trắng (Mono 1-channel).
        """
        if crop_img is None or crop_img.size == 0:
            return None
        
        # 1. Chuẩn hóa về ảnh xám Grayscale
        if len(crop_img.shape) == 2:
            gray = crop_img
        elif len(crop_img.shape) == 3:
            if crop_img.shape[2] == 1:
                gray = crop_img[:, :, 0]
            else:
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            return None

        # 2. Thử giải mã bằng zxing-cpp (Rất nhanh và chịu nghiêng tốt)
        if _HAS_ZXING:
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                for im in (gray, enhanced):
                    res = zxingcpp.read_barcodes(im)
                    for r in res:
                        if r.text:
                            return r.text.strip()
            except Exception:
                pass

        # 3. Thử giải mã dự phòng bằng pyzbar
        if _HAS_PYZBAR:
            try:
                for im in (gray, cv2.equalizeHist(gray)):
                    res = _pyzbar_decode(im)
                    for r in res:
                        if r.data:
                            return r.data.decode('utf-8', errors='ignore').strip()
            except Exception:
                pass

        return None
