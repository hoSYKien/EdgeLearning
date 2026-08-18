# -*- coding: utf-8 -*-
"""
THUẬT TOÁN PHÁT HIỆN VẬT THỂ & ĐỌC MÃ VẠCH (ZXING-CPP / PYZBAR)
"""

import time
import cv2
import numpy as np

from config import RESIZE_DIV, BG_THRESH, MIN_AREA_RATIO, MAX_AREA_RATIO

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


class ObjectDetector:
    """Trừ nền học động để phát hiện vị trí vật và đọc mã vạch."""

    def __init__(self, resize_div=RESIZE_DIV, thresh=BG_THRESH,
                 min_area_ratio=MIN_AREA_RATIO, max_area_ratio=MAX_AREA_RATIO):
        self.resize_div = resize_div
        self.thresh = thresh
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.bg_model = None

    def learn_background(self, cam, num_frames=30, timeout=10.0):
        """Học nền từ camera (khung hình không chứa sản phẩm)."""
        frames = []
        t0 = time.time()
        while len(frames) < num_frames:
            f = cam.read()
            if f is None:
                if time.time() - t0 > timeout:
                    print(f"[{cam.name}] Timeout khi học nền!")
                    return False
                time.sleep(0.02)
                continue
            h, w = f.shape[:2]
            small = cv2.resize(f, (w // self.resize_div, h // self.resize_div))
            blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
            frames.append(blur)
            time.sleep(0.03)

        self.bg_model = np.mean(frames, axis=0)
        print(f"[{cam.name}] Đã học nền thành công ({len(frames)} frames).")
        return True

    def detect_objects(self, full_bgr):
        """Trả về list ( (cx, cy), bbox(x1,y1,x2,y2) )."""
        if self.bg_model is None or full_bgr is None:
            return []
        h, w = full_bgr.shape[:2]
        small = cv2.resize(full_bgr, (w // self.resize_div, h // self.resize_div))
        blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
        diff = cv2.absdiff(blur, self.bg_model)
        b, g, r = cv2.split(diff)
        d = cv2.max(cv2.max(b, g), r)
        mask = (d > self.thresh).astype(np.uint8) * 255
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        area_img = small.shape[0] * small.shape[1]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area_ratio * area_img <= a <= self.max_area_ratio * area_img:
                x, y, ww, hh = cv2.boundingRect(c)
                x1, y1 = x * self.resize_div, y * self.resize_div
                x2, y2 = (x + ww) * self.resize_div, (y + hh) * self.resize_div
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                objects.append(((cx, cy), (x1, y1, x2, y2)))
        return objects

    @staticmethod
    def decode_barcode(crop_bgr):
        """Giải mã barcode từ vùng crop của vật."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        
        # 1. Thử zxing-cpp
        if _HAS_ZXING:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            for im in (gray, cv2.createCLAHE(3.0, (8, 8)).apply(gray)):
                try:
                    res = zxingcpp.read_barcodes(im)
                    for r in res:
                        if r.text:
                            return r.text.strip()
                except Exception:
                    pass

        # 2. Thử pyzbar
        if _HAS_PYZBAR:
            try:
                res = _pyzbar_decode(crop_bgr)
                for r in res:
                    if r.data:
                        return r.data.decode('utf-8', errors='ignore').strip()
            except Exception:
                pass

        return None
