"""
Chụp ảnh trực tiếp từ camera Hikrobot (nhấn 'c' để chụp, 'q' để thoát).

Mỗi lần nhấn 'c':
    1. Tự động cắt ROI vật thể: tìm contour -> làm thẳng -> ÉP USB VỀ BÊN
       PHẢI bằng hình học (phía lắp USB bị contour khoét lõm sâu hơn hẳn
       phía vết khuyết tab; chỉ khi tỉ lệ lõm không đủ rõ mới nhờ SIFT đối
       chiếu ảnh chuẩn) -> cắt ROI có padding -> ÁP MASK HÌNH HỌC CỐ ĐỊNH
       (xóa nền thành đen tuyệt đối) -> cắt sát mask.
       Hiển thị ("Anh da crop") và LƯU vào THU_MUC_ANH_CROP.
    2. Cắt tiếp các VÙNG ROI con đã vẽ sẵn (đọc từ file JSON) -> hiển thị
       từng vùng ("Vung ROI 1", ...) và LƯU vào THU_MUC_ANH_ROI_CON.
    3. Đưa vùng ROI được chỉ định (CHI_SO_ROI_PHAN_LOAI) vào model phân
       loại + vẽ Grad-CAM -> hiển thị ("Ket qua phan loai + GradCAM").

QUAN TRỌNG - phải khớp với script tạo tập training (01_tao_template_v7.py):
    - PAD_ROI_NGANG / PAD_ROI_DOC / toàn bộ THAM SỐ MASK HÌNH HỌC /
      CAT_SAT_MASK / LE_TRONG phải GIỐNG HỆT bên script tạo template.
    - KHUNG_CHUAN_W/H phải bằng đúng "Kích thước ROI chuẩn chung (trung vị)"
      mà script tạo template in ra ở lượt 1, vì mask khai báo theo TỈ LỆ của
      khung đó. Đặt sai -> mask lệch -> ảnh inference khác ảnh training ->
      model đoán bậy mà không báo lỗi gì cả.
"""

import sys
import os
import time
import json
import numpy as np
import cv2

# ============================================================
# 1. Trỏ tới thư mục chứa DLL runtime của MVS
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(dll_dir)

# ============================================================
# 2. Trỏ THẲNG vào thư mục MvImport (nằm cùng cấp với file này)
# ============================================================
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)

# ============================================================
# 3. Import kiểu module thường (không có tiền tố MvImport.)
# ============================================================
from MvCameraControl_class import *

# ============================================================
# 4. Import cho phần phân loại + Grad-CAM
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
# Ảnh crop chính (đã cắt + xoay + xóa nền) được lưu ở đây.
THU_MUC_ANH_CROP = r"D:\TongHop\RTC Technologi\PCB\crop7\NG"

# Các vùng ROI con (cắt nhỏ hơn từ ảnh crop chính) được lưu ở đây.
THU_MUC_ANH_ROI_CON = r"D:\TongHop\RTC Technologi\PCB\crop7\captured_roi_con"

# Ảnh CHUẨN CỐ ĐỊNH - CHỈ dùng cho nhánh dự phòng SIFT khi hình học không
# phân biệt được hai đầu. Nên là ảnh GỐC từ camera, cùng điều kiện chụp.
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\crop5\train\OK\Image_20260723101631285_roi.png"

# File chứa tọa độ các vùng ROI con đã vẽ sẵn bằng script 02_ve_va_luu_roi.py
DUONG_DAN_FILE_ROI = r"D:\TongHop\RTC Technologi\PCB\vung_roi.json"

# --- Cấu hình phân loại + Grad-CAM ---
MODEL_PATH = r"D:\TongHop\RTC Technologi\PCB\model\model7\edge_classifier_fewshot_mobilenet_v3_large.pt"
CHI_SO_ROI_PHAN_LOAI = 1   # dùng vùng "Vung ROI 1" để đưa vào model phân loại
TARGET_CLASS = None        # None = dùng class model dự đoán; hoặc đặt tên cụ thể vd "NG"
HEATMAP_ALPHA = 0.45
# =====================================================================

# ========= THAM SỐ CROP + MASK (PHẢI GIỐNG SCRIPT TẠO TEMPLATE) =========
# Kích thước khung crop chuẩn - lấy từ dòng "Kích thước ROI chuẩn chung
# (trung vị)" mà 01_tao_template_v7.py in ra ở lượt 1.
KHUNG_CHUAN_W = 3187
KHUNG_CHUAN_H = 1777

# Nới khung cắt ĐỐI XỨNG 2 bên (pixel ảnh gốc), đủ để chứa trọn đầu USB.
PAD_ROI_NGANG = 300
PAD_ROI_DOC = 60

# Ngưỡng tin cậy khi xác định hướng bằng hình học. Dưới ngưỡng -> nhờ SIFT.
NGUONG_TI_LE_LOM = 2.0

AP_MASK_HINH_HOC = True

# CÁC SỐ DƯỚI ĐÂY ĐÃ ĐO TRỰC TIẾP TRÊN ẢNH RAW (board bbox 2587x1657 px,
# padding 300/60 -> khung crop 3187x1777 px). Chỉ chỉnh khi đổi ống kính /
# khoảng cách chụp / loại board - và phải chỉnh ĐỒNG THỜI ở cả 2 script.

# --- Thân board: chữ nhật bo góc (tỉ lệ bề rộng W và chiều cao H)
BOARD_X1 = 0.0948    # mép trái board   (302/3187)
BOARD_X2 = 0.9049    # mép phải board   (2884/3187)
BOARD_Y1 = 0.0360    # mép trên board   (64/1777)
BOARD_Y2 = 0.9657    # mép dưới board   (1716/1777)
BOARD_BO_GOC = 0.0141  # bán kính bo góc, tỉ lệ CHIỀU CAO (25/1777)

# --- Khối USB Type-C nhô ra bên PHẢI (chồng lấn vào thân board cho kín mối)
USB_X1 = 0.8472      # (2700/3187) nằm trong thân board
USB_X2 = 0.9576      # mép ngoài cùng vỏ USB (3052/3187)
USB_Y1 = 0.2746      # mép trên vỏ USB (488/1777)
USB_Y2 = 0.6989      # mép dưới vỏ USB (1242/1777)
USB_BO_GOC = 0.0113  # (20/1777)

# --- Vết khuyết (tab bẻ) khoét ĐEN ở cạnh TRÁI. Đặt KHUYET_X2 = 0 để tắt.
KHUYET_X2 = 0.1403   # khoét từ mép trái ảnh tới đây (447/3187)
KHUYET_Y1 = 0.1435   # (255/1777)
KHUYET_Y2 = 0.8694   # (1545/1777)
KHUYET_BO_GOC = 0.0535  # (95/1777)

# Nới/co toàn bộ mask vài pixel: >0 giãn ra, <0 co vào, 0 = đúng số đo.
NOI_MASK = 0

# Làm mượt biên mask (0 = tắt, phải là số lẻ).
LAM_MUOT_BIEN = 5

# Cắt sát bounding box của mask, chừa LE_TRONG px viền đen.
CAT_SAT_MASK = True
LE_TRONG = 15
# =====================================================================

SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000
MIN_INLIERS_HUONG = 15
PAD_MASK_SIFT = 60

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ====================================================================
# CÁC HÀM XỬ LÝ ẢNH - CROP VẬT THỂ CHÍNH (đồng bộ với script template)
# ====================================================================

def tim_contour_tu_dong(img, target_w=1200):
    """Threshold Saturation + lọc thành phần liên thông lớn nhất + lấp lỗ.
    Chỉ bắt phần PCB xanh; vỏ USB do mask hình học lo."""
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    def k(size):
        return max(3, round(size * scale) // 2 * 2 + 1)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    blur = cv2.medianBlur(s, k(15))
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k(7), k(7)), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k(41), k(41)), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return (contour.astype(np.float32) / scale).astype(np.int32)


def _do_do_lom(contour_aligned):
    """Đo độ lõm ăn vào từ mép trái và mép phải của contour (đã xoay thẳng).
    Phía lắp USB lõm rất sâu vì contour (ngưỡng Saturation) không bắt được
    vỏ kim loại; phía kia chỉ có vết khuyết tab nông hơn nhiều."""
    x, y, w, h = cv2.boundingRect(contour_aligned)
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [contour_aligned - [x, y]], -1, 255, cv2.FILLED)
    m = m > 0
    trai, phai = [], []
    for r in range(h):
        cot = np.where(m[r])[0]
        if len(cot):
            trai.append(cot.min())
            phai.append(cot.max())
    if not trai:
        return 0.0, 0.0
    trai, phai = np.array(trai), np.array(phai)
    return float(np.median(trai) - trai.min()), float(phai.max() - np.median(phai))


def tinh_m_align_va_roi(contour):
    """Làm thẳng vật thể, ép quy ước ngang > cao, ÉP USB VỀ BÊN PHẢI bằng
    hình học, rồi nới ĐỐI XỨNG 2 bên.
    Trả về (M_align, roi, ti_le_lom) - ti_le_lom càng lớn càng chắc chắn."""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    M_align = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
    x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    if h > w:
        M_xoay_them = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 90, 1.0)
        M_align_3x3 = np.vstack([M_align, [0, 0, 1]])
        M_xoay_them_3x3 = np.vstack([M_xoay_them, [0, 0, 1]])
        M_align = (M_xoay_them_3x3 @ M_align_3x3)[:2, :]
        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    # --- ép USB về bên PHẢI (phía lõm sâu hơn) ---
    lom_trai, lom_phai = _do_do_lom(contour_aligned.astype(np.int32))
    if lom_trai > lom_phai:
        M_180 = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 180, 1.0)
        M_align = (np.vstack([M_180, [0, 0, 1]]) @ np.vstack([M_align, [0, 0, 1]]))[:2, :]
        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))
    ti_le_lom = max(lom_trai, lom_phai) / (min(lom_trai, lom_phai) + 1e-6)

    x -= PAD_ROI_NGANG
    y -= PAD_ROI_DOC
    w += 2 * PAD_ROI_NGANG
    h += 2 * PAD_ROI_DOC
    return M_align, (x, y, w, h), ti_le_lom


def cat_an_toan(img, x, y, w, h):
    """Cắt (x, y, w, h) kể cả khi tràn mép ảnh; phần tràn lấp ĐEN."""
    H, W = img.shape[:2]
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 > x1 and y2 > y1:
        out[y1 - y:y2 - y, x1 - x:x2 - x] = img[y1:y2, x1:x2]
    return out


def _chu_nhat_bo_goc(mask, x1, y1, x2, y2, r, mau):
    """Tô chữ nhật bo 4 góc (r = bán kính bo, pixel)."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(mask, (x1, y1), (x2, y2), mau, -1)
        return
    cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), mau, -1)
    cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), mau, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(mask, (cx, cy), r, mau, -1)


def tao_mask_hinh_hoc(W: int, H: int):
    """Dựng mask cố định cho khung ảnh crop W x H (USB nằm bên PHẢI)."""
    mask = np.zeros((H, W), np.uint8)

    # 1. Thân board (chữ nhật bo góc)
    _chu_nhat_bo_goc(mask,
                     int(BOARD_X1 * W), int(BOARD_Y1 * H),
                     int(BOARD_X2 * W), int(BOARD_Y2 * H),
                     int(BOARD_BO_GOC * H), 255)

    # 2. Khối USB nhô ra bên phải
    _chu_nhat_bo_goc(mask,
                     int(USB_X1 * W), int(USB_Y1 * H),
                     int(USB_X2 * W), int(USB_Y2 * H),
                     int(USB_BO_GOC * H), 255)

    # 3. Khoét vết khuyết cạnh trái (tô ĐEN đè lên)
    if KHUYET_X2 > 0:
        _chu_nhat_bo_goc(mask,
                         0, int(KHUYET_Y1 * H),
                         int(KHUYET_X2 * W), int(KHUYET_Y2 * H),
                         int(KHUYET_BO_GOC * H), 0)

    if NOI_MASK > 0:
        mask = cv2.dilate(mask, np.ones((2 * NOI_MASK + 1,) * 2, np.uint8))
    elif NOI_MASK < 0:
        mask = cv2.erode(mask, np.ones((2 * -NOI_MASK + 1,) * 2, np.uint8))

    if LAM_MUOT_BIEN and LAM_MUOT_BIEN >= 3:
        mask = cv2.medianBlur(mask, LAM_MUOT_BIEN | 1)
    return mask


def trich_sift_anh(img, target_w=SIFT_TARGET_WIDTH, mask_full=None, nfeatures=SIFT_NFEATURES):
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask_small = None
    if mask_full is not None:
        mask_small = cv2.resize(mask_full, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray, mask_small)
    for kpt in kp:
        kpt.pt = (kpt.pt[0] / scale, kpt.pt[1] / scale)
    return kp, des


def uoc_luong_M_sift(kp1, des1, kp2, des2, min_inliers=MIN_INLIERS_HUONG):
    if des1 is None or des2 is None:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
    if M is None or inliers is None or inliers.sum() < min_inliers:
        return None, len(good)
    return M, int(inliers.sum())


def chuan_bi_anh_chuan(duong_dan_anh_chuan):
    """Chuẩn bị dữ liệu SIFT của ảnh chuẩn - CHỈ dùng cho nhánh dự phòng khi
    hình học không phân biệt được hai đầu."""
    img_chuan = cv2.imread(duong_dan_anh_chuan)
    if img_chuan is None:
        raise FileNotFoundError(f"Không đọc được ảnh chuẩn cố định: {duong_dan_anh_chuan}")

    contour_chuan = tim_contour_tu_dong(img_chuan)
    if contour_chuan is None:
        raise RuntimeError(f"Không tìm thấy vật thể trong ảnh chuẩn: {duong_dan_anh_chuan}")

    m_align_chuan, _, ti_le_lom_chuan = tinh_m_align_va_roi(contour_chuan)

    xg, yg, wg, hg = cv2.boundingRect(contour_chuan)
    mask_chuan = np.zeros(img_chuan.shape[:2], np.uint8)
    mask_chuan[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
               max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
    kp_chuan, des_chuan = trich_sift_anh(img_chuan, mask_full=mask_chuan)

    R_chuan_inv = cv2.invertAffineTransform(m_align_chuan)[:, :2]
    dir_chuan = R_chuan_inv @ np.array([1.0, 0.0])

    return {
        "kp_chuan": kp_chuan,
        "des_chuan": des_chuan,
        "dir_chuan": dir_chuan,
        "ti_le_lom": ti_le_lom_chuan,
    }


def crop_va_dong_bo_huong(img, anh_chuan_info):
    """Trả về (ảnh crop đã xóa nền, thông báo) hoặc (None, lý do lỗi)."""
    contour = tim_contour_tu_dong(img)
    if contour is None:
        return None, "Không tìm thấy vật thể trong ảnh vừa chụp."

    # 1. Xoay thẳng + ép USB về bên phải bằng hình học
    m_align, roi, ti_le_lom = tinh_m_align_va_roi(contour)
    x, y, w, h = roi
    warped = cv2.warpAffine(img, m_align, (img.shape[1], img.shape[0]))
    crop = cat_an_toan(warped, x, y, w, h)

    # 2. Về đúng khung chuẩn (mask khai báo theo tỉ lệ của khung này)
    if crop.shape[1] != KHUNG_CHUAN_W or crop.shape[0] != KHUNG_CHUAN_H:
        crop = cv2.resize(crop, (KHUNG_CHUAN_W, KHUNG_CHUAN_H))

    # 3. Hình học không chắc -> nhờ SIFT quyết lại (phải xong TRƯỚC khi áp
    #    mask, vì mask bất đối xứng trái/phải)
    if ti_le_lom > NGUONG_TI_LE_LOM:
        thong_bao = f"Hướng xác định bằng hình học (tỉ lệ lõm {ti_le_lom:.1f})."
    else:
        xg, yg, wg, hg = cv2.boundingRect(contour)
        mask_sift = np.zeros(img.shape[:2], np.uint8)
        mask_sift[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
                  max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
        kp, des = trich_sift_anh(img, mask_full=mask_sift)
        M, so_inlier = uoc_luong_M_sift(kp, des, anh_chuan_info["kp_chuan"],
                                        anh_chuan_info["des_chuan"])
        if M is None:
            thong_bao = (f"Hình học không chắc (tỉ lệ lõm {ti_le_lom:.1f}) và SIFT cũng "
                         f"không đủ tin cậy ({so_inlier} match) - GIỮ NGUYÊN, "
                         f"HÃY KIỂM TRA BẰNG MẮT.")
        else:
            R_inv = cv2.invertAffineTransform(m_align)[:, :2]
            dir_nay = R_inv @ np.array([1.0, 0.0])
            dir_nay_trong_he_chuan = M[:, :2] @ dir_nay
            goc_do = np.degrees(np.arccos(
                np.clip(np.dot(dir_nay_trong_he_chuan, anh_chuan_info["dir_chuan"]) /
                        (np.linalg.norm(dir_nay_trong_he_chuan) *
                         np.linalg.norm(anh_chuan_info["dir_chuan"]) + 1e-9), -1, 1)))
            if goc_do > 90:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT LẬT 180° "
                             f"(lệch {goc_do:.0f}°, {so_inlier} inlier).")
            else:
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT xác nhận "
                             f"giữ nguyên (lệch {goc_do:.0f}°, {so_inlier} inlier).")

    # 4. Áp mask hình học + cắt sát
    if AP_MASK_HINH_HOC:
        mask = tao_mask_hinh_hoc(crop.shape[1], crop.shape[0])
        crop[mask == 0] = 0
        if CAT_SAT_MASK:
            bx, by, bw, bh = cv2.boundingRect(mask)
            crop = crop[max(0, by - LE_TRONG):min(crop.shape[0], by + bh + LE_TRONG),
                        max(0, bx - LE_TRONG):min(crop.shape[1], bx + bw + LE_TRONG)]

    return crop, thong_bao


def kich_thuoc_output_chuan():
    """Kích thước ảnh crop cuối cùng - suy ra từ mask, không cần chụp thử."""
    mask = tao_mask_hinh_hoc(KHUNG_CHUAN_W, KHUNG_CHUAN_H)
    if not AP_MASK_HINH_HOC or not CAT_SAT_MASK:
        return KHUNG_CHUAN_W, KHUNG_CHUAN_H
    bx, by, bw, bh = cv2.boundingRect(mask)
    x1, y1 = max(0, bx - LE_TRONG), max(0, by - LE_TRONG)
    x2 = min(KHUNG_CHUAN_W, bx + bw + LE_TRONG)
    y2 = min(KHUNG_CHUAN_H, by + bh + LE_TRONG)
    return x2 - x1, y2 - y1


def resize_de_hien_thi(img, max_dim=900):
    h, w = img.shape[:2]
    ti_le = min(1.0, max_dim / max(h, w))
    if ti_le >= 1.0:
        return img
    return cv2.resize(img, (int(w * ti_le), int(h * ti_le)))


# ====================================================================
# ĐỌC + CẮT CÁC VÙNG ROI CON (đã vẽ sẵn bằng script 02_ve_va_luu_roi.py)
# ====================================================================

def doc_roi_da_luu(duong_dan_file_roi):
    with open(duong_dan_file_roi, "r", encoding="utf-8") as f:
        du_lieu = json.load(f)
    rois = [tuple(r) for r in du_lieu["rois"]]
    kich_thuoc_anh_mau = tuple(du_lieu["kich_thuoc_anh_mau"])
    return rois, kich_thuoc_anh_mau


def cat_cac_vung_roi(crop_img, rois):
    h, w = crop_img.shape[:2]
    ket_qua = []
    for i, (x, y, ww, hh) in enumerate(rois, start=1):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + ww, w), min(y + hh, h)
        if x1 >= x2 or y1 >= y2:
            print(f"  Vùng ROI {i} nằm ngoài ảnh crop ({w}x{h}), bỏ qua.")
            continue
        ket_qua.append((i, crop_img[y1:y2, x1:x2]))
    return ket_qua


# ====================================================================
# PHÂN LOẠI + GRAD-CAM
# ====================================================================

def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        embedding_dim = model.last_channel
        model.classifier[1] = nn.Linear(embedding_dim, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        embedding_dim = 960
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        embedding_dim = 576
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model, backbone_name


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    backbone_name = checkpoint.get("backbone_name", "mobilenet_v2")

    model, backbone_name = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Đã load model: {model_path}")
    print(f"Backbone: {backbone_name} | Class: {class_names}\n")
    return model, class_names, backbone_name


def get_target_layer(model, backbone_name):
    return model.features[-1]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output


def overlay_heatmap(img_bgr, cam, alpha=HEATMAP_ALPHA):
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return overlay


def phan_loai_va_gradcam(crop_bgr, model, class_names, gradcam):
    """Nhận 1 vùng ROI (đã cắt sẵn, BGR) -> chạy model phân loại + Grad-CAM,
    trả về (ảnh overlay heatmap, nhãn dự đoán, độ tin cậy)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    if TARGET_CLASS is not None:
        class_idx = class_names.index(TARGET_CLASS)
    else:
        with torch.no_grad():
            probs_tmp = F.softmax(model(input_tensor), dim=1)
        class_idx = probs_tmp.argmax(dim=1).item()

    cam, output = gradcam.compute(input_tensor, class_idx)
    print("Logits:", output.detach().cpu().numpy())
    probs = F.softmax(output, dim=1)[0]
    pred_class = class_names[class_idx]
    confidence = probs[class_idx].item()

    overlay = overlay_heatmap(crop_bgr, cam)
    label = f"{pred_class} ({confidence*100:.1f}%)"
    cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

    return overlay, pred_class, confidence


# ====================================================================
# PHẦN CAMERA
# ====================================================================

def main():
    os.makedirs(THU_MUC_ANH_CROP, exist_ok=True)
    os.makedirs(THU_MUC_ANH_ROI_CON, exist_ok=True)

    out_w, out_h = kich_thuoc_output_chuan()
    print(f"Khung crop chuẩn: {KHUNG_CHUAN_W}x{KHUNG_CHUAN_H} "
          f"-> ảnh output sau khi áp mask + cắt sát: {out_w}x{out_h}")

    print("Đang chuẩn bị ảnh chuẩn (dự phòng cho nhánh SIFT)...")
    anh_chuan_info = chuan_bi_anh_chuan(DUONG_DAN_ANH_CHUAN_CO_DINH)
    print(f"  Ảnh chuẩn có tỉ lệ lõm {anh_chuan_info['ti_le_lom']:.1f}\n")

    print("Đang đọc các vùng ROI con đã lưu...")
    rois_con, kich_thuoc_anh_mau_roi = doc_roi_da_luu(DUONG_DAN_FILE_ROI)
    print(f"Đã đọc {len(rois_con)} vùng ROI con từ: {DUONG_DAN_FILE_ROI}")
    if (kich_thuoc_anh_mau_roi[0] != out_w or kich_thuoc_anh_mau_roi[1] != out_h):
        print(f"  CẢNH BÁO: ảnh mẫu lúc vẽ ROI có kích thước {kich_thuoc_anh_mau_roi}, "
              f"khác kích thước ảnh crop hiện tại ({out_w}x{out_h}). "
              f"Tọa độ vùng ROI con SẼ BỊ LỆCH - hãy vẽ lại ROI trên ảnh crop mới.")
    print()

    print("Đang load model phân loại...")
    model, class_names, backbone_name = load_model(MODEL_PATH)
    target_layer = get_target_layer(model, backbone_name)
    gradcam = GradCAM(model, target_layer)

    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("SDK Version:", hex(SDKVersion))

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Không tìm thấy camera nào! Mã lỗi:", ret)
        sys.exit()

    print(f"Tìm thấy {deviceList.nDeviceNum} camera")

    target_index = 0
    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            print(f"[{i}] GigE Camera: {strModeName}")
            if "MV-CS200" in strModeName:
                target_index = i
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            print(f"[{i}] USB Camera: {strModeName}")

    stDeviceInfo = cast(deviceList.pDeviceInfo[target_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print("Tạo handle lỗi:", ret)
        sys.exit()

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Mở camera lỗi:", ret)
        cam.MV_CC_DestroyHandle()
        sys.exit()

    if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing lỗi:", ret)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * nPayloadSize)()

    print("\nĐang xem trực tiếp... Nhấn 'c' để chụp + crop + phân loại, 'q' để thoát.")

    so_thu_tu = 0

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                print("Không lấy được frame, mã lỗi:", ret)
                continue

            raw = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
            raw = raw.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            pixel_type = stFrameInfo.enPixelType

            if pixel_type == PixelType_Gvsp_BayerRG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
            elif pixel_type == PixelType_Gvsp_BayerGB8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)
            elif pixel_type == PixelType_Gvsp_BayerGR8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GR2BGR)
            elif pixel_type == PixelType_Gvsp_BayerBG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)
            elif pixel_type == PixelType_Gvsp_Mono8:
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                print("Pixel format chưa được xử lý:", pixel_type)
                continue

            img_show = cv2.resize(img, (960, 640))
            cv2.imshow("Hikrobot Camera - nhan 'c' de chup, 'q' de thoat", img_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                so_thu_tu += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                ten_file = f"capture_{timestamp}_{so_thu_tu:03d}"

                crop, thong_bao = crop_va_dong_bo_huong(img, anh_chuan_info)
                if crop is None:
                    print(f"\n[Chụp #{so_thu_tu}] KHÔNG crop được: {thong_bao}")
                    continue

                print(f"\n[Chụp #{so_thu_tu}] {thong_bao}")
                cv2.imshow("Anh da crop", resize_de_hien_thi(crop))

                # --- Lưu ảnh crop chính ---
                duong_dan_crop = os.path.join(THU_MUC_ANH_CROP, f"{ten_file}_crop.png")
                cv2.imwrite(duong_dan_crop, crop)
                print(f"  -> Đã lưu ảnh crop: {duong_dan_crop}")

                # --- Cắt các vùng ROI con + lưu từng vùng ---
                cac_vung = cat_cac_vung_roi(crop, rois_con)
                vung_dict = {}
                for idx, vung in cac_vung:
                    vung_dict[idx] = vung
                    cv2.imshow(f"Vung ROI {idx}", resize_de_hien_thi(vung, max_dim=500))

                    duong_dan_roi = os.path.join(
                        THU_MUC_ANH_ROI_CON, f"{ten_file}_roi{idx}.png")
                    cv2.imwrite(duong_dan_roi, vung)
                    print(f"  -> Đã lưu vùng ROI {idx}: {duong_dan_roi}")

                # --- Phân loại + Grad-CAM trên vùng ROI được chỉ định ---
                if CHI_SO_ROI_PHAN_LOAI in vung_dict:
                    overlay, pred_class, confidence = phan_loai_va_gradcam(
                        vung_dict[CHI_SO_ROI_PHAN_LOAI], model, class_names, gradcam)
                    print(f"  -> Phân loại (ROI {CHI_SO_ROI_PHAN_LOAI}): "
                          f"{pred_class} ({confidence*100:.1f}%)")
                    cv2.imshow("Ket qua phan loai + GradCAM", resize_de_hien_thi(overlay, max_dim=600))
                else:
                    print(f"  -> Không có ROI {CHI_SO_ROI_PHAN_LOAI} để phân loại.")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("\nĐã đóng camera.")


if __name__ == "__main__":
    main()