"""
Chụp ảnh trực tiếp từ camera Hikrobot (nhấn 'c' để chụp, 'q' để thoát).

Mỗi lần nhấn 'c':
    1. Cắt ROI vật thể: tìm contour -> làm thẳng -> ÉP USB VỀ BÊN PHẢI bằng
       hình học (SIFT chỉ là nhánh dự phòng) -> cắt ROI có padding -> ÁP MASK
       HÌNH HỌC CỐ ĐỊNH (nền đen tuyệt đối) -> cắt sát mask.
       Hiển thị ("Anh da crop") và LƯU vào THU_MUC_ANH_CROP.
    2. CHIA ẢNH CROP THÀNH 5 PART theo tỉ lệ chiều rộng (RATIOS), lưu mỗi
       part vào thư mục con riêng trong THU_MUC_ANH_PART.
    3. Mỗi part chạy qua MODEL ONNX RIÊNG của nó
       (model14/partN/fewshot_wide_resnet50_2_partN.onnx) -> in kết quả từng
       part + kết luận chung (chỉ cần 1 part NG là cả con hàng NG).
    4. Vẽ HEATMAP cho từng part để xem model nhìn vào đâu.

HỖ TRỢ CẢ 2 ĐỊNH DẠNG MODEL:
    - .pt  (PyTorch, do 07_few_shot_pipeline_multi.py train ra) -> dùng
      GRAD-CAM THẬT vì torch có gradient.
    - .onnx -> ONNX Runtime chỉ chạy forward, không có gradient, nên thay
      bằng CAM cổ điển (trọng số lớp Linear cuối, 1 forward) hoặc occlusion
      sensitivity (che từng ô, đo độ tụt xác suất).
Script tự nhận định dạng theo đuôi file, không cần khai báo.

QUAN TRỌNG - phải khớp với script tạo tập training:
    - PAD_ROI_NGANG / PAD_ROI_DOC / THAM SỐ MASK HÌNH HỌC / CAT_SAT_MASK /
      LE_TRONG phải GIỐNG HỆT bên 01_tao_template_v7.py.
    - KHUNG_CHUAN_W/H phải bằng "Kích thước ROI chuẩn chung (trung vị)" mà
      script tạo template in ra ở lượt 1.
    - RATIOS phải GIỐNG HỆT crop_batch.py lúc cắt tập training, nếu không
      part đưa vào model sẽ không cùng vùng ảnh với lúc train.
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
# 4. Import ONNX Runtime cho phần phân loại
#    (pip install onnxruntime  hoặc  onnxruntime-gpu)
# ============================================================
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models as tv_models
except ImportError:
    torch = None


# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
# Ảnh crop chính (đã cắt + xoay + xóa nền).
THU_MUC_ANH_CROP = r"D:\TongHop\RTC Technologi\PCB\crop7\NG"

# Các part cắt từ ảnh crop chính. Mỗi part vào 1 thư mục con riêng
# (THU_MUC_ANH_PART\part1, ...\part2, ...) - đúng cấu trúc crop_batch.py.
THU_MUC_ANH_PART = r"D:\TongHop\RTC Technologi\PCB\crop7\NG_crop"

# Ảnh CHUẨN CỐ ĐỊNH - CHỈ dùng cho nhánh dự phòng SIFT.
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\crop5\train\OK\Image_20260723101631285_roi.png"

# Thư mục gốc chứa model của từng part.
THU_MUC_MODEL = r"D:\TongHop\RTC Technologi\PCB\model\model18"

# Mẫu đường dẫn model: {ten} sẽ được thay bằng "part1", "part2", ...
# Có thể khai báo NHIỀU mẫu; script thử lần lượt, mẫu nào có file thì dùng.
# Nhận cả .pt (PyTorch) lẫn .onnx - tự nhận theo đuôi file.
MAU_DUONG_DAN_MODEL = [
    r"{ten}\edge_classifier_fewshot_mobilenet_v3_large_{ten}.pt",   # từ 07_few_shot_pipeline_multi.py
    r"{ten}\fewshot_wide_resnet50_2_{ten}.onnx",
]

# Tên các class theo ĐÚNG THỨ TỰ output của model. Nếu file ONNX có sẵn
# metadata "class_names" thì script ưu tiên dùng metadata đó.
CLASS_NAMES_MAC_DINH = ["NG", "OK"]

# Tên class được coi là hàng lỗi (dùng để kết luận chung).
TEN_CLASS_NG = "NG"

# Ngưỡng tin cậy tối thiểu; dưới ngưỡng sẽ đánh dấu "?" để soi lại bằng mắt.
NGUONG_TIN_CAY = 0.60

LUU_ANH_PART = True       # có lưu ảnh part xuống đĩa không
HIEN_THI_PART = True      # có mở cửa sổ xem từng part không

# ==== HEATMAP (xem model đang nhìn vào đâu) ====
# ONNX Runtime chỉ chạy forward, KHÔNG có gradient -> không làm Grad-CAM
# được. Thay bằng 2 cách không cần gradient:
#   "cam"       : CAM cổ điển - dùng trọng số lớp Linear cuối nhân với
#                 feature map. Chỉ áp dụng được khi model kết thúc bằng
#                 GlobalAveragePool -> (Flatten) -> Gemm/MatMul (resnet,
#                 wide_resnet, mobilenet... đều vậy). Nhanh: 1 forward.
#   "occlusion" : che lần lượt từng ô rồi đo độ tụt xác suất. Chạy được với
#                 MỌI kiến trúc nhưng tốn OCCLUSION_LUOI[0]*[1] lần forward.
#   "auto"      : ưu tiên CAM, model nào không hợp thì tự chuyển occlusion.
#   "tat"       : không vẽ heatmap.
#   "gradcam"   : Grad-CAM thật - CHỈ dùng được với model .pt (có gradient).
KIEU_HEATMAP = "auto"
HEATMAP_ALPHA = 0.45
OCCLUSION_LUOI = (6, 6)   # (số hàng, số cột) ô che
OCCLUSION_MAU = 0         # giá trị pixel dùng để che (0 = đen, khớp nền)
LUU_HEATMAP = False       # lưu ảnh heatmap xuống <THU_MUC_ANH_PART>\_heatmap
# =====================================================================

# ==== TỈ LỆ CHIA PART (theo % chiều rộng ảnh crop, 0.0 -> 1.0) ====
# PHẢI GIỐNG HỆT RATIOS trong crop_batch.py lúc tạo tập training.
RATIOS = [
    ("part1", 0.00, 0.33),
    ("part2", 0.33, 0.66),
    ("part3", 0.66, 1.00),
    ("part4", 0.33 / 2, (0.66 - 0.33) / 2 + 0.33),
    ("part5", (0.66 - 0.33) / 2 + 0.33, (1 - 0.66) / 2 + 0.66),
]

# ==== Tiền xử lý ảnh trước khi vào model (phải giống lúc train) ====
KICH_THUOC_INPUT = 224          # dùng khi model khai báo input động
CHUAN_HOA_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
CHUAN_HOA_STD = np.array([0.229, 0.224, 0.225], np.float32)

# ========= THAM SỐ CROP + MASK (PHẢI GIỐNG SCRIPT TẠO TEMPLATE) =========
KHUNG_CHUAN_W = 3187
KHUNG_CHUAN_H = 1777

PAD_ROI_NGANG = 300
PAD_ROI_DOC = 60

NGUONG_TI_LE_LOM = 2.0
AP_MASK_HINH_HOC = True

# --- Thân board: chữ nhật bo góc (tỉ lệ bề rộng W và chiều cao H)
BOARD_X1 = 0.0948    # (302/3187)
BOARD_X2 = 0.9049    # (2884/3187)
BOARD_Y1 = 0.0360    # (64/1777)
BOARD_Y2 = 0.9657    # (1716/1777)
BOARD_BO_GOC = 0.0141  # (25/1777)

# --- Khối USB Type-C nhô ra bên PHẢI
USB_X1 = 0.8472      # (2700/3187)
USB_X2 = 0.9576      # (3052/3187)
USB_Y1 = 0.2746      # (488/1777)
USB_Y2 = 0.6989      # (1242/1777)
USB_BO_GOC = 0.0113  # (20/1777)

# --- Vết khuyết (tab bẻ) khoét ĐEN ở cạnh TRÁI. Đặt 0 để tắt.
KHUYET_X2 = 0.1403   # (447/3187)
KHUYET_Y1 = 0.1435   # (255/1777)
KHUYET_Y2 = 0.8694   # (1545/1777)
KHUYET_BO_GOC = 0.0535  # (95/1777)

NOI_MASK = 0
LAM_MUOT_BIEN = 5
CAT_SAT_MASK = True
LE_TRONG = 15
# =====================================================================

# Dropout của head lúc train (phải khớp HEAD_DROPOUT trong script train thì
# load_state_dict mới đúng cấu trúc; dropout không ảnh hưởng lúc eval).
HEAD_DROPOUT_KHI_TRAIN = 0.3

DEVICE_TORCH = None
if torch is not None:
    DEVICE_TORCH = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000
MIN_INLIERS_HUONG = 15
PAD_MASK_SIFT = 60


# ====================================================================
# XỬ LÝ ẢNH - CROP VẬT THỂ CHÍNH (đồng bộ với script tạo template)
# ====================================================================

def tim_contour_tu_dong(img, target_w=1200):
    """Threshold Saturation + lọc thành phần liên thông lớn nhất + lấp lỗ."""
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
    Phía lắp USB lõm rất sâu vì contour không bắt được vỏ kim loại."""
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
    """Trả về (M_align, roi, ti_le_lom); đã ép USB về bên PHẢI."""
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
    """Mask cố định cho khung crop W x H (USB nằm bên PHẢI)."""
    mask = np.zeros((H, W), np.uint8)
    _chu_nhat_bo_goc(mask, int(BOARD_X1 * W), int(BOARD_Y1 * H),
                     int(BOARD_X2 * W), int(BOARD_Y2 * H), int(BOARD_BO_GOC * H), 255)
    _chu_nhat_bo_goc(mask, int(USB_X1 * W), int(USB_Y1 * H),
                     int(USB_X2 * W), int(USB_Y2 * H), int(USB_BO_GOC * H), 255)
    if KHUYET_X2 > 0:
        _chu_nhat_bo_goc(mask, 0, int(KHUYET_Y1 * H),
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
    """Dữ liệu SIFT của ảnh chuẩn - CHỈ dùng cho nhánh dự phòng."""
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
    return {"kp_chuan": kp_chuan, "des_chuan": des_chuan,
            "dir_chuan": dir_chuan, "ti_le_lom": ti_le_lom_chuan}


def crop_va_dong_bo_huong(img, anh_chuan_info):
    """Trả về (ảnh crop đã xóa nền, thông báo) hoặc (None, lý do lỗi)."""
    contour = tim_contour_tu_dong(img)
    if contour is None:
        return None, "Không tìm thấy vật thể trong ảnh vừa chụp."

    m_align, roi, ti_le_lom = tinh_m_align_va_roi(contour)
    x, y, w, h = roi
    warped = cv2.warpAffine(img, m_align, (img.shape[1], img.shape[0]))
    crop = cat_an_toan(warped, x, y, w, h)

    if crop.shape[1] != KHUNG_CHUAN_W or crop.shape[0] != KHUNG_CHUAN_H:
        crop = cv2.resize(crop, (KHUNG_CHUAN_W, KHUNG_CHUAN_H))

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
            thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) và SIFT cũng không đủ "
                         f"tin cậy ({so_inlier} match) - GIỮ NGUYÊN, KIỂM TRA BẰNG MẮT.")
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
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT xác nhận giữ "
                             f"nguyên (lệch {goc_do:.0f}°, {so_inlier} inlier).")

    if AP_MASK_HINH_HOC:
        mask = tao_mask_hinh_hoc(crop.shape[1], crop.shape[0])
        crop[mask == 0] = 0
        if CAT_SAT_MASK:
            bx, by, bw, bh = cv2.boundingRect(mask)
            crop = crop[max(0, by - LE_TRONG):min(crop.shape[0], by + bh + LE_TRONG),
                        max(0, bx - LE_TRONG):min(crop.shape[1], bx + bw + LE_TRONG)]

    return crop, thong_bao


def kich_thuoc_output_chuan():
    """Kích thước ảnh crop cuối - suy ra từ mask, không cần chụp thử."""
    mask = tao_mask_hinh_hoc(KHUNG_CHUAN_W, KHUNG_CHUAN_H)
    if not AP_MASK_HINH_HOC or not CAT_SAT_MASK:
        return KHUNG_CHUAN_W, KHUNG_CHUAN_H
    bx, by, bw, bh = cv2.boundingRect(mask)
    return (min(KHUNG_CHUAN_W, bx + bw + LE_TRONG) - max(0, bx - LE_TRONG),
            min(KHUNG_CHUAN_H, by + bh + LE_TRONG) - max(0, by - LE_TRONG))


def resize_de_hien_thi(img, max_dim=900):
    h, w = img.shape[:2]
    ti_le = min(1.0, max_dim / max(h, w))
    if ti_le >= 1.0:
        return img
    return cv2.resize(img, (int(w * ti_le), int(h * ti_le)))


# ====================================================================
# CHIA ẢNH CROP THÀNH CÁC PART THEO TỈ LỆ CHIỀU RỘNG
# (logic cắt giống hệt crop_batch.py để part khớp với lúc train)
# ====================================================================

def chia_cac_part(crop_img, ratios=RATIOS):
    """Trả về list [(ten_part, ảnh part), ...] theo tỉ lệ chiều rộng."""
    h, w = crop_img.shape[:2]
    ket_qua = []
    for ten, start, end in ratios:
        x0 = max(0, min(int(round(w * start)), w))
        x1 = max(0, min(int(round(w * end)), w))
        if x1 <= x0:
            print(f"  Part '{ten}' có tỉ lệ không hợp lệ ({start}-{end}), bỏ qua.")
            continue
        ket_qua.append((ten, crop_img[0:h, x0:x1]))
    return ket_qua


def luu_cac_part(cac_part, thu_muc_goc, ten_file_goc):
    """Lưu mỗi part vào thư mục con riêng: thu_muc_goc\\partN\\<ten>_partN.png"""
    duong_dan = {}
    for ten, anh in cac_part:
        thu_muc_part = os.path.join(thu_muc_goc, ten)
        os.makedirs(thu_muc_part, exist_ok=True)
        p = os.path.join(thu_muc_part, f"{ten_file_goc}_{ten}.png")
        cv2.imwrite(p, anh)
        duong_dan[ten] = p
    return duong_dan


# ====================================================================
# PHÂN LOẠI BẰNG 5 MODEL ONNX RIÊNG CHO TỪNG PART
# ====================================================================

def _lay_class_names(session):
    """Ưu tiên metadata 'class_names' nhúng trong file ONNX; không có thì
    dùng CLASS_NAMES_MAC_DINH."""
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
        for khoa in ("class_names", "classes", "labels"):
            if khoa in meta:
                ten = json.loads(meta[khoa]) if meta[khoa].strip().startswith("[") \
                    else [t.strip() for t in meta[khoa].split(",")]
                if ten:
                    return list(ten), True
    except Exception:
        pass
    return list(CLASS_NAMES_MAC_DINH), False


def _tim_file_model(thu_muc_model, ten):
    """Thử lần lượt các mẫu trong MAU_DUONG_DAN_MODEL, trả về file đầu tiên
    có thật (hoặc None)."""
    mau = MAU_DUONG_DAN_MODEL
    if isinstance(mau, str):
        mau = [mau]
    da_thu = []
    for m in mau:
        p = os.path.join(thu_muc_model, m.format(ten=ten))
        da_thu.append(p)
        if os.path.exists(p):
            return p, da_thu
    return None, da_thu


def _nap_model_torch(duong_dan):
    """Nạp checkpoint .pt do 07_few_shot_pipeline_multi.py lưu ra.
    Checkpoint gồm: model_state, class_names, backbone_name."""
    if torch is None:
        raise RuntimeError("Cần cài PyTorch để dùng model .pt: pip install torch torchvision")

    ckpt = torch.load(duong_dan, map_location=DEVICE_TORCH, weights_only=False)
    class_names = list(ckpt["class_names"])
    backbone_name = ckpt.get("backbone_name", "mobilenet_v3_large")

    if backbone_name == "mobilenet_v2":
        model = tv_models.mobilenet_v2(weights=None)
        embedding_dim = model.last_channel
    elif backbone_name == "mobilenet_v3_large":
        model = tv_models.mobilenet_v3_large(weights=None)
        embedding_dim = 960
    elif backbone_name == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights=None)
        embedding_dim = 576
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    # head y hệt lúc train: Sequential(Dropout, Linear)
    model.classifier = nn.Sequential(nn.Dropout(HEAD_DROPOUT_KHI_TRAIN),
                                     nn.Linear(embedding_dim, len(class_names)))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE_TORCH).eval()

    return {"loai": "torch", "model": model, "class_names": class_names,
            "backbone_name": backbone_name, "kich_thuoc": (KICH_THUOC_INPUT, KICH_THUOC_INPUT),
            "gradcam": GradCAMTorch(model, model.features[-1]),
            "kfold_acc": ckpt.get("kfold_val_acc_mean"), "W_cam": None, "ten_feat": None}


def nap_cac_model(thu_muc_model, ratios=RATIOS):
    """Nạp sẵn 1 model cho mỗi part (.pt hoặc .onnx, tự nhận theo đuôi file).
    Part nào thiếu model thì bỏ qua (vẫn cắt + lưu ảnh part đó)."""
    uu_tien = None
    if ort is not None:
        providers = ort.get_available_providers()
        uu_tien = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
    if torch is not None:
        print(f"PyTorch chạy trên: {DEVICE_TORCH}")

    cac_model = {}
    for ten, _, _ in ratios:
        duong_dan, da_thu = _tim_file_model(thu_muc_model, ten)
        if duong_dan is None:
            print(f"  [{ten}] KHÔNG tìm thấy model -> bỏ qua phân loại part này. Đã thử:")
            for p in da_thu:
                print(f"           {p}")
            continue

        duoi = os.path.splitext(duong_dan)[1].lower()
        if duoi == ".pt":
            m = _nap_model_torch(duong_dan)
            tu_metadata = True
            kieu_hm = "Grad-CAM (thật)" if KIEU_HEATMAP != "tat" else "tắt"
            if KIEU_HEATMAP == "occlusion":
                kieu_hm = "occlusion"
            them = ""
            if m.get("kfold_acc") is not None:
                them = f" | K-Fold acc lúc train: {m['kfold_acc']:.1%}"
        elif duoi == ".onnx":
            if ort is None:
                print(f"  [{ten}] Có file .onnx nhưng chưa cài onnxruntime -> bỏ qua.")
                continue
            m = _chuan_bi_session_cam(duong_dan, uu_tien)
            m["loai"] = "onnx"
            m["class_names"], tu_metadata = _lay_class_names(m["session"])
            if KIEU_HEATMAP == "tat":
                kieu_hm = "tắt"
            elif KIEU_HEATMAP in ("cam", "auto") and m["W_cam"] is not None:
                kieu_hm = "CAM"
            elif KIEU_HEATMAP in ("occlusion", "auto"):
                kieu_hm = "occlusion"
            else:
                kieu_hm = "tắt"
            them = ""
            if KIEU_HEATMAP == "gradcam":
                print(f"  [{ten}] CẢNH BÁO: .onnx KHÔNG làm Grad-CAM được (không có "
                      f"gradient). Dùng KIEU_HEATMAP = \"auto\" để tự chuyển sang CAM.")
        else:
            print(f"  [{ten}] Đuôi file không hỗ trợ: {duong_dan}")
            continue

        cac_model[ten] = m
        ww, hh = m["kich_thuoc"]
        nguon = "checkpoint/metadata" if tu_metadata else "CLASS_NAMES_MAC_DINH"
        print(f"  [{ten}] {os.path.basename(duong_dan)} [{duoi[1:]}] | input {ww}x{hh} | "
              f"class {m['class_names']} (từ {nguon}) | heatmap: {kieu_hm}{them}")

    if not cac_model:
        raise RuntimeError(f"Không nạp được model nào trong: {thu_muc_model}")
    return cac_model


def _tien_xu_ly(anh_bgr, kich_thuoc):
    """BGR uint8 -> tensor NCHW float32 đã chuẩn hóa ImageNet."""
    img = cv2.resize(anh_bgr, kich_thuoc, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - CHUAN_HOA_MEAN) / CHUAN_HOA_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _chay_model(m, anh):
    """Chạy 1 ảnh part qua model. Trả về (prob, ctx) - ctx là dữ liệu phụ để
    tính heatmap sau (outputs của ONNX, hoặc tensor đầu vào của torch)."""
    x = _tien_xu_ly(anh, m["kich_thuoc"])
    if m["loai"] == "torch":
        tensor = torch.from_numpy(x).to(DEVICE_TORCH)
        with torch.no_grad():
            out = m["model"](tensor)
        prob = F.softmax(out, dim=1)[0].cpu().numpy()
        return prob, tensor
    outputs = m["session"].run(None, {m["ten_input"]: x})
    vec = np.asarray(outputs[0]).reshape(-1)
    prob = vec if (vec.min() >= 0 and abs(vec.sum() - 1.0) < 1e-3) else _softmax(vec)
    return prob, outputs


def _tinh_heatmap(m, ctx, anh, class_idx):
    """Trả về (cam, tên kiểu) hoặc (None, None) nếu tắt/không làm được.
    - model .pt   -> Grad-CAM thật (trừ khi ép KIEU_HEATMAP = "occlusion")
    - model .onnx -> CAM cổ điển, không hợp thì occlusion"""
    if KIEU_HEATMAP == "tat":
        return None, None

    if m["loai"] == "torch":
        if KIEU_HEATMAP == "occlusion":
            return tinh_occlusion(m, anh, class_idx), "occlusion"
        return m["gradcam"].tinh(ctx, class_idx), "Grad-CAM"

    if KIEU_HEATMAP in ("cam", "auto", "gradcam"):
        cam = tinh_cam(m, ctx, class_idx)
        if cam is not None:
            return cam, "CAM"
        if KIEU_HEATMAP == "cam":
            return None, None
    if KIEU_HEATMAP in ("occlusion", "auto", "gradcam"):
        return tinh_occlusion(m, anh, class_idx), "occlusion"
    return None, None


def phan_loai_cac_part(cac_part, cac_model):
    """Chạy từng part qua model riêng của nó + tính heatmap.
    Trả về (list kết quả, kết luận chung, có part nào không chắc chắn).
    Mỗi phần tử kết quả: (tên part, nhãn, độ tin cậy, lỗi, ảnh heatmap)."""
    ket_qua, co_ng, co_khong_chac = [], False, False
    for ten, anh in cac_part:
        if ten not in cac_model:
            ket_qua.append((ten, None, 0.0, "thiếu model", None))
            co_khong_chac = True
            continue
        m = cac_model[ten]
        prob, ctx = _chay_model(m, anh)
        idx = int(np.argmax(prob))
        nhan = m["class_names"][idx] if idx < len(m["class_names"]) else f"class_{idx}"
        do_tin = float(prob[idx])
        if nhan == TEN_CLASS_NG:
            co_ng = True
        if do_tin < NGUONG_TIN_CAY:
            co_khong_chac = True

        cam, kieu = _tinh_heatmap(m, ctx, anh, idx)
        anh_hm = phu_heatmap(anh, cam) if cam is not None else None
        ket_qua.append((ten, nhan, do_tin, None, anh_hm))

    ket_luan = TEN_CLASS_NG if co_ng else "OK"
    return ket_qua, ket_luan, co_khong_chac


def ve_nhan_len_part(anh, ten, nhan, do_tin):
    """Vẽ nhãn lên ảnh part để xem nhanh trên cửa sổ hiển thị."""
    ra = anh.copy()
    mau = (0, 0, 255) if nhan == TEN_CLASS_NG else (0, 200, 0)
    if nhan is None:
        mau, chu = (0, 165, 255), f"{ten}: thieu model"
    else:
        chu = f"{ten}: {nhan} {do_tin*100:.0f}%"
    co_chu = max(1.2, ra.shape[1] / 600)*1.7
    cv2.rectangle(ra, (0, 0), (ra.shape[1] - 1, ra.shape[0] - 1), mau, max(2, ra.shape[1] // 200))
    cv2.putText(ra, chu, (12, int(40 * co_chu)), cv2.FONT_HERSHEY_SIMPLEX,
                co_chu, mau, max(1, int(co_chu * 1.5)))
    return ra



# ====================================================================
# HEATMAP KHÔNG CẦN GRADIENT (CAM cổ điển + occlusion sensitivity)
# ====================================================================

def _tim_diem_cam(duong_dan_onnx):
    """Dò xem model có dạng ... -> GlobalAveragePool/ReduceMean -> (Flatten)
    -> Gemm/MatMul -> output hay không.
    Trả về (ten_tensor_feature_map, ma_tran_trong_so W) hoặc None."""
    try:
        import onnx
    except ImportError:
        return None

    model = onnx.load(duong_dan_onnx)
    g = model.graph
    tao_ra = {}
    for n in g.node:
        for o in n.output:
            tao_ra[o] = n
    khoi_tao = {t.name: t for t in g.initializer}

    ten_out = g.output[0].name
    node = tao_ra.get(ten_out)
    if node is None:
        return None

    # Add (bias) -> lùi về MatMul/Gemm
    if node.op_type == "Add":
        for i in node.input:
            n2 = tao_ra.get(i)
            if n2 is not None and n2.op_type in ("Gemm", "MatMul"):
                node = n2
                break
    if node.op_type not in ("Gemm", "MatMul"):
        return None

    ten_W = next((i for i in node.input if i in khoi_tao), None)
    ten_data = next((i for i in node.input if i not in khoi_tao), None)
    if ten_W is None or ten_data is None:
        return None

    from onnx import numpy_helper
    W = numpy_helper.to_array(khoi_tao[ten_W])
    if W.ndim != 2:
        return None

    # lùi tiếp qua Flatten/Reshape/Squeeze để tới node pooling
    n = tao_ra.get(ten_data)
    for _ in range(4):
        if n is None:
            return None
        if n.op_type in ("GlobalAveragePool", "ReduceMean"):
            return n.input[0], W
        if n.op_type in ("Flatten", "Reshape", "Squeeze", "Identity", "Transpose"):
            n = tao_ra.get(n.input[0])
            continue
        return None
    return None


def _chuan_bi_session_cam(duong_dan_onnx, providers):
    """Trả về dict gồm session (đã thêm feature map vào output nếu làm được
    CAM), tên input, kích thước input, class names, và W cho CAM."""
    diem = _tim_diem_cam(duong_dan_onnx)
    W_cam, ten_feat = None, None

    if diem is not None:
        ten_feat, W_cam = diem
        import onnx
        model = onnx.load(duong_dan_onnx)
        if ten_feat not in [o.name for o in model.graph.output]:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(ten_feat))
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    else:
        session = ort.InferenceSession(duong_dan_onnx, providers=providers)

    inp = session.get_inputs()[0]
    hh = inp.shape[2] if isinstance(inp.shape[2], int) else 224
    ww = inp.shape[3] if isinstance(inp.shape[3], int) else 224
    return {"session": session, "ten_input": inp.name, "kich_thuoc": (ww, hh),
            "ten_feat": ten_feat, "W_cam": W_cam,
            "ten_output": [o.name for o in session.get_outputs()]}


def tinh_cam(m, outputs, class_idx):
    """CAM = tổng có trọng số của các kênh feature map. Trả None nếu model
    không thuộc dạng GAP -> Linear."""
    if m["W_cam"] is None or m["ten_feat"] is None:
        return None
    idx = m["ten_output"].index(m["ten_feat"])
    A = np.asarray(outputs[idx])
    if A.ndim != 4:
        return None
    A = A[0]                      # [C, h, w]
    C = A.shape[0]
    W = m["W_cam"]
    if W.shape[0] == C:           # [C, num_class] (MatMul)
        w = W[:, class_idx]
    elif W.shape[1] == C:         # [num_class, C] (Gemm transB=1)
        w = W[class_idx, :]
    else:
        return None
    cam = np.tensordot(w, A, axes=(0, 0))
    cam = np.maximum(cam, 0)
    return cam / cam.max() if cam.max() > 0 else cam


def tinh_occlusion(m, anh_bgr, class_idx, luoi=OCCLUSION_LUOI):
    """Che lần lượt từng ô rồi đo độ tụt xác suất -> heatmap. Chậm hơn CAM
    (luoi[0]*luoi[1] lần forward) nhưng chạy được với mọi kiến trúc."""
    nh, nw = luoi
    H, W = anh_bgr.shape[:2]
    goc = _chay(m, _tien_xu_ly(anh_bgr, m["kich_thuoc"]))[class_idx]
    heat = np.zeros((nh, nw), np.float32)
    bh, bw = int(np.ceil(H / nh)), int(np.ceil(W / nw))
    for i in range(nh):
        for j in range(nw):
            tmp = anh_bgr.copy()
            tmp[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = OCCLUSION_MAU
            heat[i, j] = goc - _chay(m, _tien_xu_ly(tmp, m["kich_thuoc"]))[class_idx]
    heat = np.maximum(heat, 0)
    return heat / heat.max() if heat.max() > 0 else heat


def _chay(m, tensor):
    """Chạy 1 tensor đã tiền xử lý -> vector xác suất (dùng cho occlusion)."""
    if m["loai"] == "torch":
        with torch.no_grad():
            out = m["model"](torch.from_numpy(tensor).to(DEVICE_TORCH))
        return F.softmax(out, dim=1)[0].cpu().numpy()
    out = m["session"].run(None, {m["ten_input"]: tensor})
    vec = np.asarray(out[0]).reshape(-1)
    return vec if (vec.min() >= 0 and abs(vec.sum() - 1) < 1e-3) else _softmax(vec)


class GradCAMTorch:
    """Grad-CAM THẬT cho model .pt: hook vào tầng conv cuối để lấy activation
    lúc forward và gradient lúc backward, rồi lấy trung bình gradient theo
    không gian làm trọng số cho từng kênh."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._luu_activation)
        target_layer.register_full_backward_hook(self._luu_gradient)

    def _luu_activation(self, module, inp, out):
        self.activations = out.detach()

    def _luu_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def tinh(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            return None
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def phu_heatmap(anh_bgr, cam, alpha=HEATMAP_ALPHA):
    H, W = anh_bgr.shape[:2]
    cam_rs = cv2.resize(cam.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    cam_rs = np.clip(cam_rs, 0, 1)
    heat = cv2.applyColorMap(np.uint8(255 * cam_rs), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, anh_bgr, 1 - alpha, 0)

# ====================================================================
# PHẦN CAMERA
# ====================================================================

def main():
    os.makedirs(THU_MUC_ANH_CROP, exist_ok=True)
    os.makedirs(THU_MUC_ANH_PART, exist_ok=True)

    out_w, out_h = kich_thuoc_output_chuan()
    print(f"Khung crop chuẩn: {KHUNG_CHUAN_W}x{KHUNG_CHUAN_H} "
          f"-> ảnh crop sau khi áp mask + cắt sát: {out_w}x{out_h}")
    print("Kích thước từng part (theo RATIOS):")
    for ten, start, end in RATIOS:
        x0, x1 = int(round(out_w * start)), int(round(out_w * end))
        print(f"  {ten}: x {x0} -> {x1}  ({x1 - x0}x{out_h})")
    print()

    print("Đang chuẩn bị ảnh chuẩn (dự phòng cho nhánh SIFT)...")
    anh_chuan_info = chuan_bi_anh_chuan(DUONG_DAN_ANH_CHUAN_CO_DINH)
    print(f"  Ảnh chuẩn có tỉ lệ lõm {anh_chuan_info['ti_le_lom']:.1f}\n")

    print("Đang nạp model cho từng part (.pt hoặc .onnx)...")
    cac_model = nap_cac_model(THU_MUC_MODEL)
    print()

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

                duong_dan_crop = os.path.join(THU_MUC_ANH_CROP, f"{ten_file}_crop.png")
                cv2.imwrite(duong_dan_crop, crop)
                print(f"  -> Đã lưu ảnh crop: {duong_dan_crop}")

                # --- Chia part + lưu ---
                cac_part = chia_cac_part(crop)
                if LUU_ANH_PART:
                    dd = luu_cac_part(cac_part, THU_MUC_ANH_PART, ten_file)
                    print(f"  -> Đã lưu {len(dd)} part vào: {THU_MUC_ANH_PART}")

                # --- Phân loại từng part bằng model riêng ---
                ket_qua, ket_luan, co_khong_chac = phan_loai_cac_part(cac_part, cac_model)
                anh_theo_ten = dict(cac_part)
                for ten, nhan, do_tin, loi, anh_hm in ket_qua:
                    if loi:
                        print(f"     {ten}: {loi}")
                    else:
                        canh_bao = "  <-- độ tin cậy thấp" if do_tin < NGUONG_TIN_CAY else ""
                        print(f"     {ten}: {nhan} ({do_tin*100:.1f}%){canh_bao}")

                    # ảnh hiển thị: có heatmap thì dùng bản đã phủ nhiệt
                    nen = anh_hm if anh_hm is not None else anh_theo_ten[ten]
                    if HIEN_THI_PART:
                        cv2.imshow(f"Part {ten}", resize_de_hien_thi(
                            ve_nhan_len_part(nen, ten, nhan, do_tin), max_dim=400))
                    if LUU_HEATMAP and anh_hm is not None:
                        thu_muc_hm = os.path.join(THU_MUC_ANH_PART, "_heatmap", ten)
                        os.makedirs(thu_muc_hm, exist_ok=True)
                        cv2.imwrite(os.path.join(thu_muc_hm, f"{ten_file}_{ten}_hm.png"),
                                    ve_nhan_len_part(nen, ten, nhan, do_tin))

                dau = "!!!" if ket_luan == TEN_CLASS_NG else "==="
                them = " (có part độ tin cậy thấp - nên soi lại)" if co_khong_chac else ""
                print(f"  {dau} KẾT LUẬN: {ket_luan}{them} {dau}")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("\nĐã đóng camera.")


if __name__ == "__main__":
    main()