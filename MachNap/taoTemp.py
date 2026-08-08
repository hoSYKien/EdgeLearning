"""
BƯỚC 1/2 - Tạo NHIỀU "template" (ảnh mẫu) từ 1 thư mục chứa nhiều ảnh chụp
chuẩn của cùng 1 vật thể (nên chụp ở vài điều kiện ánh sáng/góc lóe sáng
khác nhau một chút - càng đa dạng, càng dễ khớp được ảnh mới sau này).

Chỉ cần chạy 1 LẦN DUY NHẤT. Vì camera + nền cố định, bộ template này dùng
lại được cho mọi ảnh chụp sau (vật thể có thể xoay/dịch chuyển vị trí,
nhưng kích thước thật không đổi).

Cách hoạt động (lặp lại cho MỖI ảnh trong thư mục):
1. Tự động tìm contour vật thể (threshold Saturation + lọc thành phần
   liên thông lớn nhất + lấp lỗ).
2. Tính khung ROI chuẩn (đã "làm thẳng" vật thể, không phụ thuộc đoán góc).
3. Trích đặc trưng SIFT quanh vật thể (chịu thay đổi ánh sáng tốt hơn ORB).
4. Gộp tất cả thành 1 DANH SÁCH template, lưu vào template.pkl.

Ở bước 2 (script sau), mỗi ảnh mới sẽ được thử khớp với TỪNG template
trong danh sách này, và kết quả khớp TỐT NHẤT (nhiều điểm khớp nhất) sẽ
được dùng để cắt ảnh.

Cách dùng:
    Bỏ nhiều ảnh mẫu vào 1 thư mục, sửa đường dẫn ở biến
    DUONG_DAN_THU_MUC_ANH_MAU bên dưới, rồi chạy:
        python 01_tao_template.py
"""

import os
import pickle
import cv2
import numpy as np

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_THU_MUC_ANH_MAU = r"D:\TongHop\RTC Technologi\PCB\temp"      # thư mục chứa NHIỀU ảnh mẫu
DUONG_DAN_LUU_TEMPLATE = r"D:\TongHop\RTC Technologi\PCB\crop4\template.pkl"    # nơi lưu bộ template
# =====================================================================

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")
SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000


def trich_dac_trung_sift(img, target_w=SIFT_TARGET_WIDTH, mask_full=None, nfeatures=SIFT_NFEATURES):
    """Trích SIFT trên ảnh đã thu nhỏ (nhanh hơn), rồi quy đổi tọa độ
    keypoint về đúng độ phân giải gốc của ảnh truyền vào."""
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
    for k in kp:
        k.pt = (k.pt[0] / scale, k.pt[1] / scale)
    return kp, des


def tim_contour_tu_dong(img):
    """Pipeline v5: Saturation threshold + lọc thành phần lớn nhất + lấp lỗ."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    blur = cv2.medianBlur(s, 15)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        raise RuntimeError("Không tìm thấy vật thể nào trong ảnh.")
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def tao_1_template(duong_dan_anh: str):
    """Tạo template (dict) từ 1 ảnh mẫu. Trả về None nếu ảnh lỗi."""
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  -> Không đọc được ảnh, bỏ qua.")
        return None

    try:
        contour = tim_contour_tu_dong(img)
    except RuntimeError as e:
        print(f"  -> {e} Bỏ qua ảnh này.")
        return None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect

    M_align = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(contour_aligned.astype(np.int32))

    # *** Ép về cùng 1 quy ước hướng giữa mọi template ***
    # minAreaRect không đảm bảo các template luôn "thẳng" theo cùng 1 chiều
    # (có thể lệch nhau 90°) -> nếu để vậy, ảnh nào khớp trúng template bị
    # lệch sẽ ra kết quả xoay sai hướng so với các ảnh khác. Quy ước chung:
    # chiều ngang (w) luôn LỚN HƠN chiều cao (h).
    if h_roi > w_roi:
        (cx2, cy2) = (x_roi + w_roi / 2, y_roi + h_roi / 2)
        M_xoay_them = cv2.getRotationMatrix2D((cx2, cy2), 90, 1.0)
        M_align_3x3 = np.vstack([M_align, [0, 0, 1]])
        M_xoay_them_3x3 = np.vstack([M_xoay_them, [0, 0, 1]])
        M_align = (M_xoay_them_3x3 @ M_align_3x3)[:2, :]

        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(contour_aligned.astype(np.int32))

    x, y, w, h = cv2.boundingRect(contour)
    pad = 60
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    feat_mask = np.zeros(img.shape[:2], np.uint8)
    feat_mask[y0:y1, x0:x1] = 255

    kp, des = trich_dac_trung_sift(img, mask_full=feat_mask)
    print(f"  -> ROI {w_roi}x{h_roi}, {len(kp)} keypoint")
    if len(kp) < 50:
        print(f"  -> CẢNH BÁO: quá ít keypoint ({len(kp)}), template này có thể kém tin cậy.")

    return {
        "ten_file": os.path.basename(duong_dan_anh),
        "kp_pts": np.array([k.pt for k in kp]),
        "kp_size": np.array([k.size for k in kp]),
        "kp_angle": np.array([k.angle for k in kp]),
        "kp_response": np.array([k.response for k in kp]),
        "kp_octave": np.array([k.octave for k in kp]),
        "des_ref": des,
        "contour": contour.reshape(-1, 2),
        "m_align": M_align,
        "roi": (x_roi, y_roi, w_roi, h_roi),
        "ref_size": (img.shape[1], img.shape[0]),
    }


def tao_bo_template(thu_muc_anh_mau: str, duong_dan_luu: str):
    danh_sach_anh = sorted(
        f for f in os.listdir(thu_muc_anh_mau)
        if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
    )
    if not danh_sach_anh:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong thư mục: {thu_muc_anh_mau}")

    print(f"Tìm thấy {len(danh_sach_anh)} ảnh mẫu. Đang xử lý từng ảnh...\n")

    templates = []
    for ten_file in danh_sach_anh:
        print(f"[{ten_file}]")
        duong_dan = os.path.join(thu_muc_anh_mau, ten_file)
        tpl = tao_1_template(duong_dan)
        if tpl is not None:
            templates.append(tpl)

            preview = cv2.imread(duong_dan)
            cv2.drawContours(preview, [tpl["contour"].reshape(-1, 1, 2).astype(np.int32)],
                              -1, (0, 255, 0), 8)
            os.makedirs("template_preview", exist_ok=True)
            cv2.imwrite(f"template_preview/{ten_file}_preview.png", preview)

    if not templates:
        raise RuntimeError("Không tạo được template nào - kiểm tra lại ảnh mẫu.")

    with open(duong_dan_luu, "wb") as f:
        pickle.dump(templates, f)

    print(f"\nĐã lưu {len(templates)}/{len(danh_sach_anh)} template vào: {duong_dan_luu}")
    print("Đã lưu ảnh xem trước từng template trong thư mục: template_preview/ "
          "(kiểm tra contour có đúng vật thể không)")


if __name__ == "__main__":
    tao_bo_template(DUONG_DAN_THU_MUC_ANH_MAU, DUONG_DAN_LUU_TEMPLATE)