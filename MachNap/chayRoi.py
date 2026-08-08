"""
BƯỚC 2/2 - Áp dụng BỘ template (nhiều ảnh mẫu) lên MỘT LOẠT ảnh mới: với
mỗi ảnh mới, thử khớp lần lượt với TẤT CẢ template đã tạo ở script 1,
CHỌN template cho kết quả khớp TỐT NHẤT (nhiều inlier nhất sau RANSAC) để
xoay + cắt ROI kích thước cố định.

Cách dùng: sửa đường dẫn thư mục ở phần "CHỈNH ĐƯỜNG DẪN Ở ĐÂY" bên dưới
rồi chạy:
    python 02_ap_dung_roi.py

Cách hoạt động (cho mỗi ảnh mới):
1. Trích đặc trưng SIFT trên ảnh mới MỘT LẦN (dùng chung cho mọi template).
2. Với TỪNG template trong bộ: so khớp + RANSAC, ghi lại số inlier.
3. Chọn template có số inlier CAO NHẤT (và đạt ngưỡng tối thiểu).
4. Dùng đúng template đó để suy ra vị trí/góc xoay, rồi xoay + cắt ROI
   theo kích thước chuẩn CỦA CHÍNH template đó.
5. Vì các template có thể có kích thước ROI đo được hơi khác nhau (sai số
   đo đạc), ảnh cắt cuối cùng được resize nhẹ về kích thước chuẩn chung
   (lấy từ template đầu tiên) để mọi ảnh output LUÔN cùng kích thước.

Yêu cầu: ảnh mới phải cùng góc camera/khoảng cách/nền như ảnh mẫu (không
đổi tỉ lệ phóng to thu nhỏ), vật thể có thể xoay hoặc dịch chuyển vị trí
tự do trong khung hình.
"""

import os
import pickle
import cv2
import numpy as np

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_THU_MUC_ANH_MOI = r"D:\TongHop\RTC Technologi\PCB\1"          # thư mục chứa các ảnh cần xử lý
DUONG_DAN_TEMPLATE = r"D:\TongHop\RTC Technologi\PCB\crop4\template.pkl"            # bộ template đã tạo ở script 1
DUONG_DAN_THU_MUC_KET_QUA = r"D:\TongHop\RTC Technologi\PCB\crop4\ket_qua"          # thư mục lưu ảnh vẽ contour/ROI kiểm tra
DUONG_DAN_THU_MUC_CROP = r"D:\TongHop\RTC Technologi\PCB\crop4\roi_crop"            # thư mục lưu ảnh ROI đã cắt (kích thước cố định)

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")
SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 4000
MIN_INLIERS = 15
# =====================================================================


def trich_dac_trung_sift(img, target_w=SIFT_TARGET_WIDTH, nfeatures=SIFT_NFEATURES):
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray, None)
    for k in kp:
        k.pt = (k.pt[0] / scale, k.pt[1] / scale)
    return kp, des


def load_templates(duong_dan_template: str):
    with open(duong_dan_template, "rb") as f:
        templates = pickle.load(f)
    for tpl in templates:
        tpl["kp"] = [
            cv2.KeyPoint(x=p[0], y=p[1], size=sz, angle=ang, response=resp, octave=int(oc))
            for p, sz, ang, resp, oc in zip(
                tpl["kp_pts"], tpl["kp_size"], tpl["kp_angle"],
                tpl["kp_response"], tpl["kp_octave"],
            )
        ]
        tpl["contour_arr"] = tpl["contour"].reshape(-1, 1, 2).astype(np.int32)
    return templates


def khop_voi_1_template(kp_new, des_new, tpl):
    """So khớp ảnh mới (đã trích đặc trưng sẵn) với 1 template.
    Trả về (so_inlier, M, so_match_tot) hoặc (0, None, 0) nếu không khớp được."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(tpl["des_ref"], des_new, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0, None, len(good)

    ref_pts = np.float32([tpl["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    new_pts = np.float32([kp_new[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(ref_pts, new_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
    if M is None or inliers is None:
        return 0, None, len(good)
    return int(inliers.sum()), M, len(good)


def xu_ly_1_anh(duong_dan_anh_moi: str, templates, canonical_w: int, canonical_h: int,
                 duong_dan_ket_qua: str, duong_dan_crop: str, min_inliers: int = MIN_INLIERS):
    img_new = cv2.imread(duong_dan_anh_moi)
    if img_new is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {duong_dan_anh_moi}")

    kp_new, des_new = trich_dac_trung_sift(img_new)
    if des_new is None or len(kp_new) < min_inliers:
        raise RuntimeError(
            f"Ảnh không có đủ chi tiết/đặc trưng để nhận diện "
            f"(chỉ tìm được {0 if des_new is None else len(kp_new)} keypoint)."
        )

    # Thử khớp với TẤT CẢ template, chọn cái có nhiều inlier nhất
    ket_qua_moi_template = []
    for tpl in templates:
        so_inlier, M, so_match = khop_voi_1_template(kp_new, des_new, tpl)
        ket_qua_moi_template.append((so_inlier, M, so_match, tpl))

    ket_qua_moi_template.sort(key=lambda t: t[0], reverse=True)
    so_inlier_tot_nhat, M_tot_nhat, so_match_tot_nhat, tpl_tot_nhat = ket_qua_moi_template[0]

    # In bảng xếp hạng để dễ theo dõi template nào hay thắng
    xep_hang = ", ".join(f"{t[3]['ten_file']}={t[0]}" for t in ket_qua_moi_template[:5])
    print(f"  Xếp hạng inlier theo template: {xep_hang}")

    if so_inlier_tot_nhat < min_inliers:
        raise RuntimeError(
            f"Không template nào khớp đủ tốt (cao nhất chỉ {so_inlier_tot_nhat} inlier, "
            f"template '{tpl_tot_nhat['ten_file']}')."
        )

    print(f"  -> Dùng template '{tpl_tot_nhat['ten_file']}' "
          f"({so_inlier_tot_nhat}/{so_match_tot_nhat} inlier)")

    contour_ref = tpl_tot_nhat["contour_arr"]
    m_align = tpl_tot_nhat["m_align"]
    x_roi, y_roi, w_roi, h_roi = tpl_tot_nhat["roi"]
    ref_size = tpl_tot_nhat["ref_size"]

    contour_new = cv2.transform(contour_ref.astype(np.float32), M_tot_nhat).reshape(-1, 2)
    out = img_new.copy()
    cv2.drawContours(out, [contour_new.astype(np.int32)], -1, (0, 255, 0), 4)
    cv2.imwrite(duong_dan_ket_qua, out)

    # Ghép M (mẫu->mới) với m_align (mẫu->mẫu đã thẳng) để suy ra
    # mới -> mẫu đã thẳng, rồi cắt đúng khung ROI của template đang dùng
    M_inv = cv2.invertAffineTransform(M_tot_nhat)
    M_3x3 = np.vstack([M_inv, [0, 0, 1]])
    align_3x3 = np.vstack([m_align, [0, 0, 1]])
    total_3x3 = align_3x3 @ M_3x3
    M_total = total_3x3[:2, :]

    warped = cv2.warpAffine(img_new, M_total, ref_size)
    crop = warped[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

    # Resize nhẹ về kích thước chuẩn chung (mọi ảnh output luôn cùng size,
    # dù template thắng có đo ROI hơi khác nhau vài px do sai số)
    if crop.shape[1] != canonical_w or crop.shape[0] != canonical_h:
        crop = cv2.resize(crop, (canonical_w, canonical_h))

    cv2.imwrite(duong_dan_crop, crop)
    print(f"  -> Đã lưu: {duong_dan_ket_qua}  |  {duong_dan_crop}")

    return out, crop


def xu_ly_ca_thu_muc(thu_muc_anh_moi: str, duong_dan_template: str,
                      thu_muc_ket_qua: str, thu_muc_crop: str):
    templates = load_templates(duong_dan_template)
    print(f"Đã load {len(templates)} template: "
          f"{', '.join(t['ten_file'] for t in templates)}\n")

    # kích thước chuẩn chung: lấy trung bình các template cho ổn định
    canonical_w = int(round(np.mean([t["roi"][2] for t in templates])))
    canonical_h = int(round(np.mean([t["roi"][3] for t in templates])))
    print(f"Kích thước ROI chuẩn chung (trung bình các template): {canonical_w}x{canonical_h}\n")

    os.makedirs(thu_muc_ket_qua, exist_ok=True)
    os.makedirs(thu_muc_crop, exist_ok=True)

    danh_sach_anh = sorted(
        f for f in os.listdir(thu_muc_anh_moi)
        if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
    )
    if not danh_sach_anh:
        print(f"Không tìm thấy ảnh nào trong thư mục: {thu_muc_anh_moi}")
        return

    print(f"Tìm thấy {len(danh_sach_anh)} ảnh trong '{thu_muc_anh_moi}'. Bắt đầu xử lý...\n")

    thanh_cong, that_bai = [], []
    for ten_file in danh_sach_anh:
        duong_dan_anh = os.path.join(thu_muc_anh_moi, ten_file)
        ten_goc, _ = os.path.splitext(ten_file)
        duong_dan_ket_qua = os.path.join(thu_muc_ket_qua, f"{ten_goc}_ketqua.png")
        duong_dan_crop = os.path.join(thu_muc_crop, f"{ten_goc}_roi.png")

        print(f"[{ten_file}]")
        try:
            xu_ly_1_anh(duong_dan_anh, templates, canonical_w, canonical_h,
                        duong_dan_ket_qua, duong_dan_crop)
            thanh_cong.append(ten_file)
        except Exception as e:
            print(f"  -> LỖI: {e}")
            that_bai.append(ten_file)
        print()

    print("=" * 50)
    print(f"Hoàn tất: {len(thanh_cong)}/{len(danh_sach_anh)} ảnh xử lý thành công.")
    if that_bai:
        print(f"Các ảnh bị lỗi ({len(that_bai)}): {', '.join(that_bai)}")


if __name__ == "__main__":
    xu_ly_ca_thu_muc(DUONG_DAN_THU_MUC_ANH_MOI, DUONG_DAN_TEMPLATE,
                      DUONG_DAN_THU_MUC_KET_QUA, DUONG_DAN_THU_MUC_CROP)