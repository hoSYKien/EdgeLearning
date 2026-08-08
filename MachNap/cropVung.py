"""
Vẽ ROI (Region Of Interest) BẰNG TAY trên ảnh ĐẦU TIÊN của 1 thư mục, sau
đó tự động áp dụng CÙNG vị trí ROI đó để cắt cho TẤT CẢ ảnh còn lại trong
thư mục (vì camera/nền cố định nên mọi ảnh cùng kích thước, cùng khung).

Vẽ được NHIỀU ROI cùng lúc -> mỗi ROI có 1 thư mục output riêng, chứa ảnh
đã cắt theo đúng vùng đó cho TOÀN BỘ ảnh trong thư mục nguồn.

Cách dùng:
    Sửa đường dẫn ở phần "CHỈNH ĐƯỜNG DẪN Ở ĐÂY" bên dưới rồi chạy:
        python crop_vung_tay.py

Cách thao tác khi cửa sổ chọn ROI hiện lên:
    1. Kéo chuột vẽ 1 hình chữ nhật quanh vùng muốn cắt.
    2. Nhấn ENTER (hoặc SPACE) để xác nhận vùng đó và vẽ tiếp vùng khác.
    3. Lặp lại bước 1-2 cho mỗi vùng muốn cắt.
    4. Nhấn ESC khi đã vẽ xong TẤT CẢ các vùng cần cắt.
    (Nếu vẽ nhầm 1 vùng, cứ vẽ tiếp - script chỉ tính các vùng có kích
    thước > 0; muốn hủy hẳn và vẽ lại từ đầu thì đóng cửa sổ và chạy lại.)
"""

import os
import cv2
import numpy as np

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_THU_MUC_ANH = r"D:\TongHop\RTC Technologi\PCB\crop6"   # thư mục chứa ảnh cần cắt vùng
DUONG_DAN_THU_MUC_OUTPUT = r"D:\TongHop\RTC Technologi\PCB\crop6Vung" # thư mục gốc chứa các thư mục con roi_1, roi_2...
# =====================================================================

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")

# Ảnh có thể rất lớn (vd 5472x3648) -> thu nhỏ khi HIỂN THỊ để vừa màn hình,
# tọa độ ROI vẽ được sẽ tự quy đổi lại đúng độ phân giải gốc khi cắt.
DO_RONG_HIEN_THI = 1300


def chon_roi_tren_anh_dau_tien(duong_dan_anh_dau_tien: str):
    img = cv2.imread(duong_dan_anh_dau_tien)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {duong_dan_anh_dau_tien}")
    h, w = img.shape[:2]

    scale = min(1.0, DO_RONG_HIEN_THI / w)
    img_hien_thi = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    print("\n>>> Cửa sổ chọn ROI sắp hiện lên. Cách thao tác:")
    print("    1. Kéo chuột vẽ 1 vùng chữ nhật.")
    print("    2. Nhấn ENTER hoặc SPACE để xác nhận vùng đó, rồi vẽ tiếp vùng khác.")
    print("    3. Nhấn ESC khi đã vẽ XONG hết các vùng cần cắt.\n")

    rois = cv2.selectROIs("Ve ROI - ENTER de xac nhan tung vung, ESC de ket thuc",
                           img_hien_thi, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    rois = [tuple(r) for r in rois if r[2] > 0 and r[3] > 0]
    if not rois:
        raise RuntimeError("Không có ROI nào được chọn.")

    # quy đổi tọa độ ROI về đúng độ phân giải ảnh gốc
    rois_goc = [
        (int(x / scale), int(y / scale), int(ww / scale), int(hh / scale))
        for (x, y, ww, hh) in rois
    ]

    print(f"Đã chọn {len(rois_goc)} ROI (tọa độ trên ảnh gốc {w}x{h}):")
    for i, r in enumerate(rois_goc, start=1):
        print(f"  ROI {i}: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")

    return rois_goc


def cat_ca_thu_muc(thu_muc_anh: str, thu_muc_output: str, rois: list):
    danh_sach_anh = sorted(
        f for f in os.listdir(thu_muc_anh) if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
    )
    if not danh_sach_anh:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong thư mục: {thu_muc_anh}")

    thu_muc_roi_list = []
    for i in range(len(rois)):
        tmp = os.path.join(thu_muc_output, f"roi_{i+1}")
        os.makedirs(tmp, exist_ok=True)
        thu_muc_roi_list.append(tmp)

    print(f"\nĐang cắt {len(rois)} vùng cho {len(danh_sach_anh)} ảnh...\n")

    thanh_cong, that_bai = [], []
    for ten_file in danh_sach_anh:
        duong_dan = os.path.join(thu_muc_anh, ten_file)
        img = cv2.imread(duong_dan)
        if img is None:
            print(f"[{ten_file}] Không đọc được ảnh, bỏ qua.")
            that_bai.append(ten_file)
            continue

        h, w = img.shape[:2]
        ok = True
        for i, (x, y, ww, hh) in enumerate(rois):
            x2, y2 = min(x + ww, w), min(y + hh, h)
            x1, y1 = max(0, x), max(0, y)
            if x1 >= x2 or y1 >= y2:
                print(f"[{ten_file}] ROI {i+1} nằm ngoài ảnh (ảnh {w}x{h}), bỏ qua ảnh này.")
                ok = False
                break
            crop = img[y1:y2, x1:x2]
            duong_dan_luu = os.path.join(thu_muc_roi_list[i], ten_file)
            cv2.imwrite(duong_dan_luu, crop)

        if ok:
            thanh_cong.append(ten_file)
        else:
            that_bai.append(ten_file)

    print("\n" + "=" * 50)
    print(f"Hoàn tất: {len(thanh_cong)}/{len(danh_sach_anh)} ảnh xử lý thành công.")
    if that_bai:
        print(f"Các ảnh bị lỗi ({len(that_bai)}): {', '.join(that_bai)}")
    for i, tmp in enumerate(thu_muc_roi_list, start=1):
        print(f"ROI {i} -> {tmp}")


def chay(thu_muc_anh: str, thu_muc_output: str):
    danh_sach_anh = sorted(
        f for f in os.listdir(thu_muc_anh) if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
    )
    if not danh_sach_anh:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong thư mục: {thu_muc_anh}")

    duong_dan_anh_dau_tien = os.path.join(thu_muc_anh, danh_sach_anh[0])
    print(f"Dùng ảnh đầu tiên để vẽ ROI: {danh_sach_anh[0]}")

    rois = chon_roi_tren_anh_dau_tien(duong_dan_anh_dau_tien)
    cat_ca_thu_muc(thu_muc_anh, thu_muc_output, rois)


if __name__ == "__main__":
    chay(DUONG_DAN_THU_MUC_ANH, DUONG_DAN_THU_MUC_OUTPUT)