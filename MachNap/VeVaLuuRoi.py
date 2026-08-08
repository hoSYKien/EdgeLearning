"""
Vẽ ROI (Region Of Interest) BẰNG TAY trên 1 ảnh mẫu, sau đó LƯU TỌA ĐỘ các
vùng đó ra file JSON để dùng lại về sau (không phải vẽ lại mỗi lần).

Cách dùng:
    Sửa đường dẫn ảnh mẫu bên dưới rồi chạy:
        python 02_ve_va_luu_roi.py

Cách thao tác khi cửa sổ chọn ROI hiện lên:
    1. Kéo chuột vẽ 1 hình chữ nhật quanh vùng muốn cắt.
    2. Nhấn ENTER (hoặc SPACE) để xác nhận vùng đó và vẽ tiếp vùng khác.
    3. Lặp lại bước 1-2 cho mỗi vùng muốn cắt.
    4. Nhấn ESC khi đã vẽ xong TẤT CẢ các vùng cần cắt.

Sau khi chạy xong, tọa độ (đã quy đổi về đúng độ phân giải ảnh gốc) sẽ
được lưu vào file JSON, có thể nạp lại bằng hàm doc_roi_da_luu() trong
script này (hoặc copy hàm đó sang script khác).
"""

import os
import json
import cv2

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_ANH_MAU = r"D:\TongHop\RTC Technologi\PCB\captured_crop\capture_20260727_105717_001_roi.png"
DUONG_DAN_FILE_ROI = r"D:\TongHop\RTC Technologi\PCB\vung_roi.json"
# =====================================================================

# Ảnh có thể lớn -> thu nhỏ khi HIỂN THỊ để vừa màn hình, tọa độ ROI vẽ
# được sẽ tự quy đổi lại đúng độ phân giải gốc khi lưu.
DO_RONG_HIEN_THI = 1300


def ve_va_luu_roi(duong_dan_anh_mau: str, duong_dan_file_roi: str):
    img = cv2.imread(duong_dan_anh_mau)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {duong_dan_anh_mau}")
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

    rois = [tuple(int(v) for v in r) for r in rois if r[2] > 0 and r[3] > 0]
    if not rois:
        raise RuntimeError("Không có ROI nào được chọn.")

    # quy đổi tọa độ ROI về đúng độ phân giải ảnh gốc
    rois_goc = [
        [int(x / scale), int(y / scale), int(ww / scale), int(hh / scale)]
        for (x, y, ww, hh) in rois
    ]

    print(f"Đã chọn {len(rois_goc)} ROI (tọa độ trên ảnh gốc {w}x{h}):")
    for i, r in enumerate(rois_goc, start=1):
        print(f"  ROI {i}: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")

    du_lieu_luu = {
        "anh_mau": duong_dan_anh_mau,
        "kich_thuoc_anh_mau": [w, h],
        "rois": rois_goc,
    }
    with open(duong_dan_file_roi, "w", encoding="utf-8") as f:
        json.dump(du_lieu_luu, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu {len(rois_goc)} ROI vào: {duong_dan_file_roi}")


def doc_roi_da_luu(duong_dan_file_roi: str):
    """Đọc lại danh sách ROI [(x, y, w, h), ...] đã lưu trước đó."""
    with open(duong_dan_file_roi, "r", encoding="utf-8") as f:
        du_lieu = json.load(f)
    return [tuple(r) for r in du_lieu["rois"]], tuple(du_lieu["kich_thuoc_anh_mau"])


if __name__ == "__main__":
    ve_va_luu_roi(DUONG_DAN_ANH_MAU, DUONG_DAN_FILE_ROI)