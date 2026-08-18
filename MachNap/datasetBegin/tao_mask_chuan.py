r"""
tao_mask_chuan.py - Sinh MASK CHUẨN tự động từ N ảnh OK, thay cho việc đo
tay toạ độ BOARD_X1 / USB_X1 / KHUYET_Y2...

Ý tưởng: sau khi align, MỌI ảnh của cùng một loại hàng đều nằm ở đúng một
vị trí. Chồng mask vật thể của N ảnh lên nhau rồi cho từng pixel BỎ PHIẾU:
pixel nào được >= TI_LE_PHIEU số ảnh coi là "vật" thì thuộc mask chuẩn.

    - Nhiễu biên của từng ảnh (dao động vài px) bị bỏ phiếu loại.
    - Kết quả là 1 mask CỐ ĐỊNH, giống hệt nhau ở mọi ảnh -> giống hệt tính
      chất của mask hình học cũ, nhưng không phải đo gì.
    - Sang vật mới: chụp 30 ảnh OK, chạy lệnh này, xong.

CÁCH DÙNG:
    Bước 1: đặt cauhinh.LUU_MASK_ALIGN = True và NGUON_MASK = "tat"
            (hoặc "tuc_thi"), rồi chụp/gán nhãn ~30 ảnh OK như bình thường.
            Mỗi ảnh sẽ để lại 1 mask trong THU_MUC_MASK_ALIGN.

    Bước 2: python tao_mask_chuan.py

    Bước 3: đặt cauhinh.NGUON_MASK = "file" và chạy lại pipeline.
            (nếu đã có dataset cũ thì chạy chay_xuat_lai.py)

Tuỳ chọn:
    python tao_mask_chuan.py --phieu 0.9   -> khắt khe hơn, mask co lại
    python tao_mask_chuan.py --xem         -> chỉ xem thử, không ghi file
"""

import os
import sys

import cv2
import numpy as np

import cauhinh as cf

# Pixel phải được ít nhất ngần này tỉ lệ số ảnh bầu là "vật" mới vào mask.
#   0.5  -> mask rộng rãi (quá bán là đủ), giữ được cả phần hay bị thiếu sáng
#   0.8  -> chặt hơn, chỉ giữ phần gần như ảnh nào cũng có (mặc định)
#   0.95 -> rất chặt, mask co sát lõi vật
TI_LE_PHIEU = 0.8

# Nở mask ra vài pixel sau khi bỏ phiếu, cho chắc không gặm mất mép vật.
NO_THEM = 5

# Lấp lỗ kín + làm mượt biên.
LAM_MUOT = 7
CHI_GIU_BLOB_LON_NHAT = True


def doc_cac_mask(thu_muc):
    ten = sorted(f for f in os.listdir(thu_muc)
                 if f.lower().endswith(cf.VALID_EXT))
    if not ten:
        raise SystemExit(
            f"Không có mask nào trong: {thu_muc}\n"
            "-> Đặt LUU_MASK_ALIGN = True rồi chụp/gán nhãn vài chục ảnh OK.")

    masks, kich_thuoc, bo_qua = [], None, 0
    for t in ten:
        m = cv2.imread(os.path.join(thu_muc, t), cv2.IMREAD_GRAYSCALE)
        if m is None:
            bo_qua += 1
            continue
        if kich_thuoc is None:
            kich_thuoc = m.shape
        elif m.shape != kich_thuoc:
            # ảnh khác kích thước = align ra khung khác -> không chồng được
            bo_qua += 1
            continue
        masks.append((m > 127).astype(np.uint8))
    if not masks:
        raise SystemExit("Không đọc được mask nào hợp lệ.")
    if bo_qua:
        print(f"Bỏ qua {bo_qua} file (hỏng hoặc khác kích thước).")
    return masks, kich_thuoc


def bo_phieu(masks, ti_le_phieu):
    """Cộng dồn rồi threshold theo số phiếu."""
    tong = np.zeros(masks[0].shape, np.int32)
    for m in masks:
        tong += m
    nguong = max(1, int(round(len(masks) * ti_le_phieu)))
    return (tong >= nguong).astype(np.uint8) * 255, tong


def lam_sach(mask):
    if NO_THEM > 0:
        k = np.ones((2 * NO_THEM + 1,) * 2, np.uint8)
        mask = cv2.dilate(mask, k)

    # lấp lỗ kín (đệm 1px để vật chạm mép không bị flood-fill ăn mất)
    h, w = mask.shape
    dem = np.zeros((h + 2, w + 2), np.uint8)
    dem[1:-1, 1:-1] = mask
    ff = dem.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    mask = cv2.bitwise_or(dem, cv2.bitwise_not(ff))[1:-1, 1:-1]

    if CHI_GIU_BLOB_LON_NHAT:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            lon = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(lbl == lon, 255, 0).astype(np.uint8)

    if LAM_MUOT >= 3:
        mask = cv2.medianBlur(mask, LAM_MUOT | 1)
    return mask


def anh_xem_thu(mask, tong, so_anh):
    """Bản đồ nhiệt số phiếu + viền mask chuẩn, để soi bằng mắt."""
    heat = cv2.applyColorMap(
        np.uint8(255 * tong / max(1, so_anh)), cv2.COLORMAP_JET)
    vien, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heat, vien, -1, (255, 255, 255), 3)
    ti_le = min(1.0, 1200 / max(heat.shape[:2]))
    return cv2.resize(heat, None, fx=ti_le, fy=ti_le) if ti_le < 1.0 else heat


def main():
    ti_le_phieu = TI_LE_PHIEU
    if "--phieu" in sys.argv:
        ti_le_phieu = float(sys.argv[sys.argv.index("--phieu") + 1])
    chi_xem = "--xem" in sys.argv

    masks, kich_thuoc = doc_cac_mask(cf.THU_MUC_MASK_ALIGN)
    print(f"Đọc {len(masks)} mask, kích thước {kich_thuoc[1]}x{kich_thuoc[0]}")
    if len(masks) < 10:
        print("*** Ít hơn 10 mask - kết quả sẽ nhiễu. Nên có >= 30 ảnh OK. ***")

    mask, tong = bo_phieu(masks, ti_le_phieu)
    tho = cv2.countNonZero(mask)
    mask = lam_sach(mask)
    sach = cv2.countNonZero(mask)

    dt_tb = float(np.mean([m.sum() for m in masks]))
    print(f"Ngưỡng phiếu: {ti_le_phieu:.0%} "
          f"({max(1, int(round(len(masks) * ti_le_phieu)))}/{len(masks)} ảnh)")
    print(f"Diện tích mask trung bình từng ảnh : {dt_tb:.0f} px")
    print(f"Sau bỏ phiếu                       : {tho} px")
    print(f"Sau nở {NO_THEM}px + lấp lỗ + làm mượt   : {sach} px "
          f"({sach / (kich_thuoc[0] * kich_thuoc[1]):.1%} khung)")

    x, y, w, h = cv2.boundingRect(mask)
    print(f"Bbox mask chuẩn: {w}x{h} tại ({x}, {y})")
    if sach < dt_tb * 0.7:
        print("*** Mask chuẩn NHỎ hơn hẳn mask từng ảnh -> align đang lệch "
              "giữa các ảnh, hoặc ngưỡng phiếu quá chặt. Thử --phieu 0.5 ***")

    cv2.imshow("Ban do phieu + vien mask chuan - phim bat ky de dong",
               anh_xem_thu(mask, tong, len(masks)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if chi_xem:
        print("\n(--xem: không ghi file)")
        return

    os.makedirs(os.path.dirname(cf.FILE_MASK_CHUAN) or ".", exist_ok=True)
    cv2.imwrite(cf.FILE_MASK_CHUAN, mask)
    print(f"\nĐã lưu mask chuẩn: {cf.FILE_MASK_CHUAN}")
    print('Giờ đặt  NGUON_MASK = "file"  trong cauhinh.py.')
    print("Nếu đã có dataset cũ thì chạy:  python chay_xuat_lai.py")


if __name__ == "__main__":
    main()
