"""
TACH NEN - giu lai vung SANG NHAT (mau vang dam cua chan pin), TO DEN
phan con lai (nen olive/den xung quanh).

Dung Otsu threshold tren anh XAM - do vang dam (~240) va nen (~68) chenh
lech do sang rat lon, khong can xu ly theo mau (HSV) gi ca.

Cach dung:
    python remove_background.py
"""

import os
import cv2
import numpy as np

# ==========================================================================
# CONFIG
# ==========================================================================

INPUT_PATH = r"D:\TongHop\RTC Technologi\9\test\NG"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\9\test\NG_masked"

MORPH_KERNEL_SIZE = 7        # lam sach nhieu truoc khi tim contour - tang de xu ly noise manh hon
MASK_SMOOTH_KERNEL = 9       # lam min duong vien cuoi cung (blur + threshold lai)

# Dung CONVEX HULL (bao loi) cho moi vung vang - AN TOAN TUYET DOI cho loi
# khuyet vat lieu o vien (hull luon PHONG RA NGOAI, khong bao gio lom vao,
# nen khong the "an mat" 1 khuyet that duoc). Danh doi: voi chan pin thon
# manh (dau loe, than hep), hull co the giu lai 1 it nen o cho thon do -
# chap nhan duoc, uu tien AN TOAN cho du lieu loi hon la sat hình tuyet doi.
USE_CONVEX_HULL = True

# Sau khi tinh hull, NO RONG THEM ra 1 khoang dem (margin) truoc khi mask -
# de KHONG bi "om sat" qua, giu lai them 1 chut ngu canh nen xung quanh.
# 0 = khong no them, chi dung dung hull.
HULL_MARGIN_PX = 10

MIN_CONTOUR_AREA_RATIO = 0.01   # contour nho hon ty le nay (nghi la nhieu) se bi bo qua -
                                  # giam xuong (so voi 0.03 truoc do) vi anh nhieu chan pin,
                                  # moi chan chiem ty le dien tich nho hon so voi anh 1-chan
SAVE_PREVIEW = False

# Nguong mau VANG (HSV) - chi lay vung VANG DAM THAT SU, khong bat nham
# vung nen sang mau (nhat, xam/kem) du co the CUNG do sang voi vung vang.
# DA TANG SAT_MIN/VAL_MIN len chat hon (thay vi chi dua vao 1 nguong long
# le) vi van xuoc tren nen va mep pin THAT SU trung lan nhau ve mau o muc
# pixel - khong co ranh gioi sach tuyet doi. Ngong chat hon co the "an" mat
# vai pixel bien that, nhung khong sao vi buoc fill contour se tu lap lai
# phan BEN TRONG, chi lam contour hoi nho lai (= cat sat hon, dung y muon).
HUE_MIN, HUE_MAX = 15, 45
SAT_MIN = 100                     # tang tu 80 -> 100
VAL_MIN = 150                     # tang tu 100 -> 150

BG_COLOR = (255, 255, 255)   # mau nen (B, G, R) - (255,255,255) = trang, (0,0,0) = den


# ==========================================================================
def smooth_mask(mask, kernel_size):
    if kernel_size <= 1:
        return mask
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed


def remove_background(img_bgr):
    """Tra ve (result_bgr, mask). result_bgr: anh da to den nen. mask=None
    neu khong tim thay vung vang du lon (an toan - giu nguyen anh goc)."""
    h, w = img_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([HUE_MIN, SAT_MIN, VAL_MIN])
    upper = np.array([HUE_MAX, 255, 255])
    thresh = cv2.inRange(hsv, lower, upper)

    # CHI dung OPEN (khong CLOSE) o buoc lam sach ban dau - OPEN chi loai
    # bo cac diem nhieu li ti CHU KHONG noi cac diem nhieu rai rac lai voi
    # nhau. Neu dung CLOSE o day, cac dom nhieu nen (vet xuoc, nen JPEG...)
    # nam gan nhau co the bi "noi" thanh 1 khoi lon GIA truoc khi kip loc
    # theo dien tich, khien khoi gia do vuot qua bo loc va bi giu nham
    # (day chinh la loi da gap - vung nen bi giu lai du mau khong phai vang).
    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, None

    # Giu TAT CA contour du lon (khong chi 1 contour lon nhat) - vi anh co
    # the co NHIEU chan pin tach roi nhau (nhieu blob doc lap), khong phai
    # luon chi co 1 vung vang duy nhat trong khung hinh.
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_RATIO * img_area]
    if not valid_contours:
        return img_bgr, None

    # Dung CONVEX HULL cho moi vung (neu bat) - phong ra ngoai, khong bao
    # gio lom vao, nen KHONG THE lam mat 1 khuyet vat lieu that o vien.
    if USE_CONVEX_HULL:
        shapes = [cv2.convexHull(c) for c in valid_contours]
    else:
        shapes = valid_contours

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, shapes, -1, 255, cv2.FILLED)

    # No rong them 1 khoang dem (margin) sau khi da co hull - de khong bi
    # "om sat" qua, giu lai chut ngu canh nen xung quanh.
    if HULL_MARGIN_PX > 0:
        margin_kernel = np.ones((HULL_MARGIN_PX * 2 + 1, HULL_MARGIN_PX * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, margin_kernel)

    mask = smooth_mask(mask, MASK_SMOOTH_KERNEL)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas = np.full_like(img_bgr, BG_COLOR, dtype=np.uint8)
    result = np.where(mask_3ch > 0, img_bgr, canvas)
    return result, mask


def process_one(image_path, out_dir):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  [BO QUA] Loi doc anh: {image_path}")
        return None

    result, mask = remove_background(img_bgr)
    if mask is None:
        print(f"  [CANH BAO] Khong tach duoc nen trong {os.path.basename(image_path)} "
              f"- giu nguyen anh goc.")

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(image_path)
    base, ext = os.path.splitext(fname)
    out_path = os.path.join(out_dir, f"{base}_masked{ext}")
    cv2.imwrite(out_path, result)

    if SAVE_PREVIEW and mask is not None:
        combined = np.hstack([img_bgr, result])
        cv2.imwrite(os.path.join(out_dir, f"{base}_preview{ext}"), combined)

    return out_path


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in sorted(os.listdir(INPUT_PATH)) if f.lower().endswith(exts)]
        for fname in files:
            out_path = process_one(os.path.join(INPUT_PATH, fname), OUTPUT_DIR)
            if out_path:
                print(f"{fname} -> {out_path}")
        print(f"\nXong. Da xu ly {len(files)} anh, luu tai: {OUTPUT_DIR}")
    else:
        out_path = process_one(INPUT_PATH, OUTPUT_DIR)
        print(f"Da luu: {out_path}")