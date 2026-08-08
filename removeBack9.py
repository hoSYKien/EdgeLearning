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
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\9\test\NG\removeBack"

MORPH_KERNEL_SIZE = 7        # lam sach nhieu truoc khi tim contour - tang de xu ly noise manh hon
DETACH_KERNEL_SIZE = 9       # kernel RIENG de "cat dut" cac tua rang cua noi lien vao vien chinh
                              # (open voi kernel lon hon MORPH_KERNEL_SIZE, chuyen tay cho viec
                              # tach cac vet noise dinh vao bien, khac voi lam sach nhieu li ti)
MASK_SMOOTH_KERNEL = 9       # lam min duong vien cuoi cung (blur + threshold lai)

# Kernel danh RIENG de "va" cac khe lom nho o RIA do vet loi toi tao ra
# (khien contour bi khuyet vao dung cho do), MA KHONG lam mat di do THON
# TU NHIEN cua hinh dang chan pin (vd dau loe, than hep). Chon so nay
# LON HON kich thuoc vet loi thuong gap 1 chut, nhung PHAI NHO HON nhieu
# so voi be rong cho hep nhat cua chan pin - neu khong se bi "an" mat ca
# cho hep that, giong het van de convex hull.
EDGE_DEFECT_CLOSE_KERNEL = 15
MIN_CONTOUR_AREA_RATIO = 0.01   # contour nho hon ty le nay (nghi la nhieu) se bi bo qua -
                                  # giam xuong (so voi 0.03 truoc do) vi anh nhieu chan pin,
                                  # moi chan chiem ty le dien tich nho hon so voi anh 1-chan
SAVE_PREVIEW = True

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

    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k)

    # "Cat dut" cac tua rang cua (van xuoc dinh vao bien pin) - open voi
    # kernel LON HON, chuyen de xu ly cac tua dai/mong, khac voi lam sach
    # nhieu li ti o buoc tren.
    detach_k = np.ones((DETACH_KERNEL_SIZE, DETACH_KERNEL_SIZE), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, detach_k)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, None

    # Giu TAT CA contour du lon (khong chi 1 contour lon nhat) - vi anh co
    # the co NHIEU chan pin tach roi nhau (nhieu blob doc lap), khong phai
    # luon chi co 1 vung vang duy nhat trong khung hinh.
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_RATIO * img_area]
    if not valid_contours:
        return img_bgr, None

    # Thay vi dung CONVEX HULL (bao TOAN BO hinh lom thanh loi, se "an mat"
    # ca phan THON TU NHIEN cua chan pin dang loe dau/hep than), dung
    # MORPHOLOGICAL CLOSING voi kernel VUA DU LON - chi va cac khe lom NHO
    # (co kich thuoc ~bang loi can va) ma KHONG dung cham den do thon that
    # cua hinh dang pin (vi kernel nho hon nhieu so voi do rong phan eo pin).
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, valid_contours, -1, 255, cv2.FILLED)

    if EDGE_DEFECT_CLOSE_KERNEL > 1:
        ck = EDGE_DEFECT_CLOSE_KERNEL
        close_kernel = np.ones((ck, ck), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

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