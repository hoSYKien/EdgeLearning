"""
XOA NEN QUANH CHAN PIN - tim contour bao quanh chan pin (vung sang) trong
anh da crop, giu NGUYEN kich thuoc/ty le anh (khong resize/crop them nua),
chi TO DEN moi pixel NAM NGOAI contour do.

VI SAO LAM VIEC NAY (khac voi chi crop sat hon):
  - Crop sat hon nua se lam ty le chieu dai/rong cang lech, resize ve
    224x224 vuong se bop meo cang manh.
  - Thay vi cat bot khung anh, GIU NGUYEN khung (ty le van nhu cu, do
    meo khi resize khong doi), nhung XOA THONG TIN nen ben trong khung
    do bang cach to den - model van thay dung 1 kich thuoc anh nhu truoc,
    nhung phan nen (khong phai chan pin) gio la mau den dong nhat, khong
    con texture/mau sac gi de model "hoc nham" thanh dac trung nua.

CACH DO CONTOUR:
  1. Threshold Otsu tren anh xam - chan pin la vung SANG, tach khoi nen
     TOI phia sau (giong cach lam trong crop_pins.py).
  2. Lam sach nhieu bang morphology (close roi open).
  3. Tim TAT CA contour, chon contour co DIEN TICH LON NHAT (gia dinh la
     chan pin, vi chan pin thuong la vung sang lien tuc lon nhat trong
     khung anh da crop sat).
  4. Tao mask tu contour do (fill trang ben trong, den ben ngoai).
  5. Nhan anh goc voi mask -> pixel ngoai contour thanh den (0,0,0),
     pixel trong contour giu nguyen.

Cach dung:
  - Chinh CONFIG ben duoi.
  - Chay thu tren 1 anh mau, xem preview (contour ve mau xanh + anh da
    xoa nen canh nhau) TRUOC KHI chay hang loat ca dataset.
"""

import os

import cv2
import numpy as np
from PIL import Image

# ==========================================================================
# CONFIG - CHINH O DAY
# ==========================================================================

INPUT_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\val\Pin_OK_masked"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\val\Pin_OK_masked"

MORPH_KERNEL_SIZE = 3     # lam sach nhieu truoc khi tim contour - tang neu
                           # anh nhieu hat, giam neu chan pin manh bi mat chi tiet
MIN_CONTOUR_AREA_RATIO = 0.03   # contour nho hon ty le nay (so voi dien tich
                                  # ca anh) se bi bo qua - tranh nham vao dom nhieu nho
DILATE_PX = 2              # "no rong" contour ra vai pixel truoc khi mask, tranh
                           # cat sat qua mat vien chan pin that (0 = khong no rong)

# Lam MIN duong vien mask - blur Gaussian roi threshold lai, bo tron cac
# goc canh rang cua kieu pixel. So CANG LON thi duong vien CANG MIN (nhung
# cung lam mat chi tiet nho o vien neu qua lon). Phai la SO LE. 0 hoac 1 =
# tat, khong lam min.
MASK_SMOOTH_KERNEL = 45

# Cat CUNG phan tren cung anh - khac voi mask (chi to den, van giu kich
# thuoc), cai nay XOA HAN vai hang pixel tren cung, lam anh THAP xuong that.
# Ty le tinh theo % chieu cao anh GOC, vd 0.20 = cat bo 20% tren cung.
TOP_CROP_RATIO = 0.1

SAVE_PREVIEW = True        # luu them anh preview (contour + so sanh truoc/sau)


# ==========================================================================
# LOGIC CHINH
# ==========================================================================

def smooth_mask(mask, kernel_size):
    """Lam min duong vien mask bang cach lam mo Gaussian roi threshold lai
    ve nhi phan - blur se "hoa tron" cac pixel goc canh, threshold lai bien
    no thanh 1 duong cong tron thay vi bac thang pixel."""
    if kernel_size is None or kernel_size <= 1:
        return mask
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1   # phai la so le
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed


def find_pin_mask(gray, morph_kernel_size, min_area_ratio, dilate_px, smooth_kernel):
    """Tra ve mask nhi phan (0/255) cung kich thuoc voi gray - 255 = thuoc
    chan pin (giu nguyen), 0 = nen (se bi to den). Tra ve None neu khong
    tim duoc contour du lon (an toan - luc do KHONG mask gi ca, giu nguyen
    anh goc, tranh lam hong du lieu vi 1 loi do sai)."""
    h, w = gray.shape
    img_area = h * w

    _, thresh = cv2.threshold(gray, 119, 255, cv2.THRESH_BINARY)

    k = max(1, morph_kernel_size)
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_ratio * img_area:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

    if dilate_px > 0:
        dilate_kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel)

    mask = smooth_mask(mask, smooth_kernel)

    return mask


def crop_top(img, ratio):
    """Cat bo N% chieu cao TREN CUNG cua anh - XOA HAN (khong phai to den),
    lam anh thap xuong that. ratio=0 -> khong cat gi ca."""
    if ratio <= 0:
        return img
    h = img.shape[0]
    top_px = int(round(h * ratio))
    return img[top_px:, :]


def apply_mask_black_background(img_bgr, mask):
    """Tra ve anh moi CUNG KICH THUOC voi img_bgr - pixel ngoai mask thanh
    den, pixel trong mask giu nguyen mau goc."""
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_bgr, mask_3ch)


def process_one_image(image_path, out_dir, base_name=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  [BO QUA] Loi doc anh: {image_path}")
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = find_pin_mask(gray, MORPH_KERNEL_SIZE, MIN_CONTOUR_AREA_RATIO, DILATE_PX, MASK_SMOOTH_KERNEL)

    if mask is None:
        print(f"  [CANH BAO] Khong tim thay contour du lon trong "
              f"{os.path.basename(image_path)} - giu nguyen anh goc, khong mask.")
        result_bgr = img_bgr
    else:
        result_bgr = apply_mask_black_background(img_bgr, mask)

    result_bgr = crop_top(result_bgr, TOP_CROP_RATIO)

    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_masked.png")
    cv2.imwrite(out_path, result_bgr)

    if SAVE_PREVIEW and mask is not None:
        contour_preview = img_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 2)
        contour_preview = crop_top(contour_preview, TOP_CROP_RATIO)   # dong bo voi ket qua da cat
        combined = np.hstack([contour_preview, result_bgr])
        preview_path = os.path.join(out_dir, f"{base_name}_preview.png")
        cv2.imwrite(preview_path, combined)

    return out_path


def process_folder(input_dir, output_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    if not files:
        print(f"Khong tim thay anh nao trong: {input_dir}")
        return

    for fname in files:
        out_path = process_one_image(os.path.join(input_dir, fname), output_dir)
        if out_path:
            print(f"  {fname} -> {out_path}")

    print(f"\nXong. Da xu ly {len(files)} anh, luu tai: {output_dir}")


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        process_folder(INPUT_PATH, OUTPUT_DIR)
    else:
        out_path = process_one_image(INPUT_PATH, OUTPUT_DIR)
        print(f"Da luu: {out_path}")