"""
crop_batch.py
-------------
Crop nhieu anh cung luc thanh 5 phan theo ty le chieu rong (width ratio), dung OpenCV.
Moi loai crop (part1, part2, ...) se duoc luu vao 1 thu muc con rieng trong OUTPUT_DIR.

Cach dung:
    1. Sua INPUT_DIR / OUTPUT_DIR ben duoi cho dung duong dan may ban.
    2. Chay: python crop_batch.py

Cai thu vien can thiet (neu chua co):
    pip install opencv-python
"""

import os
import glob
import cv2

# ==== SUA DUONG DAN O DAY ====
INPUT_DIR = r"D:\TongHop\RTC Technologi\PCB\crop7\NG"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\PCB\crop7\NG_crop"

# ==== TY LE CROP (5 phan, tinh theo % chieu rong anh, tu 0.0 den 1.0) ====
# Co the dat ten rieng cho tung phan (dung lam ten thu muc con), neu khong dat
# se tu dong dung "part1", "part2", ...

RATIOS = [
    ("part1", 0.00, 0.33),
    ("part2", 0.33, 0.66),
    ("part3", 0.66, 1.00),
    ("part4", 0.33 / 2, (0.66 - 0.33) / 2 + 0.33),
    ("part5", (0.66 - 0.33) / 2 + 0.33, (1 - 0.66) / 2 + 0.66),
]

# RATIOS = [
#     ("part1", 0.00, 0.5),
#     ("part2", 0.1, 1.00),
#     ("part4", 0.25, 0.75)
# ]

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def crop_one_image(image_path, output_dir, ratios):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    saved_files = []
    for name, start, end in ratios:
        x0 = int(round(w * start))
        x1 = int(round(w * end))
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        if x1 <= x0:
            continue

        # moi loai crop (name) co 1 thu muc con rieng
        part_dir = os.path.join(output_dir, name)
        os.makedirs(part_dir, exist_ok=True)

        crop = img[0:h, x0:x1]
        out_path = os.path.join(part_dir, f"{base_name}_{name}.png")
        cv2.imwrite(out_path, crop)
        saved_files.append(out_path)

    return saved_files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = []
    for ext in VALID_EXT:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"Khong tim thay anh nao trong: {INPUT_DIR}")
        return

    print(f"Tim thay {len(image_paths)} anh. Bat dau crop...")
    for path in image_paths:
        try:
            saved = crop_one_image(path, OUTPUT_DIR, RATIOS)
            print(f"- {os.path.basename(path)}: da tao {len(saved)} file")
        except Exception as e:
            print(f"- {os.path.basename(path)}: LOI - {e}")

    print(f"\nHoan tat. Ket qua nam trong: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()