"""
CAT BO ~10% PHIA TREN ANH - don gian, khong lam gi khac.

Cach dung:
  - Sua INPUT_PATH / OUTPUT_DIR ben duoi.
  - INPUT_PATH la 1 FILE -> chi cat anh do.
  - INPUT_PATH la 1 THU MUC -> cat het tat ca anh trong thu muc do.
"""

import os
from PIL import Image

# ==========================================================================
# CONFIG
# ==========================================================================

INPUT_PATH = r"D:\TongHop\RTC Technologi\HZT\HZT.Bottom.Split_crop_4_bot\val\OK"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\HZT\HZT.Bottom.Split_crop_4_bot\val\crop_ok"

TOP_CROP_RATIO = 0.20   # cat bo 10% chieu cao tren cung


# ==========================================================================
def crop_top(image, ratio):
    w, h = image.size
    top_px = int(round(h * ratio))
    return image.crop((0, top_px, w, h))


def process_one(image_path, out_dir):
    image = Image.open(image_path)
    cropped = crop_top(image, TOP_CROP_RATIO)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(image_path)
    out_path = os.path.join(out_dir, fname)
    cropped.save(out_path)
    return out_path


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in sorted(os.listdir(INPUT_PATH)) if f.lower().endswith(exts)]
        for fname in files:
            out_path = process_one(os.path.join(INPUT_PATH, fname), OUTPUT_DIR)
            print(f"{fname} -> {out_path}")
        print(f"\nXong. Da cat {len(files)} anh, luu tai: {OUTPUT_DIR}")
    else:
        out_path = process_one(INPUT_PATH, OUTPUT_DIR)
        print(f"Da luu: {out_path}")