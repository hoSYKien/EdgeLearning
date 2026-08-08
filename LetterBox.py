"""
CHEM NEN DEN TREN/DUOI (letterbox) - anh dang RONG hon nhieu so voi CAO
(vd 3823x685) - thay vi de Resize() sau nay keo meo ca 2 chieu khong deu
nhau, chem them nen DEN vao TREN va DUOI de chieu cao tang len bang chieu
rong -> anh vuong lai MA KHONG lam meo hinh dang that ben trong.

Sau buoc nay, Resize(224,224) o pipeline train se chi con phong to/nho
DEU CA 2 CHIEU (vi canh da vuong san), khong con bop meo lech ty le nua.

Cach dung:
  - Sua INPUT_PATH / OUTPUT_DIR ben duoi.
  - INPUT_PATH la 1 FILE -> chi xu ly anh do.
  - INPUT_PATH la 1 THU MUC -> xu ly het tat ca anh trong thu muc do.
"""

import os
from PIL import Image

# ==========================================================================
# CONFIG
# ==========================================================================

INPUT_PATH = r"D:\TongHop\RTC Technologi\HZT\HZT.Bottom.Split_crop_4_bot\val\crop_ng"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\HZT\HZT.Bottom.Split_crop_4_bot\val\crop_ng_letterbox"

PAD_COLOR = (0, 0, 0)   # mau nen chem vao - (0,0,0) = den


# ==========================================================================
def letterbox_pad(image):
    """Chem nen TREN/DUOI (neu anh rong hon cao) hoac TRAI/PHAI (neu anh
    cao hon rong) de anh thanh HINH VUONG, giu nguyen ty le noi dung goc
    (khong keo meo gi ca), chi them vien mau PAD_COLOR."""
    w, h = image.size
    side = max(w, h)

    canvas = Image.new(image.mode, (side, side), PAD_COLOR)
    # dan anh goc vao chinh giua canvas vuong
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def process_one(image_path, out_dir):
    image = Image.open(image_path)
    padded = letterbox_pad(image)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(image_path)
    out_path = os.path.join(out_dir, fname)
    padded.save(out_path)
    return out_path


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in sorted(os.listdir(INPUT_PATH)) if f.lower().endswith(exts)]
        for fname in files:
            out_path = process_one(os.path.join(INPUT_PATH, fname), OUTPUT_DIR)
            print(f"{fname} -> {out_path}")
        print(f"\nXong. Da xu ly {len(files)} anh, luu tai: {OUTPUT_DIR}")
    else:
        out_path = process_one(INPUT_PATH, OUTPUT_DIR)
        print(f"Da luu: {out_path}")