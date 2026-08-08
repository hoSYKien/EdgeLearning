"""
CROP CHAN PIN TU ANH LINH KIEN - tien xu ly de chuyen bai toan
"phan loai ca linh kien" (loi cuc bo, GAP xoa mat tin hieu) thanh
"phan loai tung chan pin" (loi chiem % dien tich lon hon nhieu trong
tung anh crop nho -> backbone + GAP hoat dong tot hon han).

Y TUONG:
  1. Xac dinh "dai chan pin" (pin band) - vung y-range trong anh noi
     cac chan pin nam, tuong phan voi nen toi phia sau.
  2. Tinh profile do sang theo COT trong dai do -> chan pin binh thuong
     la 1 vet sang, nen la toi.
  3. Tim cac doan sang "chac chan la pin" (dung do rong dien hinh loc
     ra, khong lay nham vao 2 mieng kep kim loai 2 ben ria).
  4. Tu cac doan chac chan do, uoc luong PITCH (khoang cach deu giua
     2 chan lien tiep) bang linear fit index<->vi tri.
  5. Dung PITCH nay de DUNG LAI toan bo luoi N chan, bat dau tu chan
     trai nhat, deu dan tien ve phai - KHONG phu thuoc viec tung chan
     co "sang" hay khong. Day la buoc quan trong nhat: chan bi loi/lech
     thuong KHONG tao vet sang binh thuong, neu chi dua vao threshold
     sang se de bo sot dung cai chan can hoc nhat.
  6. Crop vung quanh moi vi tri chan (mo rong len tren de lay them
     ngu canh phan than linh kien, khong chi rieng phan chan) va luu
     ra file rieng.

Cach dung:
  - Chinh CONFIG ben duoi cho dung dataset cua ban.
  - Chay thu tren 1 anh mau, xuat ca preview (save_preview=True) de
    KIEM TRA BANG MAT cac khung crop co dung vao tung chan khong,
    truoc khi chay hang loat ca dataset.
  - Neu co san dataset dang phan loai theo thu muc class (OK/, NG/...),
    dung ham process_dataset() de crop hang loat, GIU NGUYEN cau truc
    thu muc class -> dung thang lam DATASET_DIRS cho pipeline few-shot
    classification o buoc sau (moi anh goc -> N anh chan pin, tat ca
    ke thua nhan cua anh goc).
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ==========================================================================
# CONFIG - CHINH O DAY CHO DUNG BO ANH CUA BAN
# ==========================================================================

# Duong dan anh dau vao (1 anh) va thu muc luu ket qua.
# SUA 2 DONG NAY LA CHAY DUOC NGAY.
INPUT_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\test\NG_crop_4_bot"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\TEST\Pin_NG"

# Dai y-range (ty le 0..1 theo chieu cao anh) chua vung chan pin.
# Uoc luong tu anh mau: than linh kien (sang, dong deu) chiem ~0-47%,
# vung chan pin (tuong phan voi nen toi) nam khoang 53%-94%.
PIN_BAND_Y_RATIO = (0.53, 0.94)

EXPECTED_PIN_COUNT = 9         # so chan pin thuc te tren linh kien (dem bang mat)
MIN_PIN_WIDTH_RATIO = 0.012   # do rong toi thieu 1 doan duoc coi la "chan" (ty le theo chieu rong anh)
MAX_PIN_WIDTH_RATIO = 0.045   # do rong toi da - loc bo 2 mieng kep kim loai 2 ben ria (thuong rong hon nhieu)
EDGE_MARGIN_RATIO = 0.06      # loai bo moi doan sang nam trong khoang nay tinh tu 2 rop anh -
                               # kep kim loai 2 ben luon nam sat rop anh, co the co do rong
                               # tinh co giong chan pin nen loc rieng theo vi tri nay cho chac chan

CROP_WIDTH_RATIO = 0.062      # be rong crop DU PHONG - chi dung khi khong do duoc bien that
                               # cua chan pin (vd chan loi mat tin hieu sang, xem TIGHT_CROP ben duoi)

TIGHT_CROP_MARGIN_PX = 70       # so pixel dem them ra ngoai bien that da do duoc, tranh cat
                               # sat qua lam mat vien chan
TIGHT_CROP_MIN_BRIGHT_FRAC = 0.5   # 1 cot duoc tinh la "thuoc chan pin" neu >= ty le nay
                                    # so pixel trong cot do la pixel sang (sau nguong Otsu)
CROP_PAD_ABOVE_RATIO = 0.14    # mo rong len tren bao nhieu % chieu cao anh goc, de lay them
                               # ngu canh phan than linh kien ngay tren chan. Giam so nay neu
                               # thay crop bi keo cao qua, bop meo nhieu khi resize ve vuong.
CROP_PAD_BELOW_RATIO = 0.06   # mo rong xuong duoi 1 chut de khong cat cut dau chan




# ==========================================================================
# LOGIC CHINH - it can sua khi doi dataset, tru khi hinh dang linh kien
# khac han (vd chan nam doc thay vi nam ngang o day anh)
# ==========================================================================

def find_pin_centers(gray, band_y_ratio, expected_count,
                      min_width_ratio, max_width_ratio,
                      edge_margin_ratio=EDGE_MARGIN_RATIO):
    """Tra ve list N toa do x (center) cua N chan pin, DA duoc dung lai
    deu theo pitch - bao gom ca chan bi loi/mat tin hieu sang, cong voi
    y0/y1 cua dai pin va pitch da uoc luong (dung lai khi crop)."""
    h, w = gray.shape
    y0 = int(h * band_y_ratio[0])
    y1 = int(h * band_y_ratio[1])
    band = gray[y0:y1, :].astype(np.float32)

    col_mean = band.mean(axis=0)
    k = max(5, w // 250)  # kernel lam min ty le theo do rong anh
    kernel = np.ones(k) / k
    smooth = np.convolve(col_mean, kernel, mode="same")

    thresh = smooth.mean() + 0.5 * smooth.std()
    above = smooth > thresh

    min_w = w * min_width_ratio
    max_w = w * max_width_ratio
    edge_left = w * edge_margin_ratio
    edge_right = w * (1 - edge_margin_ratio)

    # Tim cac doan lien tuc tren nguong, loc theo do rong hop le VA
    # loai bo doan nam trong vung rop anh (kep kim loai 2 ben)
    segments = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        if not v and start is not None:
            seg_w = i - start
            center = (start + i) / 2.0
            if min_w <= seg_w <= max_w and edge_left <= center <= edge_right:
                segments.append(center)
            start = None
    if start is not None:
        seg_w = len(above) - start
        center = (start + len(above)) / 2.0
        if min_w <= seg_w <= max_w and edge_left <= center <= edge_right:
            segments.append(center)

    if len(segments) < 2:
        raise RuntimeError(
            f"Chi tim thay {len(segments)} doan hop le - khong du de uoc luong pitch. "
            f"Kiem tra lai PIN_BAND_Y_RATIO / MIN_MAX_WIDTH_RATIO."
        )

    segments = sorted(segments)
    diffs = np.diff(segments)
    pitch = float(np.median(diffs))

    # Doi voi moi doan tim duoc, uoc luong no la chan pin thu bao nhieu
    # (index) bang cach chia khoang cach toi doan dau tien cho pitch va
    # lam tron - sau do fit tuyen tinh index<->x cho on dinh (chong nhieu
    # tot hon la chi dung 2 diem hoac median don le).
    idx_guess = np.round((np.array(segments) - segments[0]) / pitch).astype(int)
    a, b = np.polyfit(idx_guess, segments, 1)  # x = a*idx + b

    n = expected_count if expected_count else int(idx_guess.max()) + 1
    # Neo lai idx=0 dung vao chan trai nhat thuc su (idx_guess co the am
    # neu doan dau khong phai chan ngoai cung ben trai do nhieu):
    offset = idx_guess.min()
    grid_centers = [a * (offset + i) + b for i in range(n)]
    return grid_centers, y0, y1, pitch


def find_tight_pin_bbox(gray, center_x, y0, y1, pitch,
                         margin_px=TIGHT_CROP_MARGIN_PX,
                         min_bright_frac=TIGHT_CROP_MIN_BRIGHT_FRAC):
    """Do bien THAT (trai/phai) cua 1 chan pin bang nguong Otsu, trong 1
    cua so cuc bo quanh vi tri du kien (center_x +- nua pitch). Tra ve
    (x0, x1) toa do TOAN CUC neu do duoc, hoac None neu khong du tin cay
    (vd chan bi loi khong tao vet sang ro rang - luc do se fallback ve
    be rong co dinh o noi goi ham nay)."""
    h, w = gray.shape
    win_half = pitch * 0.5
    wx0 = int(max(0, round(center_x - win_half)))
    wx1 = int(min(w, round(center_x + win_half)))
    band = gray[y0:y1, wx0:wx1]
    if band.size == 0 or band.shape[1] < 4:
        return None

    band_u8 = band.astype(np.uint8)
    _, thresh = cv2.threshold(band_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_frac_per_col = (thresh > 0).mean(axis=0)
    bright_cols = np.where(bright_frac_per_col >= min_bright_frac)[0]

    if len(bright_cols) == 0:
        return None  # khong tim thay vet sang du ro - de noi goi fallback

    # Chi lay doan LIEN TUC dai nhat (tranh nham cac vet sang le te khac
    # trong cua so, vd bong/nhieu), vi chan pin la 1 khoi lien tuc.
    gaps = np.where(np.diff(bright_cols) > 1)[0]
    runs = np.split(bright_cols, gaps + 1)
    longest_run = max(runs, key=len)

    local_x0, local_x1 = longest_run.min(), longest_run.max()
    global_x0 = wx0 + local_x0 - margin_px
    global_x1 = wx0 + local_x1 + margin_px
    return max(0, global_x0), min(w, global_x1)


def crop_single_pin(image, gray, center_x, y0, y1, pitch,
                     crop_width_ratio, pad_above_ratio, pad_below_ratio,
                     ):
    w, h = image.size

    bbox = find_tight_pin_bbox(gray, center_x, y0, y1, pitch)
    if bbox is not None:
        x0, x1 = bbox
    else:
        # Fallback: khong do duoc bien that (thuong la chan loi mat tin
        # hieu sang) - dung be rong co dinh quanh vi tri du kien tu luoi
        # pitch, de van co anh crop dua vao train thay vi bo sot.
        half_w = w * crop_width_ratio * 0.5
        x0 = int(round(center_x - half_w))
        x1 = int(round(center_x + half_w))

    top = int(round(y0 - h * pad_above_ratio))
    bottom = int(round(y1 + h * pad_below_ratio))

    x0, x1 = max(0, x0), min(w, x1)
    top, bottom = max(0, top), min(h, bottom)

    crop = image.crop((x0, top, x1, bottom))
    return crop   # KHONG resize o day nua - luc train pipeline se tu Resize((224,224))


def crop_pins_from_image(image_path, out_dir, base_name=None, save_preview=False):
    """Cat 1 anh linh kien thanh N anh chan pin rieng, luu vao out_dir.
    Tra ve list duong dan cac file da luu."""
    image = Image.open(image_path).convert("RGB")
    gray = np.array(image.convert("L"))

    centers, y0, y1, pitch = find_pin_centers(
        gray, PIN_BAND_Y_RATIO, EXPECTED_PIN_COUNT,
        MIN_PIN_WIDTH_RATIO, MAX_PIN_WIDTH_RATIO,
    )

    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for i, cx in enumerate(centers):
        crop = crop_single_pin(
            image, gray, cx, y0, y1, pitch,
            CROP_WIDTH_RATIO, CROP_PAD_ABOVE_RATIO, CROP_PAD_BELOW_RATIO,
        )
        out_path = os.path.join(out_dir, f"{base_name}_pin{i+1:02d}.png")
        crop.save(out_path)
        saved.append(out_path)

    if save_preview:
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        h = image.size[1]
        top_pad = y0 - h * CROP_PAD_ABOVE_RATIO
        bottom_pad = y1 + h * CROP_PAD_BELOW_RATIO
        for i, cx in enumerate(centers):
            bbox = find_tight_pin_bbox(gray, cx, y0, y1, pitch)
            if bbox is not None:
                x0, x1 = bbox
            else:
                half_w = image.size[0] * CROP_WIDTH_RATIO * 0.5
                x0, x1 = cx - half_w, cx + half_w
            draw.rectangle([x0, top_pad, x1, bottom_pad], outline=(255, 0, 0), width=4)
            draw.text((cx - 10, top_pad - 30), str(i + 1), fill=(255, 0, 0))
        preview_path = os.path.join(out_dir, f"{base_name}_preview.png")
        preview.save(preview_path)
        saved.append(preview_path)

    return saved


def crop_pins_from_folder(input_dir, output_dir, save_preview=True):
    """Cat pin cho TAT CA anh nam truc tiep trong 1 thu muc (khong phan
    biet class/nhan gi ca - chi don gian la nhieu anh). Voi moi anh goc,
    ket qua duoc luu vao 1 thu muc con rieng trong output_dir (dat theo
    ten anh goc) de khong bi lan cac file pin cua nhieu anh khac nhau."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]

    if not image_files:
        print(f"Khong tim thay anh nao trong: {input_dir}")
        return

    total_pins = 0
    for fname in image_files:
        img_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        out_sub_dir = os.path.join(output_dir, base_name)
        try:
            saved = crop_pins_from_image(img_path, out_sub_dir, save_preview=save_preview)
            n_pins = len(saved) - (1 if save_preview else 0)
            total_pins += n_pins
            print(f"  {fname}: cat duoc {n_pins} chan -> {out_sub_dir}")
        except Exception as e:
            print(f"  [BO QUA] {fname}: {e}")

    print(f"\nXong. Da xu ly {len(image_files)} anh, tong cong {total_pins} anh chan pin, "
          f"luu tai: {output_dir}")


def process_dataset(input_dirs, output_root):
    """Cat pin hang loat cho ca dataset dang to chuc theo thu muc class
    (class1/, class2/,...) - GIU NGUYEN cau truc thu muc class trong
    output_root, de dung thang lam DATASET_DIRS cho pipeline few-shot
    classification o buoc huan luyen sau nay. Moi anh goc se sinh ra
    N anh chan pin, tat ca ke thua nhan (class) cua anh goc."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    total_in, total_out = 0, 0

    for d in input_dirs:
        for entry in os.scandir(d):
            if not entry.is_dir():
                continue
            cls = entry.name
            out_cls_dir = os.path.join(output_root, cls)
            for fname in os.listdir(entry.path):
                if not fname.lower().endswith(exts):
                    continue
                img_path = os.path.join(entry.path, fname)
                try:
                    saved = crop_pins_from_image(img_path, out_cls_dir)
                    total_in += 1
                    total_out += len(saved)
                except Exception as e:
                    print(f"  [BO QUA] {img_path}: {e}")

    print(f"Da xu ly {total_in} anh goc -> sinh ra {total_out} anh chan pin, "
          f"luu tai: {output_root}")


if __name__ == "__main__":
    # INPUT_PATH la 1 FILE anh -> chi cat 1 anh do.
    # INPUT_PATH la 1 THU MUC -> tu dong cat het TAT CA anh trong thu muc do,
    # moi anh 1 thu muc con rieng trong OUTPUT_DIR.
    if os.path.isdir(INPUT_PATH):
        crop_pins_from_folder(INPUT_PATH, OUTPUT_DIR, save_preview=True)
    else:
        saved = crop_pins_from_image(INPUT_PATH, OUTPUT_DIR, save_preview=True)
        print(f"Da luu {len(saved)} file vao: {OUTPUT_DIR}")
        for path in saved:
            print(" -", path)

    # Neu dataset da chia san theo thu muc class (OK/, NG/...) va muon
    # GIU NGUYEN cau truc class do trong output (de dung thang lam
    # DATASET_DIRS cho pipeline few-shot classification), dung ham
    # process_dataset() thay vi 2 cach tren:
    #
    # process_dataset(
    #     input_dirs=[r"D:\...\OK_drop\train", r"D:\...\OK_drop\val"],
    #     output_root=r"D:\...\OK_drop_pins",
    # )