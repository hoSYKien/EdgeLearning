"""
Batch crop dataset: Parts -> Parts1 (giữ nền trong box) + Parts2 (nền tím).

NỀN: lấy từ thư mục Part0 (ảnh nền trống, không có vật).
     Trung bình tất cả ảnh trong Part0 -> ảnh nền, giống hệt learn_background()
     bên code camera. Part0 KHÔNG được xử lý và KHÔNG xuất ra output.

Pipeline (mượn từ conveyor script):
  [1] Color diff per-channel max
  [2] Khử bóng (chroma + tỉ lệ độ sáng)
  [3] Hysteresis 2 ngưỡng (chống loang)
  [4] Morphology + lấp lỗ kín
  [5] Chọn blob lớn nhất -> bbox -> crop

ĐẦU RA:
  Parts1/<Part>/<ten_anh>.png   - crop bbox, GIỮ NGUYÊN nền bên trong box
  Parts2/<Part>/<ten_anh>.png   - crop bbox, nền bị thay bằng MÀU TÍM

Cách chạy:
    python crop_parts.py
Chạy thử vài ảnh trước khi chạy full:
    đặt DEBUG_PREVIEW = True
"""

import os
import shutil

import cv2
import numpy as np

# ============================================================
# 1. Cấu hình - SỬA Ở ĐÂY
# ============================================================
PARTS_ROOT = r"D:\TongHop\RTC Technologi\G8\dataset\Parts"
BG_FOLDER = r"D:\TongHop\RTC Technologi\G8\dataset\Parts\Part0"   # ảnh NỀN TRỐNG
OUT_DIR_1 = r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1"        # giữ nền trong box
OUT_DIR_2 = r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts2"        # nền tím

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Thư mục bỏ qua khi duyệt (Part0 là nền, không phải class)
SKIP_FOLDERS = {"Part0"}

# --- Ngưỡng pipeline (mượn từ conveyor script) ---
THRESH_STRONG = 30
THRESH_WEAK = 15

CHROMA_TOL = 0.04
SHADOW_DIM = (0.3, 0.95)
SHADOW_MAX_DIFF = 70

MIN_AREA_RATIO = 0.002      # blob nhỏ hơn tỉ lệ này của ảnh -> bỏ (nhiễu)
MAX_AREA_RATIO = 0.95       # blob to hơn mức này -> coi như lỗi nền, bỏ

MORPH_OPEN_KSIZE = (5, 5)
MORPH_CLOSE_KSIZE = (7, 7)

FILL_HOLES = True           # lấp lỗ kín bên trong vật
USE_CONVEX_HULL = False     # True = bọc lồi vật (mất chỗ lõm thật). Thường để False.
SOLIDITY_MIN = 0.97         # chỉ dùng khi USE_CONVEX_HULL = True

CROP_PADDING = 15           # đệm quanh bbox (pixel). 0 = ôm sát mép mask.

# --- Riêng cho Parts2 (nền tím) ---
PURPLE_BGR = (255, 0, 255)  # màu nền thay thế, dạng BGR. (255,0,255) = tím magenta
MASK_SMOOTH = 9             # làm mượt mép mask trước khi xoá nền. 0 = tắt.
MASK_ERODE = 2              # ăn mòn mép vài pixel để không sót viền nền. 0 = tắt.

# --- Chạy thử ---
DEBUG_PREVIEW = False       # True = hiện cửa sổ xem kết quả từng ảnh, không ghi file
DEBUG_SHOW_BG = False       # True = hiện ảnh nền đã học rồi mới chạy tiếp
DEBUG_MAX_IMAGES = 12       # số ảnh xem thử mỗi folder khi DEBUG_PREVIEW
PREVIEW_MAX_WIDTH = 1400

CLEAN_OUTPUT = False        # True = xoá sạch Parts1/Parts2 trước khi chạy


# ============================================================
# 2. Tiện ích
# ============================================================
def list_images(folder):
    return sorted([f for f in os.listdir(folder)
                   if f.lower().endswith(IMG_EXTENSIONS)])


def imread_unicode(path):
    """cv2.imread không đọc được đường dẫn có ký tự unicode trên Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def resize_for_display(img, max_width=PREVIEW_MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / float(w)
    return cv2.resize(img, (int(w * s), int(h * s)))


# ============================================================
# 3. Học nền từ Part0
# ============================================================
def learn_background(bg_folder):
    """Trung bình tất cả ảnh nền trong Part0 -> khử nhiễu cảm biến.
    Dùng mean (không phải median) vì mọi ảnh ở đây đều là nền trống thật."""
    if not os.path.isdir(bg_folder):
        raise FileNotFoundError(f"Không tìm thấy thư mục nền: {bg_folder}")

    files = list_images(bg_folder)
    if not files:
        raise RuntimeError(f"Thư mục nền không có ảnh nào: {bg_folder}")

    print(f"Học nền từ {len(files)} ảnh trong: {bg_folder}")

    acc = None
    ref_shape = None
    n = 0
    for f in files:
        img = imread_unicode(os.path.join(bg_folder, f))
        if img is None:
            print(f"  LỖI đọc ảnh nền: {f}")
            continue
        if ref_shape is None:
            ref_shape = img.shape
            acc = np.zeros(ref_shape, np.float64)
        elif img.shape != ref_shape:
            img = cv2.resize(img, (ref_shape[1], ref_shape[0]))
        acc += img
        n += 1

    if n == 0:
        raise RuntimeError("Không đọc được ảnh nền nào.")

    bg = (acc / n).astype(np.uint8)
    print(f"Xong! Nền dựng từ {n} ảnh, kích thước {bg.shape[1]}x{bg.shape[0]}\n")
    return bg


class BackgroundModel:
    """Giữ sẵn các mảng dẫn xuất từ nền để khỏi tính lại cho mỗi ảnh."""

    def __init__(self, bg_bgr):
        self.bg_uint8 = bg_bgr
        self.bg = bg_bgr.astype(np.float32)

        b, g, r = cv2.split(self.bg)
        self.bg_sum = b + g + r + 3.0
        self.bg_sum_3d = cv2.merge([self.bg_sum, self.bg_sum, self.bg_sum])
        self.bg_plus_1 = self.bg + 1.0
        self.shadow_dim_low = SHADOW_DIM[0] * self.bg_sum
        self.shadow_dim_high = SHADOW_DIM[1] * self.bg_sum

    def fit_to(self, shape):
        """Ảnh vật khác kích thước nền -> dựng lại model theo đúng cỡ (cache 1 lần)."""
        if self.bg_uint8.shape[:2] == shape[:2]:
            return self
        if getattr(self, "_cached_shape", None) != shape[:2]:
            resized = cv2.resize(self.bg_uint8, (shape[1], shape[0]))
            self._cached = BackgroundModel(resized)
            self._cached_shape = shape[:2]
            print(f"    (ảnh khác cỡ nền -> đã resize nền về {shape[1]}x{shape[0]})")
        return self._cached


# ============================================================
# 4. Tạo mask vật thể
# ============================================================
def mask_from_background(img_bgr, bgm):
    """Pipeline [1]-[3]: diff màu -> khử bóng -> hysteresis -> mask nhị phân."""
    blur = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    img_f = blur.astype(np.float32)

    # [1] Diff per-channel max
    diff_c = cv2.absdiff(img_f, bgm.bg)
    b, g, r = cv2.split(diff_c)
    diff = cv2.max(cv2.max(b, g), r)

    # [2] Khử bóng: cùng sắc màu + tối đi vừa phải -> là bóng, không phải vật
    ib, ig, ir = cv2.split(img_f)
    S_img = ib + ig + ir + 3.0
    S_img_3d = cv2.merge([S_img, S_img, S_img])

    term_img = (img_f + 1.0) * bgm.bg_sum_3d
    term_bg = bgm.bg_plus_1 * S_img_3d
    chroma_diff = cv2.absdiff(term_img, term_bg)
    cb, cg, cr = cv2.split(chroma_diff)
    max_chroma = cv2.max(cv2.max(cb, cg), cr)

    same_color = max_chroma < (CHROMA_TOL * bgm.bg_sum) * S_img
    dim_moderate = (S_img > bgm.shadow_dim_low) & (S_img < bgm.shadow_dim_high)
    diff[same_color & dim_moderate & (diff < SHADOW_MAX_DIFF)] = 0

    # [3] Hysteresis: giữ vùng weak nào có chạm seed strong
    strong = (diff > THRESH_STRONG).astype(np.uint8)
    weak = (diff > THRESH_WEAK).astype(np.uint8)
    if cv2.countNonZero(strong) == 0:
        return np.zeros(img_bgr.shape[:2], np.uint8)

    n_lbl, lbl = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(lbl[strong > 0])
    strong_labels = strong_labels[strong_labels != 0]
    keep = np.zeros(n_lbl, dtype=bool)
    keep[strong_labels] = True
    return (keep[lbl].astype(np.uint8) * 255)


def fill_holes(mask):
    """Flood-fill từ ngoài vào; vùng không tràn tới = lỗ kín -> tô đầy.
    Đệm 1px quanh mask để flood-fill luôn có đường đi ở biên."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    ff = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    filled = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
    return filled[1:-1, 1:-1]


def refine_mask(mask):
    """Morphology dọn nhiễu + nối khe hở + lấp lỗ."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN_KSIZE)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KSIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    if FILL_HOLES:
        mask = fill_holes(mask)
    return mask


def largest_blob(mask, img_shape):
    """Giữ lại DUY NHẤT blob lớn nhất hợp lệ. Trả (mask_blob, bbox) hoặc (None, None)."""
    area_img = img_shape[0] * img_shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if MIN_AREA_RATIO * area_img <= cv2.contourArea(c) <= MAX_AREA_RATIO * area_img]
    if not valid:
        return None, None

    c = max(valid, key=cv2.contourArea)

    if USE_CONVEX_HULL:
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and cv2.contourArea(c) / hull_area >= SOLIDITY_MIN:
            c = hull

    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out, cv2.boundingRect(c)


def polish_mask_edges(mask):
    """Mượt mép + ăn mòn nhẹ để khi thay nền không sót viền nền quanh vật."""
    if MASK_SMOOTH >= 3:
        k = int(MASK_SMOOTH) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if MASK_ERODE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (MASK_ERODE * 2 + 1, MASK_ERODE * 2 + 1))
        mask = cv2.erode(mask, k)
    return mask


# ============================================================
# 5. Xử lý 1 ảnh
# ============================================================
def process_image(img_bgr, bgm):
    """Trả về (crop_giu_nen, crop_nen_tim) hoặc (None, None) nếu không tìm được vật."""
    mask = refine_mask(mask_from_background(img_bgr, bgm))

    blob, bbox = largest_blob(mask, img_bgr.shape[:2])
    if blob is None:
        return None, None

    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1 = max(0, x - CROP_PADDING)
    y1 = max(0, y - CROP_PADDING)
    x2 = min(W, x + w + CROP_PADDING)
    y2 = min(H, y + h + CROP_PADDING)

    # --- Parts1: crop nguyên xi, còn nguyên nền trong box ---
    crop1 = img_bgr[y1:y2, x1:x2].copy()

    # --- Parts2: cùng khung crop, nhưng nền -> màu tím ---
    blob = polish_mask_edges(blob)
    mask_crop = blob[y1:y2, x1:x2]
    crop2 = np.full_like(crop1, PURPLE_BGR, dtype=np.uint8)
    np.copyto(crop2, crop1, where=(mask_crop[:, :, None] > 0))

    return crop1, crop2


# ============================================================
# 6. Duyệt toàn bộ dataset
# ============================================================
def process_folder(folder, rel_name, bgm, stats):
    files = list_images(folder)
    if not files:
        return

    print(f"\n[{rel_name}] {len(files)} ảnh")

    out1 = os.path.join(OUT_DIR_1, rel_name)
    out2 = os.path.join(OUT_DIR_2, rel_name)
    if not DEBUG_PREVIEW:
        os.makedirs(out1, exist_ok=True)
        os.makedirs(out2, exist_ok=True)

    shown = 0
    for fname in files:
        img = imread_unicode(os.path.join(folder, fname))
        if img is None:
            print(f"    LỖI đọc: {fname}")
            stats["error"] += 1
            continue

        crop1, crop2 = process_image(img, bgm.fit_to(img.shape))

        if crop1 is None:
            print(f"    BỎ QUA (không tìm thấy vật): {fname}")
            stats["failed"] += 1
            continue

        if DEBUG_PREVIEW:
            if shown < DEBUG_MAX_IMAGES:
                th = max(crop1.shape[0], crop2.shape[0])
                pad1 = cv2.copyMakeBorder(crop1, 0, th - crop1.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
                pad2 = cv2.copyMakeBorder(crop2, 0, th - crop2.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
                combo = np.hstack([pad1, pad2])
                cv2.imshow(f"{rel_name} - trai: Parts1 | phai: Parts2  (phim=tiep, ESC=dung)",
                           resize_for_display(combo))
                k = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                shown += 1
                if k == 27:
                    return "stop"
        else:
            stem = os.path.splitext(fname)[0]
            imwrite_unicode(os.path.join(out1, stem + ".png"), crop1)
            imwrite_unicode(os.path.join(out2, stem + ".png"), crop2)

        stats["ok"] += 1


def main():
    if not os.path.isdir(PARTS_ROOT):
        print(f"Không tìm thấy thư mục: {PARTS_ROOT}")
        return

    bg = learn_background(BG_FOLDER)
    if DEBUG_SHOW_BG:
        cv2.imshow("Nen da hoc tu Part0 (phim bat ky de tiep tuc)",
                   resize_for_display(bg))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    bgm = BackgroundModel(bg)

    if CLEAN_OUTPUT and not DEBUG_PREVIEW:
        for d in (OUT_DIR_1, OUT_DIR_2):
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"Đã xoá: {d}")

    subfolders = sorted([d for d in os.listdir(PARTS_ROOT)
                         if os.path.isdir(os.path.join(PARTS_ROOT, d))
                         and d not in SKIP_FOLDERS])

    stats = {"ok": 0, "failed": 0, "error": 0}

    print(f"Tìm thấy {len(subfolders)} Part: {', '.join(subfolders)}")
    for sub in subfolders:
        r = process_folder(os.path.join(PARTS_ROOT, sub), sub, bgm, stats)
        if r == "stop":
            print("Đã nhấn ESC - dừng.")
            break

    print("\n" + "=" * 50)
    print(f"Thành công  : {stats['ok']}")
    print(f"Không detect: {stats['failed']}")
    print(f"Lỗi đọc file: {stats['error']}")
    if not DEBUG_PREVIEW:
        print(f"\nParts1 (giữ nền): {OUT_DIR_1}")
        print(f"Parts2 (nền tím): {OUT_DIR_2}")


if __name__ == "__main__":
    main()