"""Detect + crop bằng threshold cổ điển. Duyệt 1 thư mục, lưu ra thư mục cùng tên + "_drop"."""

import os
import cv2
import numpy as np

METHOD = "threshold"          # "threshold" | "background_subtraction"
BACKGROUND_REF_PATH = ""
DIFF_THRESHOLD = 1
FIXED_THRESHOLD = 60        # ngưỡng sáng 0-255, chỉnh tay. Với ảnh mẫu: pin ăn được ở khoảng 55-80,
                                # dưới ~45 dễ dính sang vật/khe kế bên, trên ~90 bắt đầu mất pin
MORPH_KERNEL_SIZE = 7
MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.98
BORDER_TOUCH_MAX = 0.5        # contour chạm viền nhiều = nền
MERGE_GAP_PX = 40             # gộp các mảnh cách nhau <= X px thành 1 vật
CROP_PADDING = 15

INPUT_DIR = r"D:\TongHop\RTC Technologi\HZT.Top"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_bg = None


def _clean(m):
    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)


def _border_ratio(c, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
    b = np.zeros(shape, bool)
    b[:2, :] = b[-2:, :] = b[:, :2] = b[:, -2:] = True
    return np.count_nonzero((m > 0) & b) / b.sum()


def _gap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    gx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return max(gx, gy)


def _union(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x, y = min(x1, x2), min(y1, y2)
    return x, y, max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y


def _select(mask, img_area):
    """Trả về (bbox, area, is_fallback) hoặc (None, 0, True)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if MIN_AREA_RATIO * img_area <= cv2.contourArea(c) <= MAX_AREA_RATIO * img_area]
    if not valid:
        return None, 0, True

    shape = mask.shape[:2]
    clean = [c for c in valid if _border_ratio(c, shape) <= BORDER_TOUCH_MAX]
    is_fallback = not clean
    if not clean:
        clean = sorted(valid, key=lambda c: _border_ratio(c, shape))[:1]

    boxes = [cv2.boundingRect(c) for c in clean]
    areas = [cv2.contourArea(c) for c in clean]
    box = boxes[areas.index(max(areas))]
    changed = True
    used = {areas.index(max(areas))}
    while changed:
        changed = False
        for i, b in enumerate(boxes):
            if i not in used and _gap(box, b) <= MERGE_GAP_PX:
                used.add(i)
                box = _union(box, b)
                changed = True

    return box, sum(areas[i] for i in used), is_fallback


def detect_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = img.shape[0] * img.shape[1]

    global _bg
    if METHOD == "background_subtraction" and BACKGROUND_REF_PATH:
        if _bg is None:
            _bg = cv2.cvtColor(cv2.imread(BACKGROUND_REF_PATH), cv2.COLOR_BGR2GRAY)
        bg = cv2.resize(_bg, (gray.shape[1], gray.shape[0])) if _bg.shape != gray.shape else _bg
        _, m = cv2.threshold(cv2.absdiff(gray, bg), DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        box, _, _ = _select(_clean(m), area)
        return box

    candidates = []
    for inv in (False, True):
        t = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        _, m = cv2.threshold(gray, FIXED_THRESHOLD, 255, t)
        b, a, fb = _select(_clean(m), area)
        if b is not None:
            candidates.append((b, a, fb))

    if not candidates:
        return None
    clean_c = [c for c in candidates if not c[2]]
    pool = clean_c if clean_c else candidates
    return max(pool, key=lambda t: t[1])[0]


def crop(img, bbox, pad=CROP_PADDING):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    return img[y1:y2, x1:x2]


def run(input_dir):
    out_dir = input_dir.rstrip("\\/") + "_drop"
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXT)]

    ok = 0
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None:
            print(f"[LỖI ĐỌC] {f}")
            continue
        bbox = detect_bbox(img)
        if bbox is None:
            print(f"[FAIL] {f}")
            continue
        cv2.imwrite(os.path.join(out_dir, f), crop(img, bbox))
        print(f"[OK] {f}")
        ok += 1

    print(f"\n{ok}/{len(files)} thành công -> {out_dir}")


if __name__ == "__main__":
    run(INPUT_DIR)