# -*- coding: utf-8 -*-
"""
Script kiểm tra dataset YOLO: random ~100 ảnh, vẽ bounding box lên ảnh
và hiển thị lần lượt để xem label có đúng không.

Điều khiển khi cửa sổ ảnh hiện lên:
    - Phím bất kỳ / SPACE : ảnh tiếp theo
    - B                   : quay lại ảnh trước
    - Q hoặc ESC          : thoát

Cài thư viện (nếu chưa có):
    pip install opencv-python pyyaml

Chạy:
    python check_labels.py
"""

import os
import random
import cv2

# ==================== CẤU HÌNH ====================
# Trỏ vào dataset muốn kiểm tra (dataset đã augment)
DATASET_DIR = r"D:\TongHop\Hoc_Tren_Truong\DuAnChoThayVan\GiaoThongThongMinh\Dataset\DatasetTuyenQuang\merged_dataset_augmented"
SPLIT = "train"          # kiểm tra tập nào: train / valid / test
N_SAMPLES = 100          # số ảnh random để xem
MAX_DISPLAY = 1000       # ảnh to hơn sẽ thu nhỏ về cạnh dài này khi hiển thị
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ==================================================

# Bảng màu cho từng class (BGR)
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0),
]


def load_class_names(dataset_dir):
    """Đọc tên class từ data.yaml (nếu có)."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):          # dạng {0: 'car', 1: 'bus'}
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(n) for i, n in enumerate(names)}  # dạng list
    except Exception as e:
        print(f"⚠️ Không đọc được data.yaml ({e}), sẽ hiển thị class id thay vì tên.")
        return {}


def parse_label_line(parts):
    """Hỗ trợ cả detection (5 số) lẫn polygon/segmentation (class + các cặp x y).
    Polygon được chuyển về box bao quanh. Trả về (cid, xc, yc, w, h) hoặc None."""
    if len(parts) == 5:
        cid = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:])
        return cid, xc, yc, bw, bh
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        cid = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return cid, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min
    return None


def draw_labels(image, label_path, class_names):
    """Vẽ các bounding box YOLO lên ảnh. Trả về (ảnh đã vẽ, số box)."""
    h, w = image.shape[:2]
    n_boxes = 0
    if not os.path.exists(label_path):
        return image, 0

    with open(label_path, "r") as f:
        for line in f:
            parsed = parse_label_line(line.strip().split())
            if parsed is None:
                continue
            cid, xc, yc, bw, bh = parsed
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS[cid % len(COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label_text = class_names.get(cid, f"class {cid}")
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 6 if y1 - th - 8 > 0 else y1 + th + 8
            cv2.rectangle(image, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
            cv2.putText(image, label_text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            n_boxes += 1

    return image, n_boxes


def resize_for_display(image):
    """Thu nhỏ ảnh nếu quá to để vừa màn hình."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DISPLAY:
        return image
    scale = MAX_DISPLAY / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def main():
    images_dir = os.path.join(DATASET_DIR, SPLIT, "images")
    labels_dir = os.path.join(DATASET_DIR, SPLIT, "labels")

    if not os.path.isdir(images_dir):
        print(f"❌ Không tìm thấy '{images_dir}'. Kiểm tra lại DATASET_DIR và SPLIT.")
        return

    class_names = load_class_names(DATASET_DIR)

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)]
    if not files:
        print(f"❌ Không có ảnh nào trong '{images_dir}'.")
        return

    samples = random.sample(files, min(N_SAMPLES, len(files)))
    print(f"🔍 Kiểm tra {len(samples)} ảnh random trong '{SPLIT}'.")
    print("Phím bất kỳ: ảnh tiếp | B: quay lại | Q/ESC: thoát\n")

    idx = 0
    while 0 <= idx < len(samples):
        filename = samples[idx]
        name, _ = os.path.splitext(filename)
        image = cv2.imread(os.path.join(images_dir, filename))
        if image is None:
            print(f"⚠️ Không đọc được {filename}, bỏ qua.")
            idx += 1
            continue

        label_path = os.path.join(labels_dir, name + ".txt")
        image, n_boxes = draw_labels(image, label_path, class_names)
        image = resize_for_display(image)

        # Thông tin ảnh ở góc trên
        info = f"[{idx + 1}/{len(samples)}] {filename} - {n_boxes} box"
        cv2.putText(image, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Kiem tra label - Q/ESC de thoat", image)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):   # Q hoặc ESC
            break
        elif key in (ord("b"), ord("B")):     # quay lại
            idx = max(0, idx - 1)
        else:                                  # ảnh tiếp theo
            idx += 1

    cv2.destroyAllWindows()
    print("✅ Đã kiểm tra xong.")


if __name__ == "__main__":
    main()