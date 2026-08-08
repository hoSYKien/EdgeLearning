# -*- coding: utf-8 -*-
"""
Script chẩn đoán label: xác định box bị thiếu là do LABEL GỐC thiếu sẵn
hay do AUGMENTATION làm mất.

So sánh:
    1. Số ảnh gốc có 0 box (label trống hoặc không có file label)
    2. Phân bố class trong dataset GỐC vs dataset ĐÃ AUGMENT
    3. Các trường hợp ảnh _aug bị mất box so với ảnh gốc của nó

Chạy:
    python diagnose_labels.py
"""

import os
import re
from collections import Counter

# ==================== CẤU HÌNH ====================
ORIGINAL_DIR = r"D:\TongHop\Hoc_Tren_Truong\DuAnChoThayVan\GiaoThongThongMinh\Dataset\DatasetTuyenQuang\merged_dataset"
AUGMENTED_DIR = ORIGINAL_DIR + "_augmented"
SPLIT = "train"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ==================================================


def load_class_names(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        return {}


def count_labels(labels_dir, name):
    """Trả về Counter class id trong 1 file label. None nếu file không tồn tại."""
    path = os.path.join(labels_dir, name + ".txt")
    if not os.path.exists(path):
        return None
    c = Counter()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                c[int(float(parts[0]))] += 1
    return c


def class_str(counter, names):
    if not counter:
        return "(0 box)"
    return ", ".join(
        f"{names.get(cid, f'class {cid}')}: {n}" for cid, n in sorted(counter.items())
    )


def main():
    names = load_class_names(ORIGINAL_DIR)
    orig_images = os.path.join(ORIGINAL_DIR, SPLIT, "images")
    orig_labels = os.path.join(ORIGINAL_DIR, SPLIT, "labels")
    aug_labels = os.path.join(AUGMENTED_DIR, SPLIT, "labels")

    if not os.path.isdir(orig_images):
        print(f"❌ Không tìm thấy '{orig_images}'. Kiểm tra lại ORIGINAL_DIR.")
        return

    files = [f for f in os.listdir(orig_images) if f.lower().endswith(IMAGE_EXTS)]

    # ========== 1. Kiểm tra label GỐC ==========
    no_label_file, empty_label, total_orig = [], [], Counter()
    orig_counts = {}
    for filename in files:
        name, _ = os.path.splitext(filename)
        c = count_labels(orig_labels, name)
        if c is None:
            no_label_file.append(filename)
            orig_counts[name] = Counter()
            continue
        orig_counts[name] = c
        if sum(c.values()) == 0:
            empty_label.append(filename)
        total_orig.update(c)

    print("=" * 60)
    print(f"📊 DATASET GỐC ({SPLIT}): {len(files)} ảnh")
    print(f"   - Không có file label : {len(no_label_file)} ảnh")
    print(f"   - File label trống    : {len(empty_label)} ảnh")
    print(f"   - Phân bố class       : {class_str(total_orig, names)}")

    if no_label_file[:5]:
        print("   Ví dụ ảnh KHÔNG có file label:")
        for f in no_label_file[:5]:
            print(f"      {f}")
    if empty_label[:5]:
        print("   Ví dụ ảnh có label TRỐNG:")
        for f in empty_label[:5]:
            print(f"      {f}")

    # ========== 2. So sánh với dataset ĐÃ AUGMENT ==========
    if not os.path.isdir(aug_labels):
        print("\n⚠️ Không tìm thấy dataset augmented, bỏ qua phần so sánh.")
        return

    total_aug = Counter()
    lost_cases = []
    aug_pattern = re.compile(r"^(.+)_aug(\d+)$")

    for f in os.listdir(aug_labels):
        if not f.endswith(".txt"):
            continue
        name = f[:-4]
        c = count_labels(aug_labels, name)
        total_aug.update(c)

        m = aug_pattern.match(name)
        if m:
            base = m.group(1)
            orig_c = orig_counts.get(base)
            if orig_c is not None:
                n_orig, n_aug = sum(orig_c.values()), sum(c.values())
                # Cảnh báo nếu bản aug mất hơn nửa số box so với gốc
                if n_orig > 0 and n_aug < n_orig / 2:
                    lost_cases.append((name, n_orig, n_aug))

    print("\n" + "=" * 60)
    print(f"📊 DATASET ĐÃ AUGMENT ({SPLIT}):")
    print(f"   - Phân bố class: {class_str(total_aug, names)}")
    print(f"\n🔎 Số bản _aug bị mất hơn 50% box so với ảnh gốc: {len(lost_cases)}")
    for name, n_orig, n_aug in lost_cases[:10]:
        print(f"   {name}: gốc {n_orig} box -> aug còn {n_aug} box")
    if len(lost_cases) > 10:
        print(f"   ... và {len(lost_cases) - 10} trường hợp khác")

    # ========== 3. Kết luận ==========
    print("\n" + "=" * 60)
    print("💡 CÁCH ĐỌC KẾT QUẢ:")
    print("   - Nếu nhiều ảnh gốc 0 box / thiếu file label -> vấn đề nằm ở")
    print("     DATASET GỐC (label thiếu từ trước khi augment).")
    print("   - Nếu ảnh gốc đủ box nhưng nhiều bản _aug mất box -> vấn đề")
    print("     nằm ở AUGMENTATION, gửi kết quả này cho mình xem.")


if __name__ == "__main__":
    main()