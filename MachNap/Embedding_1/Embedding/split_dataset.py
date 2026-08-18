import os
import random
import shutil

# ------------------- Cấu hình -------------------
BASE_DIR = r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

VAL_RATIO = 0.2          # 20% cho val, 80% cho train
SEED = 42                # cố định để lần chạy sau ra kết quả giống nhau
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ------------------------------------------------

random.seed(SEED)


def is_part_folder(name):
    """Chỉ lấy các folder tên Part1, Part2, ... trong BASE_DIR."""
    path = os.path.join(BASE_DIR, name)
    return os.path.isdir(path) and name.lower().startswith("part") \
        and name[4:].isdigit()


def split_one_part(part_name):
    src = os.path.join(BASE_DIR, part_name)

    # Lấy danh sách ảnh
    images = [f for f in os.listdir(src)
              if f.lower().endswith(IMG_EXTS)]
    if not images:
        print(f"[SKIP] {part_name}: không có ảnh.")
        return

    random.shuffle(images)
    n_val = round(len(images) * VAL_RATIO)
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    # Tạo folder đích
    train_part = os.path.join(TRAIN_DIR, part_name)
    val_part = os.path.join(VAL_DIR, part_name)
    os.makedirs(train_part, exist_ok=True)
    os.makedirs(val_part, exist_ok=True)

    for f in train_imgs:
        shutil.copy2(os.path.join(src, f), os.path.join(train_part, f))
    for f in val_imgs:
        shutil.copy2(os.path.join(src, f), os.path.join(val_part, f))

    print(f"[{part_name}] tổng {len(images):>4} -> "
          f"train {len(train_imgs):>4} | val {len(val_imgs):>4}")


if __name__ == "__main__":
    parts = sorted(
        [d for d in os.listdir(BASE_DIR) if is_part_folder(d)],
        key=lambda x: int(x[4:])          # sắp Part1, Part2, ... đúng thứ tự số
    )

    if not parts:
        print("Không tìm thấy folder Part nào.")
    else:
        print(f"Tìm thấy {len(parts)} part: {parts}\n")
        total_train = total_val = 0
        for p in parts:
            split_one_part(p)

        print("\nXong. Cấu trúc:")
        print(f"  {TRAIN_DIR}\\Part1..Part{len(parts)}")
        print(f"  {VAL_DIR}\\Part1..Part{len(parts)}")