r"""
Duyệt qua TẤT CẢ ảnh trong 1 thư mục, crop mỗi ảnh thành 4 vùng theo bố cục
đã khoanh tay, lưu vào 4 THƯ MỤC RIÊNG (mỗi vùng 1 thư mục) - tên thư mục =
tên thư mục gốc + "_crop_<số vùng>".

Ví dụ INPUT_DIR = "...\NG" sẽ tạo ra (cùng cấp với NG):
    ...\NG_crop_1\   (toàn bộ ảnh đã crop vùng 1 - gần kẹp trái)
    ...\NG_crop_2\   (vùng 2 - gờ giữa)
    ...\NG_crop_3\   (vùng 3 - gần kẹp phải)
    ...\NG_crop_4\   (vùng 4 - gần chân pin)
Tên file bên trong giữ nguyên tên ảnh gốc.

Mỗi vùng tính theo L (chiều dài - trục ngang) và W (chiều rộng - trục dọc)
của TỪNG ảnh đang xử lý - tự đúng tỉ lệ dù các ảnh to nhỏ khác nhau.

Cách dùng: sửa INPUT_DIR bên dưới rồi chạy:
    python crop_4_regions.py
"""
import os
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_regions(L, W):
    """Trả về box (x0, y0, x1, y1) từng vùng, tính theo L, W của ảnh đang crop."""
    return {
        "1": (0, 0, L / 4, 2 * W / 3),          # gần kẹp trái
        "2": (L / 4, 0, 3 * L / 4, 2 * W / 3),  # gờ giữa
        "3": (3 * L / 4, 0, L, 2 * W / 3),      # gần kẹp phải
        "4": (0, 2 * W / 3, L, W),              # gần chân pin
    }


# Nhãn mô tả cho từng vùng - dùng làm hậu tố cho cả tên thư mục và tên file
REGION_LABELS = {
    "1": "left",
    "2": "center",
    "3": "right",
    "4": "bot",
}


INPUT_DIR = r"D:\TongHop\RTC Technologi\HZT\HZT.Bottom.Split"


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {INPUT_DIR}")

    base_name = os.path.basename(os.path.normpath(INPUT_DIR))
    parent_dir = os.path.dirname(os.path.normpath(INPUT_DIR))

    # 4 thư mục output riêng - tên = tên thư mục gốc + "_crop_<số vùng>"
    region_names = ["1", "2", "3", "4"]
    output_dirs = {name: os.path.join(parent_dir, f"{base_name}_crop_{name}_{REGION_LABELS[name]}")
                    for name in region_names}
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTS)]
    print(f"Tìm thấy {len(image_files)} ảnh trong {INPUT_DIR}")
    for name, d in output_dirs.items():
        print(f"  Vùng {name} -> {d}")
    print()

    for fname in image_files:
        image_path = os.path.join(INPUT_DIR, fname)
        image = Image.open(image_path).convert("RGB")
        L, W = image.size
        ext = os.path.splitext(fname)[1] or ".jpg"
        name_no_ext = os.path.splitext(fname)[0]

        for region_name, (x0, y0, x1, y1) in get_regions(L, W).items():
            box = (int(x0), int(y0), int(x1), int(y1))
            crop = image.crop(box)
            out_path = os.path.join(output_dirs[region_name],
                                     f"{name_no_ext}_crop_{REGION_LABELS[region_name]}{ext}")
            crop.save(out_path)

        print(f"Đã crop: {fname}")

    print(f"\nXong! Đã xử lý {len(image_files)} ảnh, lưu vào 4 thư mục trên.")


if __name__ == "__main__":
    main()