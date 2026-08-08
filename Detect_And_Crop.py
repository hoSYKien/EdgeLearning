"""
KIẾN TRÚC CHUNG: Detect + Crop sản phẩm bằng ngưỡng màu (thresholding) + blob
detection (contour), TRƯỚC khi đưa ảnh vào pipeline train phân loại.

Ý TƯỞNG:
- Không cần train gì cả (không phải deep learning) - chỉ dùng xử lý ảnh cổ
  điển (classical CV), phù hợp khi sản phẩm có độ tương phản rõ với nền
  (nền tối/sáng khác biệt rõ với sản phẩm).
- Các bước: Grayscale -> Threshold (Otsu tự động hoặc ngưỡng cố định) ->
  Morphological cleanup (lọc nhiễu) -> Tìm contour lớn nhất -> Bounding box
  -> Crop + lưu ảnh mới.

CÁCH ÁP DỤNG CHO BÀI TOÁN MỚI:
Chỉ cần sửa phần CONFIG bên dưới (đường dẫn, ngưỡng nếu ảnh khác biệt).
Nếu Otsu tự động không ra kết quả tốt (nền và sản phẩm độ sáng gần nhau),
đổi USE_OTSU = False và tự chỉnh MANUAL_THRESHOLD.

Cách chạy:
    python detect_and_crop_product.py
"""

import os

import cv2
import numpy as np

# ==========================================================================
# CONFIG
# ==========================================================================

# Chế độ chạy: "preview" = xử lý thử 1 ảnh, HIỂN THỊ từng bước lên màn hình
# để kiểm tra trước khi chạy hàng loạt. "batch" = xử lý toàn bộ thư mục.
MODE = "preview"   # "preview" | "batch"

# --- Dùng cho MODE = "preview" ---
PREVIEW_IMAGE_PATH = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\Goi hut am\test\type_1\Image_0027_20260330073942.bmp"

# True  = chỉ hiển thị (cv2.imshow), KHÔNG lưu file gì cả.
# False = hiển thị VÀ lưu ra PREVIEW_OUTPUT_DIR (giống hành vi cũ).
SHOW_PREVIEW_ONLY = True

# Chỉ dùng khi SHOW_PREVIEW_ONLY = False (muốn lưu ảnh minh hoạ ra đĩa)
PREVIEW_OUTPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\preview_crop"

# Kích thước tối đa (chiều rộng, tính bằng px) khi hiển thị lên màn hình,
# tránh ảnh gốc quá lớn tràn màn hình. Ảnh sẽ được resize (chỉ để hiển thị,
# không ảnh hưởng đến kết quả xử lý/crop thật).
PREVIEW_DISPLAY_MAX_WIDTH = 700

# --- Dùng cho MODE = "batch" ---
# Cấu trúc INPUT: mỗi thư mục con là 1 class (giống ImageFolder)
#   input_dir/class1/anh1.jpg, input_dir/class2/anh2.jpg, ...
# Script sẽ tạo lại ĐÚNG cấu trúc thư mục đó ở OUTPUT_DIR, nhưng ảnh đã crop.
BATCH_INPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset\train"
BATCH_OUTPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset_cropped\train"

# --- Tham số threshold + blob detection ---
USE_OTSU = True              # True = tự động chọn ngưỡng; False = dùng MANUAL_THRESHOLD
MANUAL_THRESHOLD = 178        # chỉ dùng khi USE_OTSU = False
INVERT_THRESHOLD = False      # True nếu sản phẩm TỐI hơn nền (ngược lại trường hợp hiện tại)
MORPH_KERNEL_SIZE = 7         # kích thước kernel lọc nhiễu - tăng nếu ảnh nhiễu nhiều
MIN_CONTOUR_AREA_RATIO = 0.01 # bỏ qua contour quá nhỏ (< 1% diện tích ảnh) - tránh nhận nhầm nhiễu
CROP_PADDING = 15             # số pixel đệm thêm quanh bounding box khi crop

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================================================================
# LOGIC CHUNG - không cần sửa khi đổi bài toán
# ==========================================================================
def detect_product_bbox(img_bgr):
    """
    Nhận ảnh BGR (OpenCV), trả về (x, y, w, h) của bounding box sản phẩm,
    hoặc None nếu không tìm thấy contour hợp lệ nào.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    thresh_type = cv2.THRESH_BINARY_INV if INVERT_THRESHOLD else cv2.THRESH_BINARY

    if USE_OTSU:
        _, thresh = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, MANUAL_THRESHOLD, 255, thresh_type)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh, cleaned

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    valid_contours = [c for c in contours if cv2.contourArea(c) >= img_area * MIN_CONTOUR_AREA_RATIO]
    if not valid_contours:
        return None, thresh, cleaned

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h), thresh, cleaned


def crop_with_padding(img_bgr, bbox, padding):
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    return img_bgr[y1:y2, x1:x2]


def _resize_for_display(img, max_width):
    """Resize ảnh (chỉ để hiển thị) sao cho chiều rộng <= max_width."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def show_image(window_name, img):
    display_img = _resize_for_display(img, PREVIEW_DISPLAY_MAX_WIDTH)
    cv2.imshow(window_name, display_img)


def process_single_image(image_path, output_dir=None, save_debug_steps=False,
                          show_preview=False):
    """
    Xử lý 1 ảnh, trả về ảnh đã crop (numpy array) hoặc None nếu thất bại.

    show_preview=True: hiển thị từng bước lên màn hình bằng cv2.imshow
    (nhấn phím bất kỳ trên cửa sổ ảnh để đóng), KHÔNG lưu file.
    save_debug_steps=True: lưu từng bước ra output_dir (hành vi cũ).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Lỗi: không đọc được ảnh {image_path}")
        return None

    bbox, thresh, cleaned = detect_product_bbox(img)
    if bbox is None:
        print(f"  Cảnh báo: không tìm thấy sản phẩm trong {os.path.basename(image_path)}")
        if show_preview:
            show_image("1 - Threshold (khong tim thay san pham)", thresh)
            show_image("2 - Cleaned", cleaned)
            print("  Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    cropped = crop_with_padding(img, bbox, CROP_PADDING)

    bbox_img = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    if show_preview:
        show_image("1 - Threshold", thresh)
        show_image("2 - Cleaned", cleaned)
        show_image("3 - Bounding box", bbox_img)
        show_image("4 - Cropped", cropped)
        print("  Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_debug_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_1_threshold.png"), thresh)
        cv2.imwrite(os.path.join(output_dir, f"{base}_2_cleaned.png"), cleaned)
        cv2.imwrite(os.path.join(output_dir, f"{base}_3_bbox.png"), bbox_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_4_cropped.png"), cropped)

    return cropped


def run_preview():
    print(f"Đang xử lý thử ảnh: {PREVIEW_IMAGE_PATH}")

    if SHOW_PREVIEW_ONLY:
        cropped = process_single_image(PREVIEW_IMAGE_PATH, show_preview=True)
        if cropped is not None:
            print("Thành công! (Chỉ hiển thị lên màn hình, không lưu file nào)")
        else:
            print("Thất bại - xem cảnh báo phía trên để biết nguyên nhân.")
    else:
        os.makedirs(PREVIEW_OUTPUT_DIR, exist_ok=True)
        cropped = process_single_image(
            PREVIEW_IMAGE_PATH, PREVIEW_OUTPUT_DIR,
            save_debug_steps=True, show_preview=True,
        )
        if cropped is not None:
            print(f"Thành công! Ảnh minh họa cũng đã lưu tại: {PREVIEW_OUTPUT_DIR}")
        else:
            print("Thất bại - xem cảnh báo phía trên để biết nguyên nhân.")

    print("\nNếu bbox chưa đúng, thử chỉnh: USE_OTSU, MANUAL_THRESHOLD, "
          "INVERT_THRESHOLD, MORPH_KERNEL_SIZE, MIN_CONTOUR_AREA_RATIO")


def run_batch():
    if not os.path.isdir(BATCH_INPUT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {BATCH_INPUT_DIR}")

    class_names = sorted([
        d for d in os.listdir(BATCH_INPUT_DIR)
        if os.path.isdir(os.path.join(BATCH_INPUT_DIR, d))
    ])
    print(f"Các class tìm thấy: {class_names}")

    total_success, total_fail = 0, 0

    for cls in class_names:
        in_dir = os.path.join(BATCH_INPUT_DIR, cls)
        out_dir = os.path.join(BATCH_OUTPUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(in_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        print(f"\n[{cls}] Đang xử lý {len(image_files)} ảnh...")

        for fname in image_files:
            in_path = os.path.join(in_dir, fname)
            cropped = process_single_image(in_path)

            if cropped is not None:
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, cropped)
                total_success += 1
            else:
                total_fail += 1

    print(f"\n=== Hoàn tất ===")
    print(f"Thành công: {total_success} ảnh")
    print(f"Thất bại (không detect được): {total_fail} ảnh")
    print(f"Kết quả lưu tại: {BATCH_OUTPUT_DIR}")
    if total_fail > 0:
        print("Với các ảnh thất bại, kiểm tra lại tham số threshold hoặc crop tay riêng.")


if __name__ == "__main__":
    if MODE == "preview":
        run_preview()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError("MODE phải là 'preview' hoặc 'batch'")