import os

import cv2
import numpy as np

# ==========================================================================
# CONFIG
# ==========================================================================

MODE = "preview"   # "preview" | "batch"
METHOD = "background_subtraction"   # "background_subtraction" | "threshold"

BACKGROUND_REF_PATH = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\HutAm\Goi hut am\Type2_blue_white\Image_0031_20260330080913.bmp"
DIFF_THRESHOLD = 30   # ngưỡng chênh lệch pixel để coi là "khác nền" (0-255, tăng nếu bắt nhiễu quá nhiều)

INVERT_THRESHOLD = "auto"   # "auto" | True | False

# --- Tham số chung (áp dụng cho cả 2 phương pháp) ---
MORPH_KERNEL_SIZE = 7          # kích thước kernel lọc nhiễu - tăng nếu ảnh nhiễu nhiều
MIN_CONTOUR_AREA_RATIO = 0.01  # bỏ contour quá nhỏ (< 1% diện tích ảnh) - tránh nhận nhầm nhiễu
CROP_PADDING = 15              # số pixel đệm thêm quanh bounding box khi crop

# --- Dùng cho MODE = "preview" ---
# Có thể là ĐƯỜNG DẪN 1 FILE ẢNH, hoặc ĐƯỜNG DẪN 1 THƯ MỤC chứa nhiều ảnh.
PREVIEW_IMAGE_PATH = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\HutAm\Goi hut am\Type2_blue_white"
PREVIEW_DISPLAY_MAX_WIDTH = 700

# --- Dùng cho MODE = "batch" ---
BATCH_INPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset\train"
BATCH_OUTPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset_cropped\train"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_background_gray = None   # cache ảnh nền trống (grayscale), load 1 lần


# ==========================================================================
# LOGIC DETECT
# ==========================================================================
def get_background_gray():
    """Load + cache ảnh nền trống (grayscale). Trả về None nếu không có/không đọc được."""
    global _background_gray
    if _background_gray is not None:
        return _background_gray
    if not BACKGROUND_REF_PATH or not os.path.isfile(BACKGROUND_REF_PATH):
        return None
    bg = cv2.imread(BACKGROUND_REF_PATH)
    if bg is None:
        return None
    _background_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return _background_gray


def find_largest_valid_contour(binary_mask, img_area):
    """Tìm contour lớn nhất đạt diện tích tối thiểu trong ảnh nhị phân. Trả về contour hoặc None."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [c for c in contours if cv2.contourArea(c) >= img_area * MIN_CONTOUR_AREA_RATIO]
    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def detect_bbox_background_subtraction(img_bgr):
    """
    So sánh ảnh với nền trống đã lưu, tìm vùng khác biệt lớn nhất -> bbox.
    Trả về (bbox, debug_dict) hoặc (None, debug_dict).
    """
    bg_gray = get_background_gray()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if bg_gray is None:
        return None, {"error": "Không có/không đọc được ảnh nền trống"}

    if bg_gray.shape != gray.shape:
        bg_gray = cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]))

    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    contour = find_largest_valid_contour(cleaned, img_area)
    debug = {"diff": diff, "thresh": thresh, "cleaned": cleaned}
    if contour is None:
        return None, debug

    bbox = cv2.boundingRect(contour)   # (x, y, w, h)
    return bbox, debug


def _threshold_and_find(gray, invert, img_area):
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contour = find_largest_valid_contour(cleaned, img_area)
    return contour, thresh, cleaned


def detect_bbox_threshold(img_bgr):
    """
    Threshold cổ điển (Otsu) + contour. Nếu INVERT_THRESHOLD='auto', tự thử
    cả 2 hướng và chọn hướng cho contour lớn hơn (giả định sản phẩm là vật
    thể chính, chiếm diện tích đáng kể trong khung).
    Trả về (bbox, debug_dict) hoặc (None, debug_dict).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]

    if INVERT_THRESHOLD == "auto":
        contour_a, thresh_a, cleaned_a = _threshold_and_find(gray, False, img_area)
        contour_b, thresh_b, cleaned_b = _threshold_and_find(gray, True, img_area)

        area_a = cv2.contourArea(contour_a) if contour_a is not None else 0
        area_b = cv2.contourArea(contour_b) if contour_b is not None else 0

        if area_a == 0 and area_b == 0:
            return None, {"thresh": thresh_a, "cleaned": cleaned_a}

        if area_a >= area_b:
            contour, thresh, cleaned = contour_a, thresh_a, cleaned_a
        else:
            contour, thresh, cleaned = contour_b, thresh_b, cleaned_b
    else:
        contour, thresh, cleaned = _threshold_and_find(gray, bool(INVERT_THRESHOLD), img_area)

    debug = {"thresh": thresh, "cleaned": cleaned}
    if contour is None:
        return None, debug

    bbox = cv2.boundingRect(contour)
    return bbox, debug


def detect_bbox(img_bgr):
    """Điều phối theo METHOD - luôn tìm toạ độ qua ảnh grayscale."""
    if METHOD == "background_subtraction" and get_background_gray() is not None:
        return detect_bbox_background_subtraction(img_bgr)
    if METHOD == "background_subtraction":
        print("  CẢNH BÁO: METHOD='background_subtraction' nhưng không có ảnh nền trống hợp lệ "
              f"tại BACKGROUND_REF_PATH='{BACKGROUND_REF_PATH}' - lùi về 'threshold'.")
    return detect_bbox_threshold(img_bgr)


def crop_with_padding(img_bgr, bbox, padding):
    """CROP LUÔN THỰC HIỆN TRÊN ẢNH MÀU GỐC (RGB/BGR), không phải ảnh grayscale dùng để detect."""
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    return img_bgr[y1:y2, x1:x2]


# ==========================================================================
# PREVIEW / HIỂN THỊ
# ==========================================================================
def _resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def show_image(window_name, img):
    cv2.imshow(window_name, _resize_for_display(img, PREVIEW_DISPLAY_MAX_WIDTH))


_LAST_KEY = None


def process_single_image(image_path, output_dir=None, save_debug_steps=False, show_preview=False):
    """Xử lý 1 ảnh, trả về ảnh đã crop (trên ảnh màu gốc) hoặc None nếu thất bại."""
    global _LAST_KEY
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Lỗi: không đọc được ảnh {image_path}")
        return None

    bbox, debug = detect_bbox(img)
    if bbox is None:
        print(f"  Cảnh báo: không tìm thấy sản phẩm trong {os.path.basename(image_path)}")
        if show_preview:
            for name, dimg in debug.items():
                if dimg is not None:
                    show_image(f"debug - {name}", dimg)
            print("  Nhấn phím bất kỳ để qua ảnh tiếp theo (ESC để dừng)...")
            _LAST_KEY = cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    cropped = crop_with_padding(img, bbox, CROP_PADDING)   # crop trên ảnh MÀU gốc

    bbox_img = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    if show_preview:
        for name, dimg in debug.items():
            if dimg is not None:
                show_image(f"1 - debug - {name}", dimg)
        show_image("2 - Bounding box (tren anh mau)", bbox_img)
        show_image("3 - Cropped (tren anh mau goc)", cropped)
        print("  Nhấn phím bất kỳ trên cửa sổ ảnh để qua ảnh tiếp theo (ESC để dừng)...")
        _LAST_KEY = cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_debug_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        for name, dimg in debug.items():
            if dimg is not None:
                cv2.imwrite(os.path.join(output_dir, f"{base}_debug_{name}.png"), dimg)
        cv2.imwrite(os.path.join(output_dir, f"{base}_bbox.png"), bbox_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_cropped.png"), cropped)

    return cropped


def run_preview():
    path = PREVIEW_IMAGE_PATH

    if os.path.isdir(path):
        image_files = sorted([f for f in os.listdir(path) if f.lower().endswith(IMG_EXTENSIONS)])
        if not image_files:
            print(f"Không tìm thấy ảnh nào trong thư mục: {path}")
            return

        print(f"Tìm thấy {len(image_files)} ảnh trong thư mục: {path}")
        print(f"METHOD = '{METHOD}'"
              + (f" (nền trống: {BACKGROUND_REF_PATH})" if METHOD == "background_subtraction" else ""))
        print("Duyệt lần lượt từng ảnh - nhấn phím bất kỳ để qua ảnh tiếp theo, ESC để dừng.\n")

        for idx, fname in enumerate(image_files, start=1):
            fpath = os.path.join(path, fname)
            print(f"[{idx}/{len(image_files)}] {fname}")
            cropped = process_single_image(fpath, show_preview=True)
            print("  -> Thành công." if cropped is not None else "  -> Thất bại.")
            if _LAST_KEY == 27:
                print("\nĐã nhấn ESC - dừng duyệt.")
                break

        print("\nHoàn tất duyệt thư mục preview. (Chỉ hiển thị, không lưu file nào)")

    elif os.path.isfile(path):
        print(f"Đang xử lý thử ảnh: {path}")
        cropped = process_single_image(path, show_preview=True)
        print("Thành công! (Chỉ hiển thị lên màn hình, không lưu file nào)" if cropped is not None
              else "Thất bại - thử chỉnh DIFF_THRESHOLD / MIN_CONTOUR_AREA_RATIO / MORPH_KERNEL_SIZE.")
    else:
        print(f"Không tìm thấy file hoặc thư mục: {path}")


def run_batch():
    if not os.path.isdir(BATCH_INPUT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {BATCH_INPUT_DIR}")

    class_names = sorted([d for d in os.listdir(BATCH_INPUT_DIR) if os.path.isdir(os.path.join(BATCH_INPUT_DIR, d))])
    print(f"Các class tìm thấy: {class_names}")

    total_success, total_fail = 0, 0
    for cls in class_names:
        in_dir = os.path.join(BATCH_INPUT_DIR, cls)
        out_dir = os.path.join(BATCH_OUTPUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(in_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        print(f"\n[{cls}] Đang xử lý {len(image_files)} ảnh...")

        for fname in image_files:
            cropped = process_single_image(os.path.join(in_dir, fname))
            if cropped is not None:
                cv2.imwrite(os.path.join(out_dir, fname), cropped)
                total_success += 1
            else:
                total_fail += 1

    print(f"\n=== Hoàn tất ===\nThành công: {total_success} ảnh\nThất bại: {total_fail} ảnh\nKết quả lưu tại: {BATCH_OUTPUT_DIR}")


if __name__ == "__main__":
    if MODE == "preview":
        run_preview()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError("MODE phải là 'preview' hoặc 'batch'")