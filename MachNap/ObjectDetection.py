"""
detect_any_object_static_bg.py
==============================

Tìm mọi vật thể xuất hiện trên background tĩnh / camera cố định.

Ý tưởng:
    1. Học background từ các ảnh KHÔNG có vật.
    2. So sánh ảnh mới với background bằng:
       - độ lệch từng kênh BGR
       - độ lệch màu Lab
       - loại vùng chỉ tối đi nhưng giữ nguyên màu (bóng)
    3. Hysteresis: vùng lệch nhẹ chỉ được giữ nếu nối với vùng lệch mạnh.
    4. Morphology + fill hole + lọc blob nhỏ.
    5. Xuất mask, bbox, ảnh preview và JSON.

Không cần biết trước loại vật hoặc màu vật.
Không cần YOLO để tìm vật mới.

Cài thư viện:
    pip install opencv-python numpy

Cách dùng:
    1. Đặt ảnh nền trống vào BACKGROUND_DIR.
    2. Đặt ảnh cần detect vào INPUT_DIR.
    3. Chạy:
       python detect_any_object_static_bg.py

Lưu ý:
    - Nếu ảnh camera rất lớn, SCALE = 0.5 để chạy nhẹ hơn.
    - Khi đổi ánh sáng/camera/vị trí, hãy học lại background.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# CẤU HÌNH
# ============================================================================

# Ảnh nền trống: không có sản phẩm/vật thể cần bắt.
BACKGROUND_DIR = r"D:\TongHop\RTC Technologi\PCB\BackGround"

# Ảnh cần detect. Có thể đổi thành thư mục MVS\Data để thử.
INPUT_DIR = r"C:\Users\LAP273\MVS\Data"

OUTPUT_DIR = r"D:\TongHop\RTC Technologi\PCB\Detect"

# 0.5 nghĩa là xử lý ở 1/2 kích thước để nhẹ hơn.
# Bbox xuất ra vẫn được quy về kích thước ảnh gốc.
SCALE = 0.5

# Giới hạn số ảnh nền dùng để học. Ảnh nền càng nhiều càng ổn định.
MAX_BACKGROUND_IMAGES = 30

# Ngưỡng khác nền:
# Tăng các giá trị này nếu bị bắt nhiễu.
# Giảm nếu vật nhỏ/màu gần nền bị bỏ sót.
BGR_STRONG = 38
BGR_WEAK = 20
LAB_STRONG = 28
LAB_WEAK = 15

# Loại bóng: pixel tối đi nhưng gần như không đổi sắc độ.
# Nếu vật tối bị mất, giảm SHADOW_MIN_RATIO hoặc đặt False.
REMOVE_SHADOW = True
SHADOW_MIN_RATIO = 0.45
SHADOW_MAX_RATIO = 0.98
SHADOW_CHROMA_TOL = 0.055

# Lọc blob.
MIN_AREA_PX = 800          # diện tích theo ảnh đã scale
MAX_AREA_RATIO = 0.92      # blob quá to thường là đổi ánh sáng/camera
PADDING_PX = 12            # padding bbox, theo ảnh gốc

# Morphology.
OPEN_KERNEL = 3
CLOSE_KERNEL = 9

# ROI tùy chọn.
# [] = chạy toàn bộ khung hình.
# Ví dụ: ROI_POLYGON = [(100, 50), (1800, 50), (1800, 1000), (100, 1000)]
ROI_POLYGON: list[tuple[int, int]] = []

VALID_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


# ============================================================================
# IO / BACKGROUND
# ============================================================================

def list_images(folder: str) -> list[Path]:
    path = Path(folder)
    if not path.is_dir():
        return []

    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    return image


def resize_for_process(image: np.ndarray) -> np.ndarray:
    if SCALE == 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, round(w * SCALE)), max(1, round(h * SCALE))),
        interpolation=cv2.INTER_AREA,
    )


def build_roi_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.full((h, w), 255, dtype=np.uint8)

    if not ROI_POLYGON:
        return mask

    points = np.array(ROI_POLYGON, dtype=np.float32)
    points *= SCALE
    points = np.round(points).astype(np.int32)

    mask[:] = 0
    cv2.fillPoly(mask, [points], 255)
    return mask


def learn_background(background_paths: list[Path]) -> np.ndarray:
    """Tạo background trung bình từ ảnh nền trống."""
    if not background_paths:
        raise RuntimeError(
            "Khong co anh nen. Hay tao BACKGROUND_DIR va dat anh khong co vat vao do."
        )

    selected = background_paths[:MAX_BACKGROUND_IMAGES]
    accumulator = None
    expected_shape = None

    print(f"Hoc background tu {len(selected)} anh...")

    for index, path in enumerate(selected, start=1):
        image = resize_for_process(read_bgr(path))

        if expected_shape is None:
            expected_shape = image.shape
            accumulator = np.zeros(image.shape, dtype=np.float32)
        elif image.shape != expected_shape:
            raise ValueError(
                f"Kich thuoc anh nen khong dong nhat: {path.name} "
                f"{image.shape} != {expected_shape}"
            )

        cv2.accumulate(image.astype(np.float32), accumulator)
        print(f"  [{index}/{len(selected)}] {path.name}")

    return (accumulator / len(selected)).astype(np.uint8)


# ============================================================================
# MASK
# ============================================================================

def hysteresis_mask(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """
    Chỉ giữ component weak có chứa ít nhất một pixel strong.
    Giúp bắt biên vật nhẹ nhưng bỏ loang/nhiễu nhỏ.
    """
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)

    if count <= 1:
        return np.zeros_like(weak, dtype=np.uint8)

    strong_labels = np.unique(labels[strong > 0])
    strong_labels = strong_labels[strong_labels != 0]

    keep = np.zeros(count, dtype=np.uint8)
    keep[strong_labels] = 1

    return (keep[labels] * 255).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Lấp các lỗ kín bên trong object."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)

    return cv2.bitwise_or(mask, holes)


def shadow_mask(current_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """
    True tại pixel giống màu background nhưng tối đi:
    thường là bóng, không phải object.
    """
    current_f = current_bgr.astype(np.float32)
    background_f = background_bgr.astype(np.float32)

    current_sum = current_f.sum(axis=2) + 3.0
    background_sum = background_f.sum(axis=2) + 3.0
    brightness_ratio = current_sum / background_sum

    # So sánh chroma chuẩn hóa, độc lập tương đối với cường độ sáng.
    current_chroma = (current_f + 1.0) / current_sum[..., None]
    background_chroma = (background_f + 1.0) / background_sum[..., None]
    chroma_difference = np.max(
        np.abs(current_chroma - background_chroma),
        axis=2,
    )

    is_dim = (
        (brightness_ratio >= SHADOW_MIN_RATIO)
        & (brightness_ratio <= SHADOW_MAX_RATIO)
    )
    same_chroma = chroma_difference <= SHADOW_CHROMA_TOL

    return is_dim & same_chroma


def make_foreground_mask(
    current_bgr: np.ndarray,
    background_bgr: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """
    Tạo foreground mask tổng quát, không dựa vào màu vật thể.
    """
    current = cv2.GaussianBlur(current_bgr, (5, 5), 0)
    background = cv2.GaussianBlur(background_bgr, (5, 5), 0)

    # 1) Sai khác BGR từng kênh.
    bgr_difference = cv2.absdiff(current, background)
    bgr_score = np.max(bgr_difference, axis=2)

    # 2) Sai khác trong Lab: nhạy hơn với đổi sắc độ/màu.
    current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    lab_difference = cv2.absdiff(current_lab, background_lab)
    lab_score = np.max(lab_difference, axis=2)

    strong = (
        (bgr_score >= BGR_STRONG)
        | (lab_score >= LAB_STRONG)
    ).astype(np.uint8)

    weak = (
        (bgr_score >= BGR_WEAK)
        | (lab_score >= LAB_WEAK)
    ).astype(np.uint8)

    # Bỏ vùng ngoài ROI.
    strong[roi_mask == 0] = 0
    weak[roi_mask == 0] = 0

    # 3) Loại bóng.
    if REMOVE_SHADOW:
        shadows = shadow_mask(current, background)
        strong[shadows] = 0
        weak[shadows] = 0

    # 4) Hysteresis.
    mask = hysteresis_mask(strong, weak)

    # 5) Làm sạch mask.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OPEN_KERNEL, OPEN_KERNEL),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSE_KERNEL, CLOSE_KERNEL),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = fill_holes(mask)

    return mask


# ============================================================================
# BBOX
# ============================================================================

def extract_objects(mask: np.ndarray) -> list[dict]:
    """
    Lấy bbox của tất cả blob đủ lớn.
    Toạ độ ở kích thước ảnh đã scale.
    """
    h, w = mask.shape[:2]
    max_area = h * w * MAX_AREA_RATIO
    objects: list[dict] = []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    for label in range(1, count):
        x, y, bw, bh, area = stats[label]

        if area < MIN_AREA_PX or area > max_area:
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)

        objects.append(
            {
                "bbox_processed": [int(x), int(y), int(x + bw), int(y + bh)],
                "area_processed": int(area),
            }
        )

    return objects


def bbox_to_original(
    bbox_processed: list[int],
    original_shape: tuple[int, int],
) -> list[int]:
    x1, y1, x2, y2 = bbox_processed
    original_h, original_w = original_shape

    x1 = max(0, round(x1 / SCALE) - PADDING_PX)
    y1 = max(0, round(y1 / SCALE) - PADDING_PX)
    x2 = min(original_w, round(x2 / SCALE) + PADDING_PX)
    y2 = min(original_h, round(y2 / SCALE) + PADDING_PX)

    return [int(x1), int(y1), int(x2), int(y2)]


# ============================================================================
# CHẠY BATCH
# ============================================================================

def detect_one(
    image_path: Path,
    background: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    original = read_bgr(image_path)
    processed = resize_for_process(original)

    if processed.shape != background.shape:
        raise ValueError(
            f"Anh {image_path.name} khac kich thuoc background: "
            f"{processed.shape} != {background.shape}"
        )

    mask = make_foreground_mask(processed, background, roi_mask)
    objects = extract_objects(mask)

    for obj in objects:
        obj["bbox"] = bbox_to_original(
            obj["bbox_processed"],
            original.shape[:2],
        )

    return original, mask, objects


def draw_preview(
    original: np.ndarray,
    objects: list[dict],
) -> np.ndarray:
    preview = original.copy()

    for index, obj in enumerate(objects, start=1):
        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            preview,
            f"object {index}",
            (x1, max(28, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        preview,
        f"Objects: {len(objects)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return preview


def main() -> None:
    background_paths = list_images(BACKGROUND_DIR)
    input_paths = list_images(INPUT_DIR)

    if not input_paths:
        raise RuntimeError(f"Khong co anh can detect trong: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    background = learn_background(background_paths)
    roi_mask = build_roi_mask(background.shape[:2])

    # Lưu background để kiểm tra.
    cv2.imwrite(
        str(Path(OUTPUT_DIR) / "background_learned.png"),
        background,
    )

    all_results = {}

    for index, image_path in enumerate(input_paths, start=1):
        try:
            original, mask, objects = detect_one(
                image_path,
                background,
                roi_mask,
            )

            preview = draw_preview(original, objects)
            stem = image_path.stem

            cv2.imwrite(
                str(Path(OUTPUT_DIR) / f"{stem}_mask.png"),
                mask,
            )
            cv2.imwrite(
                str(Path(OUTPUT_DIR) / f"{stem}_detect.png"),
                preview,
            )

            all_results[image_path.name] = {
                "objects": objects,
            }

            print(
                f"[{index}/{len(input_paths)}] {image_path.name}: "
                f"{len(objects)} object(s)"
            )

        except Exception as error:
            print(f"[{index}/{len(input_paths)}] LOI {image_path.name}: {error}")
            all_results[image_path.name] = {"error": str(error)}

    json_path = Path(OUTPUT_DIR) / "detections.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print(f"\nHoan tat. Ket qua: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()