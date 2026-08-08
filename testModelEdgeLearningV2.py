"""
Test model đã train trên một thư mục test có nhiều thư mục con theo class
(ví dụ: test/bright_blue/, test/bright_red/, ...).

MỚI: trước khi đưa ảnh vào model classifier, tự động DETECT + CROP sản phẩm
bằng threshold cổ điển (không model AI, giống detect_and_crop_edge_lightweight.py)
- giúp model chỉ nhìn thấy sản phẩm, bớt nhiễu bởi nền xung quanh. Đồng thời
vẽ bounding box + % tin cậy lên ảnh và hiển thị lần lượt từng ảnh.

In ra bảng đánh giá: mỗi thư mục con -> số ảnh, số đoán đúng, accuracy,
và nếu đoán sai thì đoán nhầm thành class nào.

LƯU Ý: script này KHÔNG dùng ImageFolder để tránh trường hợp tên class
trong thư mục test không khớp thứ tự/index với lúc train. Thay vào đó,
mỗi ảnh được dự đoán độc lập bằng model, rồi so sánh TÊN thư mục chứa nó
với TÊN class model dự đoán ra.

Cách chạy:
    python test_model_report.py
"""

import os
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ----------------------------
# 1. Cấu hình - SỬA Ở ĐÂY
# ----------------------------
MODEL_PATH = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\Goi hut am\model\edge_classifier_fewshot_mobilenet_v3_large.pt"
TEST_DIR = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\Goi hut am\test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Xuất thêm file CSV chi tiết từng ảnh (mở bằng Excel dễ xem hơn khi có nhiều class)
# Để trống "" nếu không cần xuất CSV
EXPORT_CSV_PATH = r"D:\TongHop\RTC Technologi\RD\detect_color\model\test_report.csv"

# --- Detect + crop trước khi đưa vào model (threshold cổ điển, không model AI) ---
USE_DETECT_CROP = True   # False để tắt, dùng nguyên ảnh gốc như code cũ
METHOD = "threshold"     # "threshold" | "background_subtraction"
BACKGROUND_REF_PATH = ""  # chỉ dùng khi METHOD="background_subtraction"
DIFF_THRESHOLD = 30
INVERT_THRESHOLD = "auto"   # "auto" | True | False
MORPH_KERNEL_SIZE = 7
MIN_CONTOUR_AREA_RATIO = 0.01
MAX_CONTOUR_AREA_RATIO = 0.98
BORDER_TOUCH_MAX_RATIO = 0.3
CROP_PADDING = 15

# --- Hiển thị từng ảnh kèm bbox + % tin cậy ---
SHOW_PREVIEW = True
PREVIEW_DISPLAY_MAX_WIDTH = 800

# Transform test - PHẢI giống hệt val_transform lúc train (không augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

_bg_gray = None
_preview_on = SHOW_PREVIEW   # có thể tắt giữa chừng bằng phím ESC


# ----------------------------
# 2. Detect + crop (threshold cổ điển)
# ----------------------------
def _get_bg_gray():
    global _bg_gray
    if _bg_gray is None and BACKGROUND_REF_PATH and os.path.isfile(BACKGROUND_REF_PATH):
        bg = cv2.imread(BACKGROUND_REF_PATH)
        if bg is not None:
            _bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return _bg_gray


def _clean(mask):
    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def _border_touch_ratio(contour, shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    border = np.zeros((H, W), bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True
    return np.count_nonzero((m > 0) & border) / border.sum()


def _best_contour(mask, img_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if MIN_CONTOUR_AREA_RATIO * img_area <= cv2.contourArea(c) <= MAX_CONTOUR_AREA_RATIO * img_area]
    if not valid:
        return None
    shape = mask.shape[:2]
    clean = [c for c in valid if _border_touch_ratio(c, shape) <= BORDER_TOUCH_MAX_RATIO]
    if not clean:
        clean = sorted(valid, key=lambda c: _border_touch_ratio(c, shape))[:1]
    return max(clean, key=cv2.contourArea)


def detect_bbox(img_bgr):
    """Trả về (x, y, w, h) hoặc None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    bg_gray = _get_bg_gray() if METHOD == "background_subtraction" else None

    if bg_gray is not None:
        if bg_gray.shape != gray.shape:
            bg_gray = cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]))
        diff = cv2.absdiff(gray, bg_gray)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        contour = _best_contour(_clean(mask), img_area)
    else:
        directions = [False, True] if INVERT_THRESHOLD == "auto" else [bool(INVERT_THRESHOLD)]
        candidates = []
        for inv in directions:
            t = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray, 0, 255, t + cv2.THRESH_OTSU)
            c = _best_contour(_clean(mask), img_area)
            if c is not None:
                candidates.append(c)
        contour = max(candidates, key=cv2.contourArea) if candidates else None

    return cv2.boundingRect(contour) if contour is not None else None


def crop_img(img_bgr, bbox, padding=CROP_PADDING):
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    return img_bgr[y1:y2, x1:x2]


# ----------------------------
# 3. Load model từ checkpoint
# ----------------------------
def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        embedding_dim = model.last_channel
        model.classifier[1] = nn.Linear(embedding_dim, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        embedding_dim = 960
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        embedding_dim = 576
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    backbone_name = checkpoint.get("backbone_name", "mobilenet_v2")

    model = build_model_architecture(backbone_name, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Đã load model từ: {model_path}")
    print(f"Backbone: {backbone_name}")
    print(f"Các class model biết: {class_names}\n")
    return model, class_names


# ----------------------------
# 4. Dự đoán 1 ảnh (có detect+crop trước, vẽ bbox, hiển thị)
# ----------------------------
def _resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def predict_image(model, class_names, image_path, folder_name):
    """
    Trả về (pred_class, confidence, probs_dict, detected).
    Nếu SHOW_PREVIEW=True, hiện ảnh gốc kèm bbox (nếu detect được) + nhãn dự đoán + % tin cậy.
    """
    global _preview_on

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path}")

    bbox = None
    crop_bgr = img_bgr
    detected = False
    if USE_DETECT_CROP:
        bbox = detect_bbox(img_bgr)
        if bbox is not None:
            crop_bgr = crop_img(img_bgr, bbox)
            detected = True
        # nếu không detect được, dùng nguyên ảnh gốc để vẫn có kết quả dự đoán

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    input_tensor = test_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)

    pred_class = class_names[pred_idx.item()]
    probs_dict = {name: p for name, p in zip(class_names, probs.tolist())}
    is_correct = (pred_class == folder_name)

    if _preview_on:
        vis = img_bgr.copy()
        color = (0, 200, 0) if is_correct else (0, 0, 255)   # xanh=đúng, đỏ=sai (BGR)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 4)
        else:
            cv2.putText(vis, "KHONG DETECT DUOC - dung anh goc", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        label = f"{pred_class} ({confidence.item()*100:.1f}%)  [that: {folder_name}]"
        cv2.putText(vis, label, (20, vis.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        disp = _resize_for_display(vis, PREVIEW_DISPLAY_MAX_WIDTH)
        win = "Test model - nhan phim bat ky de qua anh tiep theo (ESC = tat preview)"
        cv2.imshow(win, disp)
        key = cv2.waitKey(0)
        cv2.destroyWindow(win)
        if key == 27:
            _preview_on = False
            print("  (Đã tắt preview, tiếp tục chạy ngầm không hiển thị...)")

    return pred_class, confidence.item(), probs_dict, detected


def export_csv(results_per_folder, class_names, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["Thu_muc_that", "File", "Du_doan", "Dung_Sai", "Detect_duoc"] + [f"Xac_suat_{c}" for c in class_names]
        writer.writerow(header)

        for r in results_per_folder:
            for img in r["per_image"]:
                row = [
                    r["folder"], img["file"], img["pred"],
                    "Dung" if img["correct"] else "Sai",
                    "Co" if img["detected"] else "Khong",
                ]
                row += [f"{img['probs'].get(c, 0.0):.4f}" for c in class_names]
                writer.writerow(row)

    print(f"\nĐã xuất bảng chi tiết ra file CSV: {csv_path}")


# ----------------------------
# 5. Duyệt qua từng thư mục con, đánh giá
# ----------------------------
def evaluate_test_dir(model, class_names, test_dir):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục test tại: {test_dir}")

    subfolders = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    if not subfolders:
        print(f"Thư mục {test_dir} không có thư mục con nào.")
        return

    results_per_folder = []
    total_all, correct_all = 0, 0
    total_detected = 0

    for folder_name in subfolders:
        folder_path = os.path.join(test_dir, folder_name)
        image_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        if not image_files:
            print(f"[Bỏ qua] Thư mục '{folder_name}' không có ảnh.")
            continue

        total = 0
        correct = 0
        confidences = []
        wrong_predictions = {}
        per_image_results = []

        for fname in image_files:
            image_path = os.path.join(folder_path, fname)
            try:
                pred_class, confidence, probs_dict, detected = predict_image(
                    model, class_names, image_path, folder_name)
            except Exception as e:
                print(f"  Lỗi đọc ảnh '{fname}': {e}")
                continue

            total += 1
            if detected:
                total_detected += 1
            confidences.append(confidence)
            is_correct = (pred_class == folder_name)

            if is_correct:
                correct += 1
            else:
                wrong_predictions[pred_class] = wrong_predictions.get(pred_class, 0) + 1

            per_image_results.append({
                "file": fname,
                "pred": pred_class,
                "correct": is_correct,
                "detected": detected,
                "probs": probs_dict,
            })

        acc = correct / total if total > 0 else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        results_per_folder.append({
            "folder": folder_name,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "avg_confidence": avg_conf,
            "wrong_predictions": wrong_predictions,
            "per_image": per_image_results,
        })

        total_all += total
        correct_all += correct

    cv2.destroyAllWindows()

    # ----------------------------
    # In bảng kết quả
    # ----------------------------
    print("=" * 78)
    print(f"{'Thư mục (class thật)':<25}{'Số ảnh':>8}{'Đúng':>8}{'Accuracy':>12}{'Conf. TB':>12}")
    print("-" * 78)
    for r in results_per_folder:
        print(f"{r['folder']:<25}{r['total']:>8}{r['correct']:>8}"
              f"{r['accuracy']:>11.2%} {r['avg_confidence']:>11.2%}")
    print("-" * 78)

    overall_acc = correct_all / total_all if total_all > 0 else 0.0
    print(f"{'TỔNG CỘNG':<25}{total_all:>8}{correct_all:>8}{overall_acc:>11.2%}")
    if USE_DETECT_CROP:
        print(f"Detect được sản phẩm: {total_detected}/{total_all} ảnh "
              f"({total_detected/total_all:.1%})" if total_all else "")
    print("=" * 78)

    # ----------------------------
    # In chi tiết các trường hợp đoán sai
    # ----------------------------
    print("\nChi tiết nhầm lẫn (folder thật -> đoán nhầm thành đâu, số lần):")
    any_wrong = False
    for r in results_per_folder:
        if r["wrong_predictions"]:
            any_wrong = True
            wrong_str = ", ".join(
                f"{cls}: {count}" for cls, count in
                sorted(r["wrong_predictions"].items(), key=lambda x: -x[1])
            )
            print(f"  {r['folder']:<20} -> {wrong_str}")
    if not any_wrong:
        print("  Không có trường hợp nào đoán sai.")

    # ----------------------------
    # In bảng chi tiết TỪNG ẢNH
    # ----------------------------
    print("\n" + "=" * 100)
    print("BẢNG CHI TIẾT TỪNG ẢNH - % xác suất từng class")
    print("=" * 100)

    col_width = max(10, max(len(c) for c in class_names) + 2)

    for r in results_per_folder:
        print(f"\n--- Thư mục: {r['folder']} ---")
        header = f"{'File':<30}{'Đúng/Sai':<10}"
        for c in class_names:
            header += f"{c:>{col_width}}"
        print(header)
        print("-" * len(header))

        for img in r["per_image"]:
            status = "Đúng" if img["correct"] else "SAI"
            row = f"{img['file']:<30}{status:<10}"
            for c in class_names:
                p = img["probs"].get(c, 0.0)
                row += f"{p:>{col_width-1}.1%} "
            print(row)

    if EXPORT_CSV_PATH:
        export_csv(results_per_folder, class_names, EXPORT_CSV_PATH)


# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    model, class_names = load_model(MODEL_PATH)
    evaluate_test_dir(model, class_names, TEST_DIR)