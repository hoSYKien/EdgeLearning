"""
Test model đã train trên một thư mục test có nhiều thư mục con theo class
(ví dụ: test/bright_blue/, test/bright_red/, ...).

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

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ----------------------------
# 1. Cấu hình - SỬA Ở ĐÂY
# ----------------------------
MODEL_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\model\model2\edge_classifier_fewshot_mobilenet_v3_large_multicrop.pt"
TEST_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Xuất thêm file CSV chi tiết từng ảnh (mở bằng Excel dễ xem hơn khi có nhiều class)
# Để trống "" nếu không cần xuất CSV
EXPORT_CSV_PATH = r"D:\TongHop\RTC Technologi\RD\detect_color\model\test_report.csv"

# Transform test - PHẢI giống hệt val_transform lúc train (không augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ----------------------------
# 2. Load model từ checkpoint
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
    # Checkpoint cũ không có "backbone_name" -> mặc định mobilenet_v2
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
# 3. Dự đoán 1 ảnh
# ----------------------------
def predict_image(model, class_names, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)

    pred_class = class_names[pred_idx.item()]
    # probs_dict: xác suất của TẤT CẢ class, dùng để in bảng chi tiết
    probs_dict = {name: p for name, p in zip(class_names, probs.tolist())}
    return pred_class, confidence.item(), probs_dict


def export_csv(results_per_folder, class_names, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["Thu_muc_that", "File", "Du_doan", "Dung_Sai"] + [f"Xac_suat_{c}" for c in class_names]
        writer.writerow(header)

        for r in results_per_folder:
            for img in r["per_image"]:
                row = [
                    r["folder"], img["file"], img["pred"],
                    "Dung" if img["correct"] else "Sai",
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

    results_per_folder = []  # lưu (tên_folder, total, correct, avg_confidence, sai_nhầm_thành)
    total_all, correct_all = 0, 0

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
        wrong_predictions = {}  # class dự đoán sai -> số lần
        per_image_results = []  # lưu chi tiết từng ảnh để in bảng

        for fname in image_files:
            image_path = os.path.join(folder_path, fname)
            try:
                pred_class, confidence, probs_dict = predict_image(model, class_names, image_path)
            except Exception as e:
                print(f"  Lỗi đọc ảnh '{fname}': {e}")
                continue

            total += 1
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
    print("=" * 78)

    # ----------------------------
    # In chi tiết các trường hợp đoán sai (nhầm thành class nào)
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
    # In bảng chi tiết TỪNG ẢNH: % xác suất của tất cả các class
    # ----------------------------
    print("\n" + "=" * 100)
    print("BẢNG CHI TIẾT TỪNG ẢNH - % xác suất từng class")
    print("=" * 100)

    # Độ rộng cột cho mỗi class, dựa theo tên class dài nhất
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

    # ----------------------------
    # Xuất CSV nếu có khai báo đường dẫn
    # ----------------------------
    if EXPORT_CSV_PATH:
        export_csv(results_per_folder, class_names, EXPORT_CSV_PATH)


# ----------------------------
# 5. Main
# ----------------------------
if __name__ == "__main__":
    model, class_names = load_model(MODEL_PATH)
    evaluate_test_dir(model, class_names, TEST_DIR)