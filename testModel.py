"""
DANH GIA MODEL PHAN LOAI OK/NG (kieu Linear+Softmax, tu few_shot_pipeline.py)
TREN TAP TEST CO NHAN.

Khac voi ban one-class (evaluate_on_test.py): file nay load model DA CO
SAN classifier head (Linear), chi can forward qua model roi argmax, KHONG
tinh khoang cach/threshold gi ca.

Cau truc thu muc TEST_DIR: 1 thu muc con cho MOI class (giong luc train),
vd OK/ va NG/ (hoac OK/, A/, B/... neu ban giu nguyen ma loi rieng).

Cach dung:
    python evaluate_classifier_on_test.py
"""

import os
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ==========================================================================
# CONFIG - SUA O DAY
# ==========================================================================

# File .pt da luu tu few_shot_pipeline.py (chua model_state + class_names)
MODEL_PATH = r"D:\TongHop\RTC Technologi\9\model\model3\edge_classifier_fewshot_mobilenet_v3_large.pt"

# Thu muc TEST - co cac thu muc con la TEN CLASS giong luc train (vd OK/, NG/)
TEST_DIR = r"D:\TongHop\RTC Technologi\9\test"

OUTPUT_CSV = r"D:\TongHop\RTC Technologi\9\test\danh_gia_chi_tiet.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# Load model (kien truc phai khop chinh xac voi luc train)
# ==========================================================================

def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        embedding_dim = model.last_channel
        model.classifier[1] = nn.Linear(embedding_dim, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(960, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(576, num_classes))
    else:
        raise ValueError(f"Backbone khong duoc ho tro: {backbone_name}")
    return model


def load_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    backbone_name = ckpt.get("backbone_name", "mobilenet_v2")

    model = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Da load model: {model_path}")
    print(f"Backbone: {backbone_name} | Class: {class_names}")
    if "kfold_val_acc_mean" in ckpt:
        print(f"K-Fold accuracy luc train: {ckpt['kfold_val_acc_mean']:.2%} "
              f"± {ckpt.get('kfold_val_acc_std', 0):.2%}")
    return model, class_names, backbone_name


def build_clean_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def list_images_with_class(test_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths, class_raw = [], []
    for entry in os.scandir(test_dir):
        if not entry.is_dir():
            continue
        cls = entry.name
        for fname in os.listdir(entry.path):
            if fname.lower().endswith(exts):
                image_paths.append(os.path.join(entry.path, fname))
                class_raw.append(cls)
    return image_paths, class_raw


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print(f"Dang chay tren: {DEVICE}")

    model, class_names, backbone_name = load_model(MODEL_PATH)
    transform = build_clean_transform()

    image_paths, class_raw = list_images_with_class(TEST_DIR)
    if not image_paths:
        raise RuntimeError(f"Khong tim thay anh nao trong: {TEST_DIR}")

    unknown = set(class_raw) - set(class_names)
    if unknown:
        print(f"\n[CANH BAO] Cac thu muc sau trong TEST_DIR khong khop ten class "
              f"luc train ({class_names}): {sorted(unknown)} - anh trong cac thu "
              f"muc nay se bi BO QUA.")

    print(f"\nTong so anh test: {len(image_paths)}")
    print(f"Cac class tim thay trong TEST_DIR: {sorted(set(class_raw))}")

    results = []
    with torch.no_grad():
        for path, cls_true in zip(image_paths, class_raw):
            if cls_true not in class_names:
                continue
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_cls = class_names[pred_idx]
            confidence = probs[pred_idx].item()

            results.append({
                "path": path, "class_true": cls_true, "class_pred": pred_cls,
                "confidence": confidence, "probs": probs.cpu().tolist(),
            })

    # --- Accuracy tong ---
    n = len(results)
    n_correct = sum(1 for r in results if r["class_pred"] == r["class_true"])
    acc = n_correct / n if n else 0.0

    print("\n" + "=" * 70)
    print("KET QUA DANH GIA TREN TAP TEST")
    print("=" * 70)
    print(f"Tong so anh: {n} | Dung: {n_correct} | Accuracy: {acc:.2%}")

    # --- Confusion matrix day du (khong chi OK/NG, ma tung class that su) ---
    print("\nConfusion matrix (hang = that, cot = doan):")
    header = "".ljust(15) + "".join(f"{c[:10]:>12s}" for c in class_names)
    print(header)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for r in results:
        i = class_names.index(r["class_true"])
        j = class_names.index(r["class_pred"])
        confusion[i, j] += 1
    for i, cls in enumerate(class_names):
        row = cls.ljust(15) + "".join(f"{confusion[i, j]:>12d}" for j in range(len(class_names)))
        print(row)

    # --- Accuracy theo tung class ---
    print("\nAccuracy theo tung class:")
    print(f"{'Class':<15}{'So anh':>8}{'Dung':>8}{'Accuracy':>12}")
    print("-" * 45)
    for cls in class_names:
        idx = [r for r in results if r["class_true"] == cls]
        total = len(idx)
        if total == 0:
            continue
        correct = sum(1 for r in idx if r["class_pred"] == cls)
        print(f"{cls:<15}{total:>8}{correct:>8}{correct/total:>11.2%}")

    # --- Danh sach doan sai ---
    wrong = [r for r in results if r["class_pred"] != r["class_true"]]
    if wrong:
        print(f"\n{len(wrong)} anh doan SAI:")
        for r in wrong:
            print(f"  {os.path.basename(r['path'])}: that={r['class_true']} "
                  f"doan={r['class_pred']} (tin cay={r['confidence']:.2%})")

    # --- Xuat CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Class_that", "Class_doan", "Dung_Sai", "Tin_cay"] +
                         [f"Xac_suat_{c}" for c in class_names])
        for r in results:
            status = "Dung" if r["class_pred"] == r["class_true"] else "Sai"
            row = [os.path.basename(r["path"]), r["class_true"], r["class_pred"],
                   status, f"{r['confidence']:.4f}"]
            row += [f"{p:.4f}" for p in r["probs"]]
            writer.writerow(row)
    print(f"\nDa xuat bao cao chi tiet ra CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()