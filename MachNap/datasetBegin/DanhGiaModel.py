"""
EVALUATE: Đánh giá classifier + OOD gate trên tập test CÓ NHÃN
=============================================================
Dùng lại đúng:
  - Cách load model (backbone từ checkpoint)
  - OOD payload (mahalanobis/cosine + threshold đã calibrate)
  - Định nghĩa embedding (pooled features) y hệt live_detect.py
  -> nên điểm số OOD ở đây khớp với lúc chạy thật.

HƯỚNG ĐÁNH GIÁ (theo lựa chọn của bạn):
  - Đánh giá CẢ PIPELINE: ảnh bị OOD gate từ chối -> nhãn "UNKNOWN".
  - Có thư mục OOD thật -> chấm luôn chất lượng OOD gate (TPR/FPR).

CẤU TRÚC TẬP TEST (bắt buộc): mỗi class 1 thư mục con, tên = tên class.
  test_dir/
    ├─ Part1/  *.jpg ...
    ├─ Part2/  *.jpg ...
    ├─ ...
    └─ UNKNOWN/  *.jpg ...   (thư mục ảnh vật lạ / OOD thật)

Tên thư mục coi là OOD nếu nằm trong OOD_FOLDER_NAMES (không phân biệt hoa/thường).

XUẤT RA (trong OUTPUT_DIR):
  - per_class_accuracy.png       : biểu đồ độ chính xác (recall) từng class
  - confusion_matrix.png         : ma trận nhầm lẫn (gồm cả cột/hàng UNKNOWN)
  - precision_recall_f1.png      : precision / recall / F1 từng class
  - ood_score_distribution.png   : phân bố điểm OOD (ID vs OOD) + ngưỡng
  - per_class_metrics.csv        : số liệu từng class
  - predictions.csv              : dự đoán từng ảnh (để soi lỗi)
  - metrics_summary.txt          : tổng hợp + số liệu OOD gate

CHẠY:
  python evaluate.py
  python evaluate.py <thu_muc_test>
"""

import os
import sys
import csv
import time

import joblib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CẤU HÌNH  (sửa cho khớp máy bạn)
# ============================================================

# Thư mục test: mỗi class 1 subfolder (xem docstring)
TEST_DIR = r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1"

OOD_RUNS_DIR = r"D:\TongHop\RTC Technologi\G8\model\model4\ood_runs"
OOD_PAYLOAD_PATH = ""      # để trống -> đọc con trỏ ood_detector_latest.txt

OUTPUT_DIR = r"D:\TongHop\RTC Technologi\G8\eval_results"

# Tên thư mục được coi là OOD thật (không phân biệt hoa/thường)
OOD_FOLDER_NAMES = {"unknown", "ood", "unknown_ood", "vat_la", "vatla"}

# Bật OOD gate (đánh giá cả pipeline). Đặt False để so sánh classifier thuần.
APPLY_OOD_GATE = True
USE_COSINE_GATE = False     # giống live_detect.py

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRANSFORM (giống hệt live_detect.py)
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ============================================================
# TIỆN ÍCH ĐỌC ẢNH (hỗ trợ path Unicode/tiếng Việt)
# ============================================================

def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def list_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(folder, name))
    return files


# ============================================================
# 2. LOAD MODEL + OOD  (bê nguyên logic từ live_detect.py)
# ============================================================

def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(960, num_classes),
        )

    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(576, num_classes),
        )

    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone_name}")

    return model


def resolve_payload_path():
    if OOD_PAYLOAD_PATH:
        return OOD_PAYLOAD_PATH

    pointer = os.path.join(OOD_RUNS_DIR, "ood_detector_latest.txt")
    if not os.path.exists(pointer):
        raise FileNotFoundError(
            f"Không thấy con trỏ latest: {pointer}\n"
            "Hãy đặt OOD_PAYLOAD_PATH trỏ tay tới file .joblib."
        )

    with open(pointer, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_everything():
    payload_path = resolve_payload_path()
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Không thấy payload OOD: {payload_path}")

    payload = joblib.load(payload_path)
    print("Đã load payload OOD:", payload_path)
    print(
        f"  method={payload['ood_method']}"
        f" | threshold={payload['threshold']:.4f}"
        f" | target_tpr={payload.get('target_tpr')}"
    )

    ckpt_path = payload["classifier_ckpt"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Không thấy classifier mà payload trỏ tới: {ckpt_path}"
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    backbone_name = ckpt.get(
        "backbone_name",
        payload.get("backbone_name", "mobilenet_v3_large"),
    )

    model = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()

    print(f"Đã load classifier: {backbone_name} | {len(class_names)} class\n")
    return model, class_names, payload


# ============================================================
# 3. FORWARD (batched) -> probs + embedding trong 1 lần
# ------------------------------------------------------------
# Với mobilenet_v2/v3: forward = features -> avgpool -> flatten -> classifier.
# Nên "pooled flatten" ở đây chính là embedding mà OOD payload dùng.
# ============================================================

@torch.no_grad()
def forward_batch(model, batch_tensor):
    feat = model.features(batch_tensor)
    pooled = F.adaptive_avg_pool2d(feat, (1, 1))
    emb = torch.flatten(pooled, 1)
    logits = model.classifier(emb)
    probs = F.softmax(logits, dim=1)
    return (
        probs.cpu().numpy().astype(np.float32),
        emb.cpu().numpy().astype(np.float32),
    )


# ============================================================
# OOD SCORE (batched) — cùng công thức live_detect.py
# ============================================================

def l2norm(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def score_cosine(X, centroids):
    return (l2norm(X) @ centroids.T).max(1)


def score_mahalanobis(X, means, precision):
    N, C = X.shape[0], means.shape[0]
    dists = np.empty((N, C), np.float64)
    for c in range(C):
        d = X - means[c]
        dists[:, c] = np.einsum("nd,de,ne->n", d, precision, d)
    return -dists.min(1)


def ood_scores_batch(emb, payload):
    """Trả về (score, best_cos) cho cả batch."""
    best_cos = score_cosine(emb, payload["centroids"])

    if payload["ood_method"] == "mahalanobis":
        m = payload["mahalanobis"]
        score = score_mahalanobis(emb, m["means"], m["precision"])
    else:
        score = best_cos

    return score.astype(np.float64), best_cos.astype(np.float64)


# ============================================================
# 4. GOM TẬP TEST
# ============================================================

def gather_test_set(test_dir, class_names, unknown_idx):
    """
    Trả về:
      samples : list (path, true_idx, folder_name)
                true_idx = unknown_idx nếu folder là OOD
      unmatched : set tên folder không khớp class nào & không phải OOD
    """
    name_to_idx = {c.lower(): i for i, c in enumerate(class_names)}
    samples = []
    unmatched = set()

    for entry in sorted(os.listdir(test_dir)):
        sub = os.path.join(test_dir, entry)
        if not os.path.isdir(sub):
            continue

        imgs = list_images(sub)
        if not imgs:
            continue

        low = entry.lower()
        if low in OOD_FOLDER_NAMES:
            true_idx = unknown_idx
        elif low in name_to_idx:
            true_idx = name_to_idx[low]
        else:
            unmatched.add(entry)
            continue

        for p in imgs:
            samples.append((p, true_idx, entry))

    return samples, unmatched


# ============================================================
# 5. CHẠY INFERENCE TRÊN TOÀN TẬP TEST
# ============================================================

def run_inference(model, class_names, payload, samples, unknown_idx):
    threshold = float(payload["threshold"])
    cos_threshold = float(payload.get("cosine_threshold", -1))

    records = []      # dict mỗi ảnh
    y_true = []
    y_pred = []

    n = len(samples)
    t0 = time.time()

    batch_imgs = []
    batch_meta = []

    def flush(batch_imgs, batch_meta):
        if not batch_imgs:
            return
        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        probs, emb = forward_batch(model, batch_tensor)
        score, best_cos = ood_scores_batch(emb, payload)

        for i, (path, true_idx, folder) in enumerate(batch_meta):
            p = probs[i]
            cls_idx = int(p.argmax())
            conf = float(p[cls_idx])

            is_ood = bool(score[i] < threshold)
            if USE_COSINE_GATE and best_cos[i] < cos_threshold:
                is_ood = True

            if APPLY_OOD_GATE and is_ood:
                pred_idx = unknown_idx
            else:
                pred_idx = cls_idx

            y_true.append(true_idx)
            y_pred.append(pred_idx)

            records.append({
                "path": path,
                "folder": folder,
                "true_idx": true_idx,
                "true_name": (
                    "UNKNOWN" if true_idx == unknown_idx
                    else class_names[true_idx]
                ),
                "argmax_idx": cls_idx,
                "argmax_name": class_names[cls_idx],
                "confidence": conf,
                "ood_score": float(score[i]),
                "best_cos": float(best_cos[i]),
                "is_ood": is_ood,
                "pred_idx": pred_idx,
                "pred_name": (
                    "UNKNOWN" if pred_idx == unknown_idx
                    else class_names[pred_idx]
                ),
                "correct": (pred_idx == true_idx),
            })

    for k, (path, true_idx, folder) in enumerate(samples):
        img = imread_unicode(path)
        if img is None:
            print(f"  [!] Lỗi đọc ảnh, bỏ qua: {path}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb))
        batch_imgs.append(tensor)
        batch_meta.append((path, true_idx, folder))

        if len(batch_imgs) >= BATCH_SIZE:
            flush(batch_imgs, batch_meta)
            batch_imgs, batch_meta = [], []
            done = k + 1
            print(f"  ...{done}/{n} ảnh "
                  f"({done / (time.time() - t0):.1f} ảnh/s)", end="\r")

    flush(batch_imgs, batch_meta)
    print(f"  Xong {len(records)}/{n} ảnh trong "
          f"{time.time() - t0:.1f}s" + " " * 20)

    return records, np.array(y_true), np.array(y_pred)


# ============================================================
# 6. TÍNH METRIC (thuần numpy, không cần sklearn)
# ============================================================

def confusion_matrix_np(y_true, y_pred, n_labels):
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """Trả precision, recall, f1, support cho từng nhãn."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)     # tổng thật của class
    pred_sum = cm.sum(axis=0).astype(np.float64)    # tổng dự đoán class

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    return precision, recall, f1, support.astype(np.int64)


# ============================================================
# 7. VẼ BIỂU ĐỒ
# ============================================================

def _bar_labels(ax, bars, values, fmt="{:.0f}"):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 1,
                fmt.format(v),
                ha="center", va="bottom", fontsize=8)


def plot_per_class_accuracy(labels, recall, support, out_path):
    acc = recall * 100
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 6))

    colors = ["#e74c3c" if a < 80 else "#f39c12" if a < 95 else "#2ecc71"
              for a in acc]
    bars = ax.bar(range(n), acc, color=colors)
    _bar_labels(ax, bars, acc, "{:.1f}")

    mean_acc = acc[support > 0].mean() if (support > 0).any() else 0
    ax.axhline(mean_acc, color="#34495e", ls="--", lw=1,
               label=f"TB = {mean_acc:.1f}%")

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [f"{l}\n(n={s})" for l, s in zip(labels, support)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Độ chính xác / Recall (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Độ chính xác từng class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out_path, normalize=True):
    n = len(labels)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, np.where(row == 0, 1, row)) * 100
        fmt = "{:.0f}"
    else:
        data = cm.astype(float)
        fmt = "{:.0f}"

    fig, ax = plt.subplots(figsize=(max(7, n * 0.7), max(6, n * 0.6)))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn"
                 + (" (% theo hàng)" if normalize else " (số lượng)"))

    thr = data.max() / 2 if data.max() else 0.5
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if v > 0:
                ax.text(j, i, fmt.format(v),
                        ha="center", va="center", fontsize=7,
                        color="white" if v > thr else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_prf(labels, precision, recall, f1, out_path):
    n = len(labels)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 6))

    ax.bar(x - w, precision * 100, w, label="Precision", color="#3498db")
    ax.bar(x,     recall * 100,    w, label="Recall",    color="#2ecc71")
    ax.bar(x + w, f1 * 100,        w, label="F1",        color="#9b59b6")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Precision / Recall / F1 từng class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ood_distribution(records, threshold, unknown_name, out_path):
    id_scores = [r["ood_score"] for r in records
                 if r["true_name"] != unknown_name]
    ood_scores = [r["ood_score"] for r in records
                  if r["true_name"] == unknown_name]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 40
    if id_scores:
        ax.hist(id_scores, bins=bins, alpha=0.6,
                label=f"Known / ID (n={len(id_scores)})", color="#2ecc71")
    if ood_scores:
        ax.hist(ood_scores, bins=bins, alpha=0.6,
                label=f"OOD thật (n={len(ood_scores)})", color="#e74c3c")

    ax.axvline(threshold, color="#2c3e50", ls="--", lw=2,
               label=f"Ngưỡng = {threshold:.3f}")
    ax.set_xlabel("OOD score (cao = giống known)")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố điểm OOD: known vs OOD\n"
                 "(bên trái ngưỡng bị từ chối = UNKNOWN)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================
# 8. GHI FILE SỐ LIỆU
# ============================================================

def save_predictions_csv(records, path):
    fields = ["path", "folder", "true_name", "pred_name", "correct",
              "argmax_name", "confidence", "ood_score", "best_cos", "is_ood"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def save_per_class_csv(labels, precision, recall, f1, support, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1"])
        for i, l in enumerate(labels):
            w.writerow([l, support[i],
                        f"{precision[i]:.4f}",
                        f"{recall[i]:.4f}",
                        f"{f1[i]:.4f}"])


# ============================================================
# 9. MAIN
# ============================================================

def main():
    test_dir = TEST_DIR
    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        test_dir = sys.argv[1]

    if not os.path.isdir(test_dir):
        print("Thư mục test không tồn tại:", test_dir)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, class_names, payload = load_everything()

    # Nhãn đánh giá = các class đã biết + UNKNOWN (cột/hàng cuối)
    unknown_idx = len(class_names)
    eval_labels = list(class_names) + ["UNKNOWN"]
    n_labels = len(eval_labels)

    samples, unmatched = gather_test_set(test_dir, class_names, unknown_idx)
    if unmatched:
        print("[!] Bỏ qua thư mục không khớp class nào & không phải OOD:",
              ", ".join(sorted(unmatched)))
    if not samples:
        print("Không có ảnh hợp lệ trong tập test.")
        sys.exit(1)

    print(f"Tổng {len(samples)} ảnh test. Đang đánh giá...\n")

    records, y_true, y_pred = run_inference(
        model, class_names, payload, samples, unknown_idx
    )

    # ---- Ma trận nhầm lẫn + metric ----
    cm = confusion_matrix_np(y_true, y_pred, n_labels)
    precision, recall, f1, support = per_class_metrics(cm)

    overall_acc = (y_true == y_pred).mean() * 100

    # macro chỉ tính trên nhãn có mẫu thật
    has = support > 0
    macro_p = precision[has].mean() * 100
    macro_r = recall[has].mean() * 100
    macro_f1 = f1[has].mean() * 100

    # ---- Số liệu riêng cho OOD gate ----
    n_true_ood = int((y_true == unknown_idx).sum())
    n_true_known = int((y_true != unknown_idx).sum())

    # OOD bị từ chối đúng (TPR) / known bị từ chối nhầm (FPR)
    ood_caught = sum(
        1 for r in records
        if r["true_name"] == "UNKNOWN" and r["is_ood"]
    )
    known_rejected = sum(
        1 for r in records
        if r["true_name"] != "UNKNOWN" and r["is_ood"]
    )
    tpr = (ood_caught / n_true_ood * 100) if n_true_ood else float("nan")
    fpr = (known_rejected / n_true_known * 100) if n_true_known else float("nan")

    # Accuracy chỉ trên ảnh known được nhận (không bị OOD từ chối) -> phân loại đúng?
    known_accepted = [r for r in records
                      if r["true_name"] != "UNKNOWN" and not r["is_ood"]]
    cls_acc_on_accepted = (
        sum(1 for r in known_accepted if r["argmax_idx"] == r["true_idx"])
        / len(known_accepted) * 100
    ) if known_accepted else float("nan")

    # ---- Vẽ biểu đồ ----
    p_acc = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    p_cm = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    p_prf = os.path.join(OUTPUT_DIR, "precision_recall_f1.png")
    p_ood = os.path.join(OUTPUT_DIR, "ood_score_distribution.png")

    plot_per_class_accuracy(eval_labels, recall, support, p_acc)
    plot_confusion_matrix(cm, eval_labels, p_cm, normalize=True)
    plot_prf(eval_labels, precision, recall, f1, p_prf)
    plot_ood_distribution(records, float(payload["threshold"]),
                          "UNKNOWN", p_ood)

    # ---- Ghi CSV ----
    save_predictions_csv(records, os.path.join(OUTPUT_DIR, "predictions.csv"))
    save_per_class_csv(eval_labels, precision, recall, f1, support,
                       os.path.join(OUTPUT_DIR, "per_class_metrics.csv"))

    # ---- Tổng hợp text ----
    lines = []
    lines.append("=" * 60)
    lines.append("KẾT QUẢ ĐÁNH GIÁ (FULL PIPELINE, OOD gate = %s)"
                 % ("ON" if APPLY_OOD_GATE else "OFF"))
    lines.append("=" * 60)
    lines.append(f"Tổng ảnh test        : {len(records)}")
    lines.append(f"  - known            : {n_true_known}")
    lines.append(f"  - OOD thật         : {n_true_ood}")
    lines.append("")
    lines.append(f"Accuracy toàn pipeline: {overall_acc:.2f}%")
    lines.append(f"Macro precision       : {macro_p:.2f}%")
    lines.append(f"Macro recall          : {macro_r:.2f}%")
    lines.append(f"Macro F1              : {macro_f1:.2f}%")
    lines.append("")
    lines.append("--- OOD GATE ---")
    lines.append(f"Ngưỡng               : {float(payload['threshold']):.4f}")
    lines.append(f"TPR (bắt đúng OOD)    : {tpr:.2f}%  ({ood_caught}/{n_true_ood})")
    lines.append(f"FPR (từ chối nhầm)    : {fpr:.2f}%  ({known_rejected}/{n_true_known})")
    lines.append(f"Acc phân loại trên known được nhận: {cls_acc_on_accepted:.2f}%")
    lines.append("")
    lines.append("--- TỪNG CLASS ---")
    lines.append(f"{'class':<20}{'n':>6}{'prec':>9}{'recall':>9}{'f1':>9}")
    for i, l in enumerate(eval_labels):
        if support[i] == 0 and l == "UNKNOWN" and n_true_ood == 0:
            continue
        lines.append(f"{l:<20}{support[i]:>6}"
                     f"{precision[i] * 100:>8.1f}%"
                     f"{recall[i] * 100:>8.1f}%"
                     f"{f1[i] * 100:>8.1f}%")
    report = "\n".join(lines)

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print("\nĐã lưu biểu đồ + CSV + summary vào:", OUTPUT_DIR)


if __name__ == "__main__":
    main()