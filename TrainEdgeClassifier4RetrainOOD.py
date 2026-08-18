"""
TRAIN OOD DETECTOR TỪ CLASSIFIER FINE-TUNED
============================================
Thay đổi so với bản cũ (centroid + IsolationForest trên ImageNet backbone):

1) Dùng BACKBONE CỦA CLASSIFIER FINE-TUNED (edge_classifier_fewshot_*.pt) để trích
   embedding -> feature đã học riêng cho vật của bạn, tách ID/OOD tốt hơn hẳn
   backbone ImageNet đóng băng. (Đây cũng là backbone cho heatmap Grad-CAM chuẩn.)

2) BỎ AUGMENTATION khi fit OOD. Augment mạnh làm "nở" vùng ID -> OOD lọt qua.
   Chỉ dùng embedding CLEAN.

3) Thay IsolationForest bằng 2 baseline mạnh trên penultimate feature:
      - Cosine-to-centroid
      - Mahalanobis (shared covariance, LedoitWolf shrinkage cho ổn định ở dim cao / few-shot)
   Đánh giá cả hai bằng AUROC + FPR@95TPR, tự chọn cái tốt hơn.

4) CALIBRATE NGƯỠNG bằng chính mẫu lạ (OOD) của bạn, trên các tập tách bạch:
      ID-fit  : fit centroid / covariance
      ID-val  : chọn ngưỡng (giữ 95% ID)
      OOD-val : so sánh & chọn phương pháp (AUROC)
      OOD-test: báo cáo số cuối (chỉ đụng 1 lần)

Payload lưu ra tương thích để file inference LIVE nạp (cần cập nhật inference tương ứng).
"""

import os
import time
import glob
import hashlib
import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ==========================================================================
# CẤU HÌNH
# ==========================================================================
# Dữ liệu ID (các class đã biết) -- vẫn theo cấu trúc folder/class
ID_DIRS = [
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1\train",
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1\val",
]
# Dữ liệu OOD (mẫu lạ) -- ảnh nằm phẳng hay trong folder con đều được, quét đệ quy
OOD_DIRS = [
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Unknow",
]

# Classifier fine-tuned (nguồn của backbone + heatmap chuẩn)
CLASSIFIER_CKPT = r"D:\TongHop\RTC Technologi\G8\model\model3\runs\20260817_190954\edge_classifier_fewshot_mobilenet_v3_large.pt"

MODEL_DIR = r"D:\TongHop\RTC Technologi\G8\model\model6"

# --- KẾT QUẢ TRAIN: mỗi lần chạy ra 1 file MỚI, KHÔNG ghi đè model cũ ---
OOD_OUT_DIR = os.path.join(MODEL_DIR, "ood_runs")
RUN_NAME = time.strftime("%Y%m%d_%H%M%S")   # hoặc đặt tay, vd "v3_maha_thu2"
OOD_PAYLOAD_OUT = os.path.join(OOD_OUT_DIR, f"ood_detector_{RUN_NAME}.joblib")
# Con trỏ tiện dụng tới payload mới nhất (chỉ là 1 file text nhỏ, KHÔNG phải model)
LATEST_POINTER = os.path.join(OOD_OUT_DIR, "ood_detector_latest.txt")

# --- CACHE / ĐỒ DÙNG CHUNG: lưu RIÊNG, tách khỏi model, tái dùng giữa các lần train ---
SHARED_DIR = os.path.join(MODEL_DIR, "_shared")
CACHE_DIR = os.path.join(SHARED_DIR, "emb_cache_clf")
USE_CACHE = True

# Chia tách
ID_VAL_RATIO = 0.30      # phần ID giữ ra để calibrate ngưỡng
OOD_TEST_RATIO = 0.50    # phần OOD giữ ra để báo cáo số cuối
TARGET_TPR = 0.95        # giữ 95% ID -> ngưỡng
RANDOM_SEED = 42
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==========================================================================
# MODEL FINE-TUNED
# ==========================================================================
def build_model_architecture(backbone_name, num_classes):
    """Phải khớp đúng kiến trúc lúc train classifier để load_state_dict được."""
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(960, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(576, num_classes))
    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone_name}")
    return model


def load_finetuned_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy classifier: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    backbone_name = ckpt.get("backbone_name", "mobilenet_v3_large")
    model = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Đã load classifier: {backbone_name} | {len(class_names)} class: {class_names}")
    return model, class_names, backbone_name


@torch.no_grad()
def _embed_batch(model, tensors):
    x = torch.stack(tensors).to(DEVICE)
    feat = model.features(x)                       # [B, C, 7, 7]
    pooled = F.adaptive_avg_pool2d(feat, (1, 1))   # đúng embedding mà classifier "nhìn"
    return torch.flatten(pooled, 1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_paths(model, paths):
    """Trả embedding CLEAN [N, D] theo đúng thứ tự paths. Không augment."""
    embs, batch = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(clean_transform(img))
        if len(batch) == BATCH_SIZE:
            embs.append(_embed_batch(model, batch)); batch = []
    if batch:
        embs.append(_embed_batch(model, batch))
    return np.concatenate(embs, 0) if embs else np.zeros((0, 0), np.float32)


# ------- Cache toàn khối theo (ckpt + tập ảnh) để chạy lại cho nhanh -------
def _cache_key(paths, ckpt_path):
    h = hashlib.md5()
    h.update(os.path.abspath(ckpt_path).encode())
    h.update(str(os.path.getmtime(ckpt_path)).encode())
    for p in paths:  # giữ nguyên thứ tự -> khớp label
        h.update(p.encode())
        h.update(str(os.path.getmtime(p)).encode())
        h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


def embed_with_cache(model, paths, ckpt_path, tag):
    if not paths:
        return np.zeros((0, 0), np.float32)
    if not USE_CACHE:
        return embed_paths(model, paths)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"{tag}_{_cache_key(paths, ckpt_path)}.npz")
    if os.path.isfile(cp):
        try:
            return np.load(cp)["X"]
        except Exception:
            pass
    X = embed_paths(model, paths)
    try:
        np.savez(cp, X=X)
    except Exception:
        pass
    return X


# ==========================================================================
# LIỆT KÊ ẢNH
# ==========================================================================
def list_id_images(dirs, class_names):
    """Trả (paths, labels) map theo đúng thứ tự class trong classifier."""
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy: {d}")
        for cls in os.listdir(d):
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in class_to_idx:
                print(f"  [!] Bỏ qua folder '{cls}' (không có trong class của classifier)")
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(cls_dir, fn))
                    labels.append(class_to_idx[cls])
    return paths, np.array(labels, dtype=np.int64)


def list_ood_images(dirs):
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  [!] Không thấy thư mục OOD: {d}")
            continue
        for ext in IMG_EXTS:
            paths += glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True)
    return sorted(set(paths))


# ==========================================================================
# PHƯƠNG PHÁP OOD
# ==========================================================================
def l2norm(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


# ----- Cosine-to-centroid ----- (score cao = giống ID)
def fit_centroids(X, y, num_classes):
    Xn = l2norm(X)
    cents = []
    for c in range(num_classes):
        Xc = Xn[y == c]
        if len(Xc) == 0:
            cents.append(np.zeros(X.shape[1], np.float32)); continue
        m = Xc.mean(0)
        cents.append((m / (np.linalg.norm(m) + 1e-8)).astype(np.float32))
    return np.stack(cents)


def score_cosine(X, centroids):
    return (l2norm(X) @ centroids.T).max(1)


# ----- Mahalanobis (shared covariance, shrinkage) ----- (score cao = giống ID)
def fit_mahalanobis(X, y, num_classes):
    means, centered = [], []
    for c in range(num_classes):
        Xc = X[y == c]
        mu = Xc.mean(0) if len(Xc) else np.zeros(X.shape[1], np.float32)
        means.append(mu)
        if len(Xc):
            centered.append(Xc - mu)
    means = np.stack(means).astype(np.float32)
    centered = np.concatenate(centered, 0)
    # LedoitWolf: shrinkage ước lượng covariance ổn định khi n < d (few-shot, dim 960)
    precision = LedoitWolf().fit(centered).precision_.astype(np.float32)
    return means, precision


def score_mahalanobis(X, means, precision):
    N, C = X.shape[0], means.shape[0]
    dists = np.empty((N, C), np.float64)
    for c in range(C):
        d = X - means[c]
        dists[:, c] = np.einsum('nd,de,ne->n', d, precision, d)
    return -dists.min(1)


# ----- Metrics -----
def auroc(id_scores, ood_scores):
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    return roc_auc_score(y, s)


def threshold_at_tpr(id_val_scores, target_tpr):
    # Giữ target_tpr phần ID -> ngưỡng là quantile (1 - tpr) của điểm ID
    return float(np.quantile(id_val_scores, 1.0 - target_tpr))


def fpr_at_threshold(ood_scores, tau):
    return float(np.mean(ood_scores >= tau))  # OOD bị nhận nhầm là ID


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print(f"--- Thiết bị: {DEVICE} ---")
    model, class_names, backbone_name = load_finetuned_model(CLASSIFIER_CKPT)
    num_classes = len(class_names)

    # 1) Liệt kê + trích embedding CLEAN
    id_paths, y_all = list_id_images(ID_DIRS, class_names)
    ood_paths = list_ood_images(OOD_DIRS)
    print(f"ID: {len(id_paths)} ảnh | OOD: {len(ood_paths)} ảnh")
    if len(id_paths) == 0:
        raise RuntimeError("Không có ảnh ID.")

    print("\n[1/4] Trích embedding (clean, từ backbone fine-tuned)...")
    t0 = time.time()
    X_all = embed_with_cache(model, id_paths, CLASSIFIER_CKPT, "id")
    X_ood_all = embed_with_cache(model, ood_paths, CLASSIFIER_CKPT, "ood")
    D = X_all.shape[1]
    print(f"-> Xong {time.time()-t0:.1f}s | embedding_dim = {D}")

    have_ood = X_ood_all.shape[0] >= 2

    # 2) Chia tách ID-fit / ID-val, OOD-val / OOD-test
    print("\n[2/4] Chia tập...")
    try:
        X_fit, X_val, y_fit, _ = train_test_split(
            X_all, y_all, test_size=ID_VAL_RATIO, random_state=RANDOM_SEED, stratify=y_all
        )
    except ValueError:
        print("  [!] Có class quá ít mẫu để stratify -> chia ngẫu nhiên.")
        X_fit, X_val, y_fit, _ = train_test_split(
            X_all, y_all, test_size=ID_VAL_RATIO, random_state=RANDOM_SEED
        )
    print(f"  ID-fit={len(X_fit)} | ID-val={len(X_val)}")

    if have_ood:
        X_ood_val, X_ood_test = train_test_split(
            X_ood_all, test_size=OOD_TEST_RATIO, random_state=RANDOM_SEED
        )
        print(f"  OOD-val={len(X_ood_val)} | OOD-test={len(X_ood_test)}")
    else:
        X_ood_val = X_ood_test = np.zeros((0, D), np.float32)
        print("  [!] Không đủ mẫu OOD -> bỏ qua đánh giá, chỉ đặt ngưỡng theo ID.")

    # 3) Fit 2 phương pháp trên ID-fit, đánh giá trên val
    print("\n[3/4] Fit + đánh giá...")
    centroids = fit_centroids(X_fit, y_fit, num_classes)
    maha_means, maha_prec = fit_mahalanobis(X_fit, y_fit, num_classes)

    methods = {
        "cosine": lambda X: score_cosine(X, centroids),
        "mahalanobis": lambda X: score_mahalanobis(X, maha_means, maha_prec),
    }

    results = {}
    print(f"\n  {'method':12s} {'AUROC':>7s} {'FPR@95TPR(val)':>15s} {'FPR@95TPR(test)':>16s}")
    for name, fn in methods.items():
        s_id_val = fn(X_val)
        tau = threshold_at_tpr(s_id_val, TARGET_TPR)
        if have_ood:
            au = auroc(s_id_val, fn(X_ood_val))
            fpr_val = fpr_at_threshold(fn(X_ood_val), tau)
            fpr_test = fpr_at_threshold(fn(X_ood_test), tau)
        else:
            au = fpr_val = fpr_test = float("nan")
        results[name] = {"tau": tau, "auroc": au, "fpr_val": fpr_val, "fpr_test": fpr_test}
        print(f"  {name:12s} {au:7.3f} {fpr_val:15.3f} {fpr_test:16.3f}")

    # Chọn phương pháp: AUROC cao nhất (nếu có OOD), không thì mặc định cosine
    if have_ood:
        best_method = max(results, key=lambda k: results[k]["auroc"])
    else:
        best_method = "cosine"
    print(f"\n  => Chọn: {best_method}  (tau={results[best_method]['tau']:.4f})")

    # 4) Lưu payload
    # Ghi chú: model deploy fit trên ID-fit, tau calibrate trên ID-val (không optimism).
    # Nếu muốn tận dụng toàn bộ ID, sau khi đã hài lòng metrics có thể set ID_VAL_RATIO nhỏ lại
    # và chạy lại (chấp nhận tau kém độc lập hơn một chút).
    print("\n[4/4] Lưu payload...")
    cosine_tau = results["cosine"]["tau"]  # giữ cả ngưỡng cosine cho nhãn/secondary gate
    payload = {
        "class_names": class_names,
        "backbone_name": backbone_name,
        "embedding_dim": int(D),
        "classifier_ckpt": CLASSIFIER_CKPT,   # inference nạp CHÍNH model này (1 model duy nhất)
        "centroids": centroids,               # [C, D] unit -> cosine + chọn nhãn + prototype heatmap
        "ood_method": best_method,            # "cosine" | "mahalanobis"
        "threshold": float(results[best_method]["tau"]),   # accept nếu score(method) >= threshold
        "cosine_threshold": float(cosine_tau),             # gate phụ / hiển thị độ tin cậy
        "mahalanobis": {"means": maha_means, "precision": maha_prec},
        "target_tpr": TARGET_TPR,
        "metrics": results,
    }
    os.makedirs(OOD_OUT_DIR, exist_ok=True)
    joblib.dump(payload, OOD_PAYLOAD_OUT)
    # Ghi con trỏ tới file mới nhất (không đụng các file model cũ)
    try:
        with open(LATEST_POINTER, "w", encoding="utf-8") as f:
            f.write(OOD_PAYLOAD_OUT)
    except Exception:
        pass

    print("\n" + "=" * 56)
    print(" HOÀN TẤT")
    print(f" Run         : {RUN_NAME}")
    print(f" OOD payload : {OOD_PAYLOAD_OUT}")
    print(f" Latest ptr  : {LATEST_POINTER}")
    print(f" Method      : {best_method} | threshold = {payload['threshold']:.4f}")
    if have_ood:
        print(f" AUROC       : {results[best_method]['auroc']:.3f}")
        print(f" FPR@95TPR   : {results[best_method]['fpr_test']:.3f} (OOD-test)")
    print("=" * 56)
    print("\nLƯU Ý: file inference LIVE cần cập nhật để nạp payload này")
    print("       (load classifier_ckpt, dùng ood_method + threshold, giữ Grad-CAM cũ).")


if __name__ == "__main__":
    main()