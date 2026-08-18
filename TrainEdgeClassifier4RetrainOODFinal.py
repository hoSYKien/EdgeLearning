"""
==============================================================================
PIPELINE TỔNG HỢP: FEW-SHOT CLASSIFIER + OOD DETECTOR (2-IN-1)
==============================================================================
Kiến trúc luồng xử lý:
1. Backbone CNN (MobileNetV2 / MobileNetV3 Large / Small) đóng băng.
2. Cache Embedding ra đĩa (_shared/emb_cache):
   - Thêm/bớt ảnh/class chỉ tính toán ảnh mới, tự dọn dẹp ảnh cũ (prune).
3. Giai đoạn 1 - Huấn luyện Phân loại (Classifier):
   - Augmentation (mặc định 20 copies) + Head Linear nhẹ.
   - K-Fold Cross Validation Stratified đánh giá không thiên vị (Confusion Matrix + CSV).
   - Huấn luyện Final Model trên toàn bộ tập ID.
4. Giai đoạn 2 - Huấn luyện Phát hiện Dị vật (OOD Detector):
   - Dùng trực tiếp Clean Embedding từ backbone đã học.
   - Fit 2 phương pháp: Cosine-to-centroid & Mahalanobis Distance (LedoitWolf shrinkage).
   - Tách tập ID-val (calibrate ngưỡng 95% TPR), OOD-val/test (đo AUROC, FPR@95TPR).
   - Tự động chọn giải thuật tối ưu nhất và lưu payload .joblib.
5. Versioning & An toàn dữ liệu:
   - Mỗi lần chạy lưu vào thư mục riêng: runs/YYYYMMDD_HHMMSS/ (KHÔNG ghi đè bản cũ).
   - Tự động cập nhật bản copy mới nhất ra ngoài thư mục gốc để inference tiện nạp.
   - Ghi nhật ký đầy đủ vào runs/runs_index.csv.
==============================================================================
"""

import os
import time
import glob
import json
import random
import shutil
import hashlib
import joblib
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ============================================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================================

# 1. Dataset cấu hình
ID_DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts1\train",
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts1\val",
]

# Thư mục chứa ảnh lạ (OOD / Dị vật / Unknown) -- nếu chưa có ảnh hoặc trống, code vẫn chạy an toàn
OOD_DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Unknow",
]

# Danh sách class muốn LOẠI BỎ (VD: {"Part7"}). Để trống set() = lấy tất cả.
EXCLUDE_CLASSES = set()

# 2. Thư mục lưu trữ model & cấu hình versioning
ROOT_MODEL_DIR = r"D:\TongHop\RTC Technologi\G8\modelClassifierOOD"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"

SAVE_VERSIONED = True                 # True = Lưu vào runs/<timestamp>/
KEEP_LATEST_COPY = True               # True = Copy bản mới nhất ra thư mục gốc
EXPORT_ONNX = True                    # True = Tự động xuất model ONNX để chạy GPU / TensorRT / C++ / C#
RUN_TAG = ""                          # Tag phụ nếu muốn gắn vào run_id (VD: "exp1")

# Thư mục con
SHARED_DIR = os.path.join(ROOT_MODEL_DIR, "_shared")
CACHE_DIR = os.path.join(SHARED_DIR, "emb_cache")
RUNS_ROOT = os.path.join(ROOT_MODEL_DIR, "runs")

USE_EMBEDDING_CACHE = True            # Bật/Tắt cache
CLEAR_CACHE = False                   # True = Xóa sạch cache để tính lại từ đầu
PRUNE_CACHE = True                    # True = Tự xóa cache của ảnh/class không còn trong dataset

# 3. Cấu hình Huấn luyện Classifier
AUGMENT_CONFIG = {
    "horizontal_flip": True,
    "rotation_degrees": 30,
    "color_jitter": True,
    "random_resized_crop": True,
    
    # --- CẤU HÌNH GRAYSCALE (CHỌN 1 TRONG CÁC KIỂU SAU) ---
    # "none"       : Giữ nguyên ảnh màu RGB (không đổi gì)
    # "clean_only" : [Kiểu 1] Chỉ chuyển ảnh gốc (clean) sang gray, ảnh augment giữ màu
    # "all"        : [Kiểu 2] Chuyển TOÀN BỘ (cả ảnh gốc lẫn toàn bộ ảnh làm giàu augment) sang gray
    # "random_aug" : Ảnh gốc giữ màu, ảnh augment biến đổi ngẫu nhiên sang gray với xác suất 50%
    "grayscale_mode": "random_aug",   # "none" | "clean_only" | "all" | "random_aug"
}
AUGMENT_COPIES = 25
K_FOLDS = 5

HEAD_EPOCHS = 400
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 50
PRINT_EPOCH_DETAILS = True

# 4. Cấu hình Huấn luyện OOD Detector
ID_VAL_RATIO = 0.30                   # Phần trăm ID giữ lại để calibrate ngưỡng
OOD_TEST_RATIO = 0.50                 # Phần trăm OOD giữ lại để test độc lập
TARGET_TPR = 0.95                     # Mục tiêu giữ 95% mẫu ID chuẩn

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# TRANSFORMS & MODEL BUILDER
# ============================================================================

def build_transforms(augment_cfg):
    gray_mode = augment_cfg.get("grayscale_mode", "none")

    # --- Pipeline cho Augment ---
    aug_ops = [transforms.Resize((224, 224))]
    if gray_mode == "all":
        aug_ops.append(transforms.Grayscale(num_output_channels=3))
    elif gray_mode == "random_aug":
        aug_ops.append(transforms.RandomGrayscale(p=0.5))

    if augment_cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(augment_cfg["rotation_degrees"]))
    if augment_cfg.get("random_resized_crop"):
        aug_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    if augment_cfg.get("color_jitter") and gray_mode != "all":
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    aug_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # --- Pipeline cho Clean (Ảnh chuẩn không biến dạng) ---
    clean_ops = [transforms.Resize((224, 224))]
    if gray_mode in ("clean_only", "all"):
        clean_ops.append(transforms.Grayscale(num_output_channels=3))
    clean_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(aug_ops), transforms.Compose(clean_ops)


def build_backbone(backbone_name):
    if backbone_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        embedding_dim = backbone.last_channel
    elif backbone_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        embedding_dim = 960
    elif backbone_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        embedding_dim = 576
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    backbone = backbone.to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, embedding_dim


def build_full_model(backbone_name):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    for p in model.features.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_embedding(backbone, image_tensor):
    x = backbone.features(image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


class UnifiedClassifierOODModel(nn.Module):
    """Gộp toàn bộ Backbone + Head Classifier + OOD Gate vào 1 đồ thị nơ-ron duy nhất"""
    def __init__(self, full_model, centroids, threshold):
        super().__init__()
        self.features = full_model.features
        self.classifier = full_model.classifier
        
        # Đưa centroids và threshold vào buffer đồ thị ONNX
        self.register_buffer("centroids", torch.tensor(centroids, dtype=torch.float32))  # [num_classes, embedding_dim]
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))

    def forward(self, x):
        # 1. Trích xuất đặc trưng
        feat = self.features(x)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        emb = torch.flatten(pooled, 1)  # [B, D]

        # 2. Phân loại
        logits = self.classifier(emb)  # [B, num_classes]
        probs = F.softmax(logits, dim=1)
        pred_class_id = torch.argmax(probs, dim=1)  # [B]

        # 3. Tính điểm OOD Cosine Similarity
        emb_norm = emb / (torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_sims = torch.matmul(emb_norm, self.centroids.t())  # [B, num_classes]
        ood_score, _ = torch.max(cosine_sims, dim=1)  # [B]

        # 4. Kiểm tra Dị vật (1 = Dị vật/Unknown, 0 = Hàng chuẩn)
        is_unknown = (ood_score < self.threshold).to(torch.int32)  # [B]

        return logits, pred_class_id, ood_score, is_unknown


# ============================================================================
# CACHE EMBEDDING FUNCTIONS
# ============================================================================

def _augment_signature(augment_cfg, augment_copies, backbone_name):
    payload = json.dumps({"cfg": augment_cfg, "copies": augment_copies, "backbone": backbone_name}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _image_key(path):
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


def _cache_path(path, sig):
    return os.path.join(CACHE_DIR, f"{_image_key(path)}_{sig}.npz")


def _load_cached(path, sig):
    cp = _cache_path(path, sig)
    if not os.path.isfile(cp):
        return None
    try:
        data = np.load(cp, allow_pickle=False)
        if float(data["mtime"]) != os.path.getmtime(path) or int(data["size"]) != os.path.getsize(path):
            return None
        aug = torch.from_numpy(data["aug"].astype(np.float32))
        clean = torch.from_numpy(data["clean"].astype(np.float32))
        return aug, clean
    except Exception:
        return None


def _save_cache(path, sig, aug_emb, clean_emb):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(
        _cache_path(path, sig),
        aug=aug_emb.cpu().numpy().astype(np.float32),
        clean=clean_emb.cpu().numpy().astype(np.float32),
        mtime=np.array(os.path.getmtime(path), dtype=np.float64),
        size=np.array(os.path.getsize(path), dtype=np.int64),
    )


def _prune_cache(all_known_paths):
    if not os.path.isdir(CACHE_DIR):
        return 0
    current_keys = {_image_key(p) for p in all_known_paths}
    removed = 0
    for cp in glob.glob(os.path.join(CACHE_DIR, "*.npz")):
        name = os.path.basename(cp)
        key = name.split("_")[0]
        if key not in current_keys:
            try:
                os.remove(cp)
                removed += 1
            except OSError:
                pass
    return removed


def precompute_id_embeddings(backbone, image_paths, augment_transform, clean_transform, augment_copies, sig):
    augmented_emb, clean_emb = [], []
    n_hit, n_miss = 0, 0
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        cached = _load_cached(path, sig) if USE_EMBEDDING_CACHE else None
        if cached is not None:
            aug_emb, c_emb = cached
            n_hit += 1
            status = "CACHE"
        else:
            image = Image.open(path).convert("RGB")
            aug_tensors = torch.stack([augment_transform(image) for _ in range(augment_copies)]).to(DEVICE)
            aug_emb = extract_embedding(backbone, aug_tensors).cpu()

            clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
            c_emb = extract_embedding(backbone, clean_tensor).cpu()

            if USE_EMBEDDING_CACHE:
                _save_cache(path, sig, aug_emb, c_emb)
            n_miss += 1
            status = "EXTRACT"

        augmented_emb.append(aug_emb)
        clean_emb.append(c_emb)

        # Hiển thị tiến trình realtime mượt mà trên cùng 1 dòng
        pct = (i / total) * 100
        fname = os.path.basename(path)
        cls_name = os.path.basename(os.path.dirname(path))
        print(f"\r  -> [ID Embedding] [{i:4d}/{total:4d}] ({pct:5.1f}%) | [{status:7s}] | {cls_name:<10s} | {fname[:20]:<20s}", end="", flush=True)

    print(f"\n  [ID Cache] Hoàn tất {total} ảnh (Dùng lại: {n_hit} ảnh | Trích xuất mới: {n_miss} ảnh)")
    return augmented_emb, clean_emb


def precompute_ood_embeddings(backbone, ood_paths, clean_transform, sig):
    if not ood_paths:
        return np.zeros((0, 0), dtype=np.float32)
    ood_clean_embs = []
    n_hit, n_miss = 0, 0
    total = len(ood_paths)

    for i, path in enumerate(ood_paths, 1):
        cached = _load_cached(path, sig) if USE_EMBEDDING_CACHE else None
        if cached is not None:
            _, c_emb = cached
            n_hit += 1
            status = "CACHE"
        else:
            image = Image.open(path).convert("RGB")
            clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
            c_emb = extract_embedding(backbone, clean_tensor).cpu()
            dummy_aug = c_emb.repeat(1, 1)  # placeholder nếu cần lưu cache
            if USE_EMBEDDING_CACHE:
                _save_cache(path, sig, dummy_aug, c_emb)
            n_miss += 1
            status = "EXTRACT"
        ood_clean_embs.append(c_emb.squeeze(0).numpy())

        # Hiển thị tiến trình realtime
        pct = (i / total) * 100
        fname = os.path.basename(path)
        print(f"\r  -> [OOD Embedding] [{i:4d}/{total:4d}] ({pct:5.1f}%) | [{status:7s}] | {fname[:25]:<25s}", end="", flush=True)

    print(f"\n  [OOD Cache] Hoàn tất {total} ảnh (Dùng lại: {n_hit} ảnh | Trích xuất mới: {n_miss} ảnh)")
    return np.stack(ood_clean_embs)


# ============================================================================
# DATASET LISTING
# ============================================================================

def list_id_images(dataset_dirs, exclude_classes=frozenset()):
    class_names = set()
    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy thư mục ID: {d}")
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name not in exclude_classes:
                class_names.add(entry.name)
    class_names = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    image_paths, labels = [], []
    for d in dataset_dirs:
        for cls in class_names:
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTS):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

    return image_paths, labels, class_names


def list_ood_images(dataset_dirs):
    ood_paths = []
    for d in dataset_dirs:
        if not os.path.isdir(d):
            print(f"  [!] Thư mục OOD không tồn tại: {d} (bỏ qua)")
            continue
        for ext in IMG_EXTS:
            ood_paths += glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True)
    return sorted(set(ood_paths))


# ============================================================================
# GIAI ĐOẠN 1: TRAINING & EVALUATION CLASSIFIER
# ============================================================================

def stratified_k_fold_indices(labels, k, seed=42):
    labels = np.array(labels)
    rng = np.random.RandomState(seed)
    fold_assignment = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        for i, idx in enumerate(idx_c):
            fold_assignment[idx] = i % k
    return fold_assignment


def make_head(embedding_dim, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)).to(DEVICE)


def train_head(train_emb, train_labels, val_emb, val_labels, embedding_dim, num_classes,
               epochs, lr, dropout, patience, print_epochs=False, fold_label=""):
    head = make_head(embedding_dim, num_classes, dropout)

    class_weights = train_emb.shape[0] / (
        num_classes * np.bincount(train_labels.cpu().numpy(), minlength=num_classes)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    has_val = val_emb is not None and val_emb.shape[0] > 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_probs = None
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    epochs_no_improve = 0
    min_delta = 1e-4  # Ngưỡng cải thiện tối thiểu để chống overfit và dừng sớm đúng lúc

    for epoch in range(epochs):
        epoch_start = time.time()
        head.train()
        optimizer.zero_grad()
        outputs = head(train_emb)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_acc = (outputs.argmax(dim=1) == train_labels).float().mean().item()

        if has_val:
            head.eval()
            with torch.no_grad():
                val_outputs = head(val_emb)
                val_loss = criterion(val_outputs, val_labels).item()
                val_probs = F.softmax(val_outputs, dim=1)
                val_acc = (val_probs.argmax(dim=1) == val_labels).float().mean().item()
            scheduler.step(val_acc)

            status = ""
            if (best_val_loss - val_loss) > min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_probs = val_probs.cpu()
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
                epochs_no_improve = 0
                status = "Best"
            else:
                epochs_no_improve += 1

            if print_epochs:
                epoch_time_ms = (time.time() - epoch_start) * 1000
                print(f"  [{fold_label}] Epoch {epoch+1:3d}/{epochs} | Train loss: {loss.item():.4f} "
                      f"| Train acc: {train_acc:.2%} | Val loss: {val_loss:.4f} "
                      f"| Val acc: {val_acc:.2%} | {status:<5} | {epoch_time_ms:.1f}ms")

            if epochs_no_improve >= patience:
                if print_epochs:
                    print(f"  [{fold_label}] Val loss không cải thiện sau {patience} epoch -> dừng sớm.")
                break
        else:
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            if print_epochs:
                epoch_time_ms = (time.time() - epoch_start) * 1000
                print(f"  [{fold_label}] Epoch {epoch+1:3d}/{epochs} | Train loss: {loss.item():.4f} "
                      f"| Train acc: {train_acc:.2%} | {epoch_time_ms:.1f}ms")

    head.load_state_dict(best_state)
    return head, best_val_acc, best_val_loss, best_val_probs


def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names, output_dir):
    num_classes = len(class_names)
    n = len(image_paths)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(n):
        confusion[labels_np[i], oof_pred[i]] += 1

    print("\n" + "=" * 90)
    print("BÁO CÁO ĐÁNH GIÁ CLASSIFIER (K-FOLD OUT-OF-FOLD)")
    print("=" * 90)

    print("\nConfusion Matrix (Hàng: Thực tế, Cột: Dự đoán):")
    header = "".ljust(15) + "".join(f"{c[:10]:>12s}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = row_name.ljust(15) + "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes))
        print(row_str)

    print("\nĐộ chính xác & Confidence theo từng Class:")
    print(f"{'Class':<15}{'Số ảnh':>8}{'Đúng':>8}{'Accuracy':>12}{'Conf TB (Đúng)':>20}{'Conf TB (Sai)':>18}")
    print("-" * 80)
    for c in range(num_classes):
        idx_c = [i for i in range(n) if labels_np[i] == c]
        total_c = len(idx_c)
        correct_c = sum(1 for i in idx_c if oof_pred[i] == c)
        acc_c = correct_c / total_c if total_c else 0.0

        conf_correct = [oof_probs[i][oof_pred[i]] for i in idx_c if oof_pred[i] == c]
        conf_wrong = [oof_probs[i][oof_pred[i]] for i in idx_c if oof_pred[i] != c]
        avg_conf_correct = np.mean(conf_correct) if conf_correct else 0.0
        avg_conf_wrong = np.mean(conf_wrong) if conf_wrong else 0.0

        print(f"{class_names[c]:<15}{total_c:>8}{correct_c:>8}{acc_c:>11.2%} "
              f"{avg_conf_correct:>19.2%} {avg_conf_wrong:>17.2%}")

    overall_acc = sum(1 for i in range(n) if oof_pred[i] == labels_np[i]) / n
    print("-" * 80)
    print(f"{'TỔNG CỘNG':<15}{n:>8}{sum(1 for i in range(n) if oof_pred[i]==labels_np[i]):>8}{overall_acc:>11.2%}")

    csv_path = os.path.join(output_dir, "kfold_detailed_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Fold", "Class_that", "Class_doan", "Dung_Sai"] +
                         [f"Xac_suat_{c}" for c in class_names])
        for i in range(n):
            fname = os.path.basename(image_paths[i])
            true_cls = class_names[labels_np[i]]
            pred_cls = class_names[oof_pred[i]]
            status = "Dung" if oof_pred[i] == labels_np[i] else "Sai"
            row = [fname, oof_fold[i] + 1, true_cls, pred_cls, status] + [f"{p:.4f}" for p in oof_probs[i]]
            writer.writerow(row)
    print(f"\n-> Đã lưu báo cáo chi tiết K-Fold: {csv_path}")


# ============================================================================
# GIAI ĐOẠN 2: TRAINING & EVALUATION OOD DETECTOR
# ============================================================================

def l2norm(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def fit_centroids(X, y, num_classes):
    Xn = l2norm(X)
    cents = []
    for c in range(num_classes):
        Xc = Xn[y == c]
        if len(Xc) == 0:
            cents.append(np.zeros(X.shape[1], np.float32))
            continue
        m = Xc.mean(0)
        cents.append((m / (np.linalg.norm(m) + 1e-8)).astype(np.float32))
    return np.stack(cents)


def score_cosine(X, centroids):
    return (l2norm(X) @ centroids.T).max(1)


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
    precision = LedoitWolf().fit(centered).precision_.astype(np.float32)
    return means, precision


def score_mahalanobis(X, means, precision):
    N, C = X.shape[0], means.shape[0]
    dists = np.empty((N, C), np.float64)
    for c in range(C):
        d = X - means[c]
        dists[:, c] = np.einsum("nd,de,ne->n", d, precision, d)
    return -dists.min(1)


def auroc(id_scores, ood_scores):
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    return roc_auc_score(y, s)


def threshold_at_tpr(id_val_scores, target_tpr):
    return float(np.quantile(id_val_scores, 1.0 - target_tpr))


def fpr_at_threshold(ood_scores, tau):
    return float(np.mean(ood_scores >= tau))


# ============================================================================
# NHẬT KÝ TRAIN (RUNS INDEX)
# ============================================================================

def _append_runs_index(index_path, run_info_dict):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    write_header = not os.path.isfile(index_path)
    with open(index_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "run_id", "timestamp", "backbone", "num_classes", "num_id_images",
                "num_ood_images", "kfold_acc_mean", "ood_method", "ood_threshold",
                "ood_auroc", "ood_fpr_at_95tpr", "latency_ms", "model_size_mb"
            ])
        w.writerow([
            run_info_dict["run_id"],
            run_info_dict["timestamp"],
            run_info_dict["backbone_name"],
            run_info_dict["num_classes"],
            run_info_dict["num_id_images"],
            run_info_dict["num_ood_images"],
            f"{run_info_dict['kfold_val_acc_mean']:.4f}",
            run_info_dict["ood_best_method"],
            f"{run_info_dict['ood_threshold']:.4f}",
            f"{run_info_dict['ood_auroc']:.4f}" if not np.isnan(run_info_dict['ood_auroc']) else "N/A",
            f"{run_info_dict['ood_fpr_test']:.4f}" if not np.isnan(run_info_dict['ood_fpr_test']) else "N/A",
            f"{run_info_dict['avg_latency_ms']:.2f}",
            f"{run_info_dict['model_size_mb']:.2f}",
        ])
    print(f"-> Đã cập nhật bảng lịch sử chạy: {index_path}")


# ============================================================================
# MAIN PIPELINE 2-IN-1
# ============================================================================

def main():
    print("=" * 90)
    print(" BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN FEW-SHOT CLASSIFIER + OOD DETECTOR")
    print("=" * 90)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Thiết bị: GPU ({DEVICE}) | Card: {gpu_name} ({vram_gb:.1f} GB VRAM) | CUDA {torch.version.cuda}")
    else:
        print(f"Thiết bị: CPU (Không có GPU CUDA hoặc PyTorch đang cài bản CPU-only)")
    print(f"Backbone: {BACKBONE_NAME}")

    # 1. Khởi tạo Run ID & Thư mục Run riêng biệt
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if RUN_TAG:
        run_id = f"{run_id}_{RUN_TAG}"
    run_dir = os.path.join(RUNS_ROOT, run_id) if SAVE_VERSIONED else ROOT_MODEL_DIR
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(SHARED_DIR, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Thư mục lưu phiên bản này: {run_dir}")
    print(f"Thư mục cache dùng chung: {SHARED_DIR}")

    # 2. Xử lý Cache & Dữ liệu
    if CLEAR_CACHE and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("-> Đã xóa sạch cache embedding (CLEAR_CACHE=True).")

    id_paths, id_labels, class_names = list_id_images(ID_DATASET_DIRS, EXCLUDE_CLASSES)
    ood_paths = list_ood_images(OOD_DATASET_DIRS)
    num_classes = len(class_names)
    labels_np = np.array(id_labels)

    print(f"\n[Dữ liệu] Tập ID: {len(id_paths)} ảnh | {num_classes} classes: {class_names}")
    counts = np.bincount(labels_np, minlength=num_classes)
    print(f"  Số ảnh mỗi class: {dict(zip(class_names, counts))}")
    print(f"[Dữ liệu] Tập OOD: {len(ood_paths)} ảnh")

    if PRUNE_CACHE and USE_EMBEDDING_CACHE and not CLEAR_CACHE:
        removed = _prune_cache(id_paths + ood_paths)
        if removed:
            print(f"-> Đã dọn dẹp {removed} file cache của ảnh/class không còn dùng.")

    # 3. Trích xuất Embeddings (với Cache)
    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)
    sig = _augment_signature(AUGMENT_CONFIG, AUGMENT_COPIES, BACKBONE_NAME)
    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    print("\n--- [Bước 1/4] Chuẩn bị Embedding cho Tập ID ---")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_id_embeddings(
        backbone, id_paths, augment_transform, clean_transform, AUGMENT_COPIES, sig
    )
    print(f"-> Hoàn tất chuẩn bị ID embedding sau {time.time()-t0:.1f}s")

    print("\n--- [Bước 2/4] Chuẩn bị Embedding cho Tập OOD ---")
    t0 = time.time()
    X_ood_all = precompute_ood_embeddings(backbone, ood_paths, clean_transform, sig)
    print(f"-> Hoàn tất chuẩn bị OOD embedding sau {time.time()-t0:.1f}s")

    # 4. K-Fold Cross Validation cho Classifier
    print(f"\n--- [Bước 3/4] K-Fold Cross Validation (K={K_FOLDS}) ---")
    fold_assignment = stratified_k_fold_indices(id_labels, K_FOLDS, seed=RANDOM_SEED)
    fold_accs, fold_losses = [], []
    oof_pred = [None] * len(id_paths)
    oof_probs = [None] * len(id_paths)
    oof_fold = [None] * len(id_paths)

    for fold in range(K_FOLDS):
        val_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]

        train_emb = torch.cat([augmented_emb_list[i] for i in train_idx], dim=0).to(DEVICE)
        train_lbl = torch.tensor([labels_np[i] for i in train_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long).to(DEVICE)

        val_emb = torch.cat([clean_emb_list[i] for i in val_idx], dim=0).to(DEVICE)
        val_lbl = torch.tensor([labels_np[i] for i in val_idx], dtype=torch.long).to(DEVICE)

        print(f"\n-- Fold {fold+1}/{K_FOLDS} ({len(train_idx)} ảnh train, {len(val_idx)} ảnh val) --")
        _, val_acc, val_loss, val_probs = train_head(
            train_emb, train_lbl, val_emb, val_lbl, embedding_dim, num_classes,
            HEAD_EPOCHS, HEAD_LR, HEAD_DROPOUT, EARLY_STOP_PATIENCE,
            print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"Fold {fold+1}/{K_FOLDS}"
        )
        fold_accs.append(val_acc)
        fold_losses.append(val_loss)

        for local_i, global_i in enumerate(val_idx):
            probs_row = val_probs[local_i]
            oof_pred[global_i] = int(probs_row.argmax().item())
            oof_probs[global_i] = probs_row.tolist()
            oof_fold[global_i] = fold

    fold_accs = np.array(fold_accs)
    print(f"\n>>> K-Fold Accuracy TB = {fold_accs.mean():.2%} (±{fold_accs.std():.2%})")
    print_detailed_report(id_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names, output_dir=run_dir)

    # 5. Huấn luyện Classifier Model Final
    print(f"\n--- Huấn luyện Final Classifier trên toàn bộ {len(id_paths)} ảnh ---")
    all_train_emb = torch.cat(augmented_emb_list, dim=0).to(DEVICE)
    all_train_lbl = torch.tensor([labels_np[i] for i in range(len(id_paths)) for _ in range(AUGMENT_COPIES)], dtype=torch.long).to(DEVICE)
    final_epochs = max(50, int(HEAD_EPOCHS * 0.5))

    final_head, _, _, _ = train_head(
        all_train_emb, all_train_lbl, None, None, embedding_dim, num_classes,
        final_epochs, HEAD_LR, HEAD_DROPOUT, patience=final_epochs,
        print_epochs=PRINT_EPOCH_DETAILS, fold_label="Final Model"
    )

    full_model = build_full_model(BACKBONE_NAME)
    full_model.classifier = final_head
    full_model = full_model.to(DEVICE).eval()

    # 6. Huấn luyện OOD Detector (Dùng trực tiếp clean embeddings)
    print("\n--- [Bước 4/4] Huấn luyện OOD Detector (Cosine & Mahalanobis) ---")
    X_id_clean = torch.cat(clean_emb_list, dim=0).numpy()
    have_ood = X_ood_all.shape[0] >= 2

    try:
        X_fit, X_val, y_fit, _ = train_test_split(X_id_clean, labels_np, test_size=ID_VAL_RATIO, random_state=RANDOM_SEED, stratify=labels_np)
    except ValueError:
        X_fit, X_val, y_fit, _ = train_test_split(X_id_clean, labels_np, test_size=ID_VAL_RATIO, random_state=RANDOM_SEED)

    if have_ood:
        X_ood_val, X_ood_test = train_test_split(X_ood_all, test_size=OOD_TEST_RATIO, random_state=RANDOM_SEED)
        print(f"  Tách tập: ID-fit={len(X_fit)} | ID-val={len(X_val)} | OOD-val={len(X_ood_val)} | OOD-test={len(X_ood_test)}")
    else:
        X_ood_val = X_ood_test = np.zeros((0, embedding_dim), np.float32)
        print(f"  Tách tập: ID-fit={len(X_fit)} | ID-val={len(X_val)} | (Không đủ ảnh OOD để test AUROC)")

    centroids = fit_centroids(X_fit, y_fit, num_classes)
    maha_means, maha_prec = fit_mahalanobis(X_fit, y_fit, num_classes)

    ood_methods = {
        "cosine": lambda X: score_cosine(X, centroids),
        "mahalanobis": lambda X: score_mahalanobis(X, maha_means, maha_prec),
    }

    ood_results = {}
    print(f"\n  {'Method':12s} {'AUROC':>8s} {'FPR@95TPR(val)':>16s} {'FPR@95TPR(test)':>16s}")
    for name, fn in ood_methods.items():
        s_id_val = fn(X_val)
        tau = threshold_at_tpr(s_id_val, TARGET_TPR)
        if have_ood:
            au = auroc(s_id_val, fn(X_ood_val))
            fpr_val = fpr_at_threshold(fn(X_ood_val), tau)
            fpr_test = fpr_at_threshold(fn(X_ood_test), tau)
        else:
            au = fpr_val = fpr_test = float("nan")
        ood_results[name] = {"tau": tau, "auroc": au, "fpr_val": fpr_val, "fpr_test": fpr_test}
        print(f"  {name:12s} {au:8.3f} {fpr_val:16.3f} {fpr_test:16.3f}")

    best_ood_method = max(ood_results, key=lambda k: ood_results[k]["auroc"]) if have_ood else "cosine"
    best_threshold = ood_results[best_ood_method]["tau"]
    cosine_tau = ood_results["cosine"]["tau"]
    print(f"\n  => Thuật toán OOD tối ưu: {best_ood_method} (Threshold = {best_threshold:.4f})")

    # 7. Đóng gói & Lưu trữ Artifacts (Model .pt, Payload .joblib, JSON)
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    model_pt_path = os.path.join(run_dir, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")
    model_onnx_path = os.path.join(run_dir, f"edge_classifier_fewshot_{BACKBONE_NAME}.onnx")
    ood_joblib_path = os.path.join(run_dir, "ood_detector.joblib")
    ood_npz_path = os.path.join(run_dir, "ood_detector.npz")
    ood_json_path = os.path.join(run_dir, "ood_detector.json")

    # A) Lưu PyTorch Model (.pt)
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "embedding_dim": embedding_dim,
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
        "run_id": run_id,
        "timestamp": timestamp_str,
        "num_classes": num_classes,
    }, model_pt_path)

    # B) Xuất Model Unified All-In-One ONNX (.onnx) (Tích hợp cả Phân loại + Bắt Dị vật)
    onnx_exported = False
    if EXPORT_ONNX:
        try:
            full_model_cpu = full_model.cpu().eval()
            unified_model_cpu = UnifiedClassifierOODModel(full_model_cpu, centroids, best_threshold).eval()
            dummy_input_cpu = torch.randn(1, 3, 224, 224)
            try:
                torch.onnx.export(
                    unified_model_cpu,
                    dummy_input_cpu,
                    model_onnx_path,
                    input_names=["input"],
                    output_names=["logits", "pred_class_id", "ood_score", "is_unknown"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                        "pred_class_id": {0: "batch_size"},
                        "ood_score": {0: "batch_size"},
                        "is_unknown": {0: "batch_size"},
                    },
                    opset_version=13,
                    do_constant_folding=True,
                    dynamo=False,
                )
            except TypeError:
                torch.onnx.export(
                    unified_model_cpu,
                    dummy_input_cpu,
                    model_onnx_path,
                    input_names=["input"],
                    output_names=["logits", "pred_class_id", "ood_score", "is_unknown"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                        "pred_class_id": {0: "batch_size"},
                        "ood_score": {0: "batch_size"},
                        "is_unknown": {0: "batch_size"},
                    },
                    opset_version=13,
                    do_constant_folding=True,
                )
            full_model.to(DEVICE)
            onnx_exported = True
            print(f"-> [ALL-IN-ONE ONNX] Đã xuất thành công: {model_onnx_path}")
        except Exception as e:
            full_model.to(DEVICE)
            print(f"  [!] Không thể xuất ONNX: {e}")

    # C) Lưu OOD Payload Joblib (.joblib)
    ood_payload = {
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "embedding_dim": int(embedding_dim),
        "classifier_ckpt": model_pt_path,
        "centroids": centroids,
        "ood_method": best_ood_method,
        "threshold": float(best_threshold),
        "cosine_threshold": float(cosine_tau),
        "mahalanobis": {"means": maha_means, "precision": maha_prec},
        "target_tpr": TARGET_TPR,
        "metrics": ood_results,
    }
    joblib.dump(ood_payload, ood_joblib_path)

    # D) Lưu OOD Payload dạng NPZ + JSON (Độc lập không cần thư viện sklearn / joblib, đọc được bằng C++, C#, Python)
    np.savez(
        ood_npz_path,
        centroids=centroids.astype(np.float32),
        maha_means=maha_means.astype(np.float32),
        maha_precision=maha_prec.astype(np.float32),
        threshold=np.float32(best_threshold),
        cosine_threshold=np.float32(cosine_tau),
        ood_method=np.array(best_ood_method),
        class_names=np.array(class_names),
        embedding_dim=np.int32(embedding_dim),
    )
    with open(ood_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "num_classes": num_classes,
            "backbone_name": BACKBONE_NAME,
            "embedding_dim": int(embedding_dim),
            "ood_method": best_ood_method,
            "threshold": float(best_threshold),
            "cosine_threshold": float(cosine_tau),
            "target_tpr": TARGET_TPR,
            "metrics": ood_results,
        }, f, ensure_ascii=False, indent=2)

    # E) Đo thông số Edge Latency & Size
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = full_model(dummy_input)
    avg_latency_ms = (time.time() - start) / n_runs * 1000
    model_size_mb = os.path.getsize(model_pt_path) / (1024 * 1024)

    # F) Lưu Run Info JSON
    run_info = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "timestamp": timestamp_str,
        "backbone_name": BACKBONE_NAME,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_id_images": len(id_paths),
        "num_ood_images": len(ood_paths),
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
        "ood_best_method": best_ood_method,
        "ood_threshold": float(best_threshold),
        "ood_auroc": float(ood_results[best_ood_method]["auroc"]),
        "ood_fpr_test": float(ood_results[best_ood_method]["fpr_test"]),
        "avg_latency_ms": float(avg_latency_ms),
        "model_size_mb": float(model_size_mb),
        "model_pt_file": os.path.abspath(model_pt_path),
        "model_onnx_file": os.path.abspath(model_onnx_path) if onnx_exported else "",
        "ood_joblib_file": os.path.abspath(ood_joblib_path),
        "ood_npz_file": os.path.abspath(ood_npz_path),
    }
    run_info_path = os.path.join(run_dir, "run_info.json")
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    # G) Ghi Index lịch sử
    _append_runs_index(os.path.join(RUNS_ROOT, "runs_index.csv"), run_info)

    # H) Cập nhật bản LATEST ngoài ROOT_MODEL_DIR
    if KEEP_LATEST_COPY and SAVE_VERSIONED:
        fixed_pt = os.path.join(ROOT_MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")
        fixed_onnx = os.path.join(ROOT_MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.onnx")
        fixed_joblib = os.path.join(ROOT_MODEL_DIR, "ood_detector.joblib")
        fixed_npz = os.path.join(ROOT_MODEL_DIR, "ood_detector.npz")
        fixed_json_ood = os.path.join(ROOT_MODEL_DIR, "ood_detector.json")
        fixed_report = os.path.join(ROOT_MODEL_DIR, "kfold_detailed_report.csv")
        fixed_json = os.path.join(ROOT_MODEL_DIR, "run_info.json")

        shutil.copy2(model_pt_path, fixed_pt)
        if onnx_exported and os.path.isfile(model_onnx_path):
            shutil.copy2(model_onnx_path, fixed_onnx)
        shutil.copy2(ood_joblib_path, fixed_joblib)
        shutil.copy2(ood_npz_path, fixed_npz)
        shutil.copy2(ood_json_path, fixed_json_ood)
        shutil.copy2(os.path.join(run_dir, "kfold_detailed_report.csv"), fixed_report)
        shutil.copy2(run_info_path, fixed_json)

        with open(os.path.join(ROOT_MODEL_DIR, "LATEST_RUN.txt"), "w", encoding="utf-8") as f:
            f.write(f"Run mới nhất: {run_id}\nTimestamp: {timestamp_str}\nThư mục gốc của run: {os.path.abspath(run_dir)}\n")
        print(f"\n-> Đã cập nhật đầy đủ các bản Latest (.pt, .onnx, .joblib, .npz, .json) tại: {ROOT_MODEL_DIR}")

    print("\n" + "=" * 90)
    print(" TỔNG KẾT HOÀN TẤT")
    print(f" Run ID          : {run_id}")
    print(f" Thư mục Run     : {run_dir}")
    print(f" PyTorch Model   : {model_pt_path}")
    print(f" OOD Detector    : {ood_joblib_path}")
    print(f" K-Fold Acc      : {fold_accs.mean():.2%} (±{fold_accs.std():.2%})")
    print(f" OOD Method      : {best_ood_method} (Threshold = {best_threshold:.4f})")
    if have_ood:
        print(f" OOD AUROC       : {ood_results[best_ood_method]['auroc']:.3f}")
        print(f" OOD FPR@95TPR   : {ood_results[best_ood_method]['fpr_test']:.3f}")
    print(f" Latency (1 ảnh) : {avg_latency_ms:.2f} ms | Dung lượng: {model_size_mb:.2f} MB")
    print("=" * 90)


if __name__ == "__main__":
    main()
