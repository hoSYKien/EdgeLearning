"""
KIẾN TRÚC CHUNG: Pretrained-CNN + Few-shot learning + K-Fold Cross Validation.
+ CACHE EMBEDDING RA ĐĨA -> thêm/bớt class, thêm/bớt ảnh mà KHÔNG extract lại
  embedding cho những ảnh không đổi.

--------------------------------------------------------------------------
Ý TƯỞNG CỐT LÕI (GIỮ NGUYÊN so với bản gốc):
--------------------------------------------------------------------------
1. Backbone CNN đóng băng hoàn toàn, không fine-tune.
2. Extract embedding cho mỗi ảnh -> "train" chỉ là train 1 head nhẹ.
3. K-Fold Cross Validation để đánh giá không thiên vị.
4. Train model sản xuất cuối cùng trên toàn bộ dữ liệu để deploy.

--------------------------------------------------------------------------
PHẦN THÊM MỚI: CACHE EMBEDDING
--------------------------------------------------------------------------
Extract embedding (chạy ảnh qua CNN) là bước ĐẮT NHẤT. Train head thì rất
nhanh. Nên embedding của mỗi ảnh được LƯU RA ĐĨA (CACHE_DIR). Mỗi lần chạy:
  - Ảnh đã có trong cache & file không đổi  -> DÙNG LẠI, không extract.
  - Ảnh mới / ảnh vừa sửa                   -> extract rồi lưu cache.
  - Ảnh/class đã bị xoá khỏi dataset        -> cache tương ứng bị dọn đi.
Sau đó head được train lại trên tập embedding đã cập nhật (nhanh).

=> BA THAO TÁC BẠN CẦN, chỉ việc SỬA THƯ MỤC DATASET rồi CHẠY LẠI file này:

  1. THÊM CLASS MỚI:
     - Tạo thư mục class mới trong dataset, bỏ ảnh vào.
     - Chạy lại. Chỉ ảnh class mới được extract; các class cũ dùng cache.

  2. THÊM/BỚT ẢNH TRONG CLASS CŨ:
     - Copy thêm ảnh mới vào / xoá bớt ảnh cũ trong thư mục class đó.
     - Chạy lại. Chỉ ảnh mới được extract; ảnh bị xoá thì cache tự dọn.

  3. XOÁ HẲN 1 CLASS:
     - Xoá thư mục class đó, HOẶC thêm tên nó vào EXCLUDE_CLASSES bên dưới.
     - Chạy lại. Class biến mất khỏi model, cache của nó được dọn.

LƯU Ý QUAN TRỌNG về nhãn (label): class_names được sort lại mỗi lần chạy, nên
thêm/xoá class có thể làm chỉ số nhãn dịch chuyển. Điều này KHÔNG gây lỗi vì
cache được đánh dấu theo ĐƯỜNG DẪN ẢNH (không theo nhãn), và head luôn train
lại từ đầu cho đúng bộ class hiện tại. File model lưu kèm class_names nên mọi
script inference/heatmap đọc đúng tên class tự động.

LƯU Ý về augmentation: một khi ảnh đã được cache, các bản augment của nó bị
"đóng băng" (dùng lại đúng các bản đã sinh lần đầu) cho tới khi ảnh đổi hoặc
bạn xoá cache. Đây là đánh đổi để có tính tái lập + tốc độ. Muốn sinh lại
augment mới cho toàn bộ -> đặt CLEAR_CACHE = True một lần.

Cách chạy:
    python few_shot_pipeline.py
"""

import os
import time
import random
import hashlib
import json
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==========================================================================
# CONFIG - SỬA Ở ĐÂY KHI ÁP DỤNG CHO BÀI TOÁN MỚI
# ==========================================================================

DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts1\train",
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts1\val",
]

# Class muốn LOẠI khỏi lần train này (cách nhanh để "xoá class" mà không cần
# xoá thư mục). Vd: {"Part7", "Part12"}. Để trống = dùng hết.
EXCLUDE_CLASSES = set()

MODEL_DIR = r"D:\TongHop\RTC Technologi\G8\model\model3"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")

# --- CACHE EMBEDDING ---
CACHE_DIR = os.path.join(MODEL_DIR, "emb_cache")   # nơi lưu embedding từng ảnh
USE_EMBEDDING_CACHE = True     # False = luôn extract lại toàn bộ (như bản gốc)
CLEAR_CACHE = False            # True = xoá sạch cache rồi extract lại từ đầu
PRUNE_CACHE = True             # True = dọn cache của ảnh không còn trong dataset

AUGMENT_CONFIG = {
    "horizontal_flip": True,
    "rotation_degrees": 30,
    "color_jitter": True,
    "random_resized_crop": True,
}

AUGMENT_COPIES = 20
K_FOLDS = 5

HEAD_EPOCHS = 400
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 50
PRINT_EPOCH_DETAILS = True

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG
# ==========================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def build_transforms(augment_cfg):
    aug_ops = [transforms.Resize((224, 224))]
    if augment_cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(augment_cfg["rotation_degrees"]))
    if augment_cfg.get("random_resized_crop"):
        aug_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    if augment_cfg.get("color_jitter"):
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    aug_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    clean_ops = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(aug_ops), transforms.Compose(clean_ops)


def list_images(dataset_dirs, exclude_classes=frozenset()):
    """Gộp ảnh từ nhiều thư mục thành 1 danh sách chung, bỏ qua class bị loại.
    Trả về: image_paths, labels, class_names."""
    class_names = set()
    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {d}")
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name not in exclude_classes:
                class_names.add(entry.name)
    class_names = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    image_paths, labels = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for d in dataset_dirs:
        for cls in class_names:
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

    return image_paths, labels, class_names


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

    backbone = backbone.to(DEVICE)
    backbone.eval()
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


# ==========================================================================
# CACHE EMBEDDING — phần thêm mới
# ==========================================================================
def _augment_signature(augment_cfg, augment_copies, backbone_name):
    """Chữ ký cấu hình. Đổi augment/backbone/số copies -> chữ ký đổi -> cache
    cũ không còn hợp lệ (sẽ extract lại), tránh trộn embedding khác cấu hình."""
    payload = json.dumps({"cfg": augment_cfg, "copies": augment_copies,
                          "backbone": backbone_name}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _image_key(path):
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


def _cache_path(path, sig):
    return os.path.join(CACHE_DIR, f"{_image_key(path)}_{sig}.npz")


def _load_cached(path, sig):
    """Trả (aug_emb tensor, clean_emb tensor) nếu cache hợp lệ, ngược lại None.
    Hợp lệ = tồn tại + đúng chữ ký + file ảnh chưa đổi (mtime & size)."""
    cp = _cache_path(path, sig)
    if not os.path.isfile(cp):
        return None
    try:
        data = np.load(cp, allow_pickle=False)
        if float(data["mtime"]) != os.path.getmtime(path):
            return None
        if int(data["size"]) != os.path.getsize(path):
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


def _prune_cache(image_paths):
    """Xoá file cache của những ảnh KHÔNG còn trong dataset hiện tại
    (đã xoá ảnh / đã xoá hoặc loại class). Trả về số file đã dọn."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    current_keys = {_image_key(p) for p in image_paths}
    removed = 0
    for cp in glob.glob(os.path.join(CACHE_DIR, "*.npz")):
        name = os.path.basename(cp)
        key = name.split("_")[0]          # md5 (32 hex) không chứa '_'
        if key not in current_keys:
            try:
                os.remove(cp)
                removed += 1
            except OSError:
                pass
    return removed


def precompute_all_embeddings(backbone, image_paths, augment_transform,
                              clean_transform, augment_copies, sig):
    """Như bản gốc nhưng CÓ CACHE: ảnh nào đã cache & chưa đổi thì dùng lại,
    ảnh mới/đổi thì extract rồi lưu. Trả (augmented_emb_list, clean_emb_list)."""
    augmented_emb, clean_emb = [], []
    n_hit, n_miss = 0, 0

    for path in image_paths:
        cached = None
        if USE_EMBEDDING_CACHE:
            cached = _load_cached(path, sig)

        if cached is not None:
            aug_emb, c_emb = cached
            n_hit += 1
        else:
            image = Image.open(path).convert("RGB")
            aug_tensors = torch.stack(
                [augment_transform(image) for _ in range(augment_copies)]
            ).to(DEVICE)
            aug_emb = extract_embedding(backbone, aug_tensors).cpu()

            clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
            c_emb = extract_embedding(backbone, clean_tensor).cpu()

            if USE_EMBEDDING_CACHE:
                _save_cache(path, sig, aug_emb, c_emb)
            n_miss += 1

        augmented_emb.append(aug_emb)
        clean_emb.append(c_emb)

    print(f"  Cache: dùng lại {n_hit} ảnh | extract mới {n_miss} ảnh")
    return augmented_emb, clean_emb


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

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    has_val = val_emb is not None and val_emb.shape[0] > 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_probs = None
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    epochs_no_improve = 0

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
            if val_loss < best_val_loss:
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


def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names):
    import csv

    num_classes = len(class_names)
    n = len(image_paths)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(n):
        confusion[labels_np[i], oof_pred[i]] += 1

    print("\n" + "=" * 100)
    print("BÁO CÁO CHI TIẾT (out-of-fold - đánh giá trên TOÀN BỘ dữ liệu, không thiên vị)")
    print("=" * 100)

    print("\nConfusion matrix (hàng = thật, cột = dự đoán):")
    header = "".ljust(15) + "".join(f"{c[:10]:>12s}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = row_name.ljust(15) + "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes))
        print(row_str)

    print("\nAccuracy & Confidence trung bình theo từng class:")
    print(f"{'Class':<15}{'Số ảnh':>8}{'Đúng':>8}{'Accuracy':>12}{'Conf TB (khi đúng)':>22}{'Conf TB (khi sai)':>20}")
    print("-" * 90)
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
              f"{avg_conf_correct:>21.2%} {avg_conf_wrong:>19.2%}")

    overall_acc = sum(1 for i in range(n) if oof_pred[i] == labels_np[i]) / n
    print("-" * 90)
    print(f"{'TỔNG CỘNG':<15}{n:>8}{sum(1 for i in range(n) if oof_pred[i]==labels_np[i]):>8}{overall_acc:>11.2%}")

    print("\n" + "-" * 100)
    print("Chi tiết từng ảnh (out-of-fold prediction):")
    col_width = max(10, max(len(c) for c in class_names) + 2)
    header2 = f"{'File':<28}{'Fold':<6}{'Thật':<14}{'Đoán':<14}{'Đúng/Sai':<10}"
    for c in class_names:
        header2 += f"{c:>{col_width}}"
    print(header2)
    print("-" * len(header2))

    for i in range(n):
        fname = os.path.basename(image_paths[i])
        true_cls = class_names[labels_np[i]]
        pred_cls = class_names[oof_pred[i]]
        status = "Đúng" if oof_pred[i] == labels_np[i] else "SAI"
        row = f"{fname:<28}{oof_fold[i]+1:<6}{true_cls:<14}{pred_cls:<14}{status:<10}"
        for c_idx in range(num_classes):
            row += f"{oof_probs[i][c_idx]:>{col_width-1}.1%} "
        print(row)

    csv_path = os.path.join(MODEL_DIR, "kfold_detailed_report.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Fold", "Class_that", "Class_doan", "Dung_Sai"] +
                         [f"Xac_suat_{c}" for c in class_names])
        for i in range(n):
            fname = os.path.basename(image_paths[i])
            true_cls = class_names[labels_np[i]]
            pred_cls = class_names[oof_pred[i]]
            status = "Dung" if oof_pred[i] == labels_np[i] else "Sai"
            row = [fname, oof_fold[i] + 1, true_cls, pred_cls, status]
            row += [f"{p:.4f}" for p in oof_probs[i]]
            writer.writerow(row)
    print(f"\nĐã xuất báo cáo chi tiết ra CSV: {csv_path}")


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================
def main():
    print(f"Đang chạy trên: {DEVICE}")
    print(f"Backbone: {BACKBONE_NAME}")

    if CLEAR_CACHE and os.path.isdir(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Đã xoá sạch cache embedding (CLEAR_CACHE=True).")

    if EXCLUDE_CLASSES:
        print(f"Loại các class: {sorted(EXCLUDE_CLASSES)}")

    image_paths, labels, class_names = list_images(DATASET_DIRS, EXCLUDE_CLASSES)
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class ({num_classes}): {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")

    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)
    sig = _augment_signature(AUGMENT_CONFIG, AUGMENT_COPIES, BACKBONE_NAME)

    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    if PRUNE_CACHE and USE_EMBEDDING_CACHE and not CLEAR_CACHE:
        removed = _prune_cache(image_paths)
        if removed:
            print(f"Đã dọn {removed} file cache của ảnh/class không còn dùng.")

    print(f"\nĐang chuẩn bị embedding cho {len(image_paths)} ảnh "
          f"({AUGMENT_COPIES} bản augment/ảnh)...")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, image_paths, augment_transform, clean_transform, AUGMENT_COPIES, sig
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # --------------------------------------------------------------------
    fold_assignment = stratified_k_fold_indices(labels, K_FOLDS, seed=RANDOM_SEED)
    labels_np = np.array(labels)

    print(f"\n=== K-Fold Cross Validation (K={K_FOLDS}) ===")
    fold_accs, fold_losses = [], []

    oof_pred = [None] * len(image_paths)
    oof_probs = [None] * len(image_paths)
    oof_fold = [None] * len(image_paths)

    for fold in range(K_FOLDS):
        val_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]

        train_emb = torch.cat([augmented_emb_list[i] for i in train_idx], dim=0).to(DEVICE)
        train_lbl = torch.tensor(
            [labels_np[i] for i in train_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long
        ).to(DEVICE)

        val_emb = torch.cat([clean_emb_list[i] for i in val_idx], dim=0).to(DEVICE)
        val_lbl = torch.tensor([labels_np[i] for i in val_idx], dtype=torch.long).to(DEVICE)

        print(f"\n--- Fold {fold+1}/{K_FOLDS} ({len(train_idx)} ảnh train, {len(val_idx)} ảnh val) ---")
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

        print(f"Fold {fold+1}/{K_FOLDS}: {len(train_idx)} ảnh train, {len(val_idx)} ảnh val "
              f"-> Val acc: {val_acc:.2%} | Val loss: {val_loss:.4f}")

    fold_accs = np.array(fold_accs)
    print(f"\n>>> Kết quả K-Fold: Accuracy trung bình = {fold_accs.mean():.2%} "
          f"(độ lệch chuẩn ±{fold_accs.std():.2%})")

    print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names)

    # --------------------------------------------------------------------
    # Train model SẢN XUẤT cuối cùng trên TOÀN BỘ dữ liệu
    # --------------------------------------------------------------------
    print(f"\n=== Train model cuối cùng trên toàn bộ {len(image_paths)} ảnh (để deploy) ===")
    all_train_emb = torch.cat(augmented_emb_list, dim=0).to(DEVICE)
    all_train_lbl = torch.tensor(
        [labels_np[i] for i in range(len(image_paths)) for _ in range(AUGMENT_COPIES)], dtype=torch.long
    ).to(DEVICE)

    final_epochs = max(50, int(HEAD_EPOCHS * 0.5))

    final_head, _, _, _ = train_head(
        all_train_emb, all_train_lbl, None, None, embedding_dim, num_classes,
        final_epochs, HEAD_LR, HEAD_DROPOUT, patience=final_epochs,
        print_epochs=PRINT_EPOCH_DETAILS, fold_label="Model cuối cùng"
    )

    full_model = build_full_model(BACKBONE_NAME)
    full_model.classifier = final_head
    full_model = full_model.to(DEVICE)
    full_model.eval()

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
    }, MODEL_OUT)
    print(f"Đã lưu model cuối cùng tại: {MODEL_OUT}")

    # --------------------------------------------------------------------
    # Đo thông số edge
    # --------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = full_model(dummy_input)
    avg_latency_ms = (time.time() - start) / n_runs * 1000
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)

    print(f"\n--- Thông số edge ---")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Latency trung bình (1 ảnh): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")
    print(f"K-Fold accuracy (độ tin cậy thật): {fold_accs.mean():.2%} ± {fold_accs.std():.2%}")


if __name__ == "__main__":
    main()