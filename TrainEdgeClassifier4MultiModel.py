r"""
KIẾN TRÚC CHUNG: Pretrained-CNN + Few-shot learning + K-Fold Cross Validation.
BẢN NÀY: train NHIỀU MODEL trong 1 lần chạy (mỗi PART một model riêng).

Toàn bộ ý tưởng và logic giữ NGUYÊN như bản gốc:
1. Backbone CNN đóng băng hoàn toàn, không fine-tune.
2. Extract embedding 1 lần, cache lại -> "train" chỉ là train classifier nhẹ.
3. K-Fold Cross Validation thay vì chia train/val cố định.
4. Sau khi đánh giá xong, train model "sản xuất" trên toàn bộ dữ liệu.

--------------------------------------------------------------------------
PHẦN THÊM VÀO: TRAIN NHIỀU MODEL
--------------------------------------------------------------------------
Khai báo danh sách JOB trong CONFIG, mỗi job = 1 model. Script chạy lần lượt
từng job, mỗi job có báo cáo K-Fold + CSV + file model riêng, cuối cùng in
bảng tổng hợp so sánh các job với nhau.

Backbone chỉ được nạp MỘT LẦN rồi dùng lại cho mọi job (backbone bị đóng
băng nên không có chuyện job này làm bẩn job kia).

Vì sao chạy TUẦN TỰ chứ không đa luồng: phần nặng nhất là extract embedding,
việc này đã chiếm trọn GPU rồi; chạy song song chỉ tranh nhau VRAM và dễ
OOM chứ không nhanh hơn.

Cách chạy:
    python few_shot_pipeline_multi.py
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==========================================================================
# CONFIG - SỬA Ở ĐÂY KHI ÁP DỤNG CHO BÀI TOÁN MỚI
# ==========================================================================

# Thư mục gốc chứa PART1..PART5, mỗi PART có train/ và val/, trong đó mỗi
# class là 1 thư mục con (OK/, NG/).
DATASET_ROOT = r"D:\TongHop\RTC Technologi\PCB\dataset14"

# Thư mục gốc để lưu model; mỗi job lưu vào thư mục con riêng theo tên job.
MODEL_ROOT = r"D:\TongHop\RTC Technologi\PCB\model\model18"

# Danh sách JOB - mỗi phần tử tạo ra 1 model riêng.
TEN_CAC_JOB = ["PART1", "PART2", "PART3", "PART4", "PART5"]

BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"

# Cấu hình augmentation - BẬT/TẮT theo đặc thù bài toán, không hard-code cứng
AUGMENT_CONFIG = {
    "horizontal_flip": True,     # tắt nếu ảnh có chữ/hướng quan trọng
    "rotation_degrees": 10,      # 0 để tắt xoay
    "color_jitter": True,        # tắt nếu bài toán phụ thuộc MÀU SẮC
    "random_resized_crop": False,  # tắt nếu bố cục/vị trí vật thể trong ảnh quan trọng
}

# Ghi đè cấu hình cho RIÊNG một số job (job nào không khai báo thì dùng mặc
# định ở trên). Ví dụ part3 toàn vỏ USB kim loại bóng, muốn tắt color_jitter:
#   CAU_HINH_RIENG = {"PART3": {"AUGMENT_CONFIG": {..., "color_jitter": False}}}
# Các khoá cho phép ghi đè: AUGMENT_CONFIG, AUGMENT_COPIES, K_FOLDS,
# HEAD_EPOCHS, HEAD_LR, HEAD_DROPOUT, EARLY_STOP_PATIENCE, BACKBONE_NAME.
CAU_HINH_RIENG = {}

AUGMENT_COPIES = 20   # số bản augmentation/ảnh khi extract embedding cho train
K_FOLDS = 5           # số fold cross-validation - tăng nếu dataset lớn hơn

HEAD_EPOCHS = 300
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 30
PRINT_EPOCH_DETAILS = True   # True = in chi tiết từng epoch trong mỗi fold

BO_QUA_JOB_LOI = True   # True = job nào lỗi thì báo rồi chạy tiếp job sau

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa khi đổi bài toán
# ==========================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def tao_danh_sach_job():
    """Dựng danh sách job từ CONFIG. Mỗi job là 1 dict đủ thông tin để chạy
    độc lập: dữ liệu ở đâu, lưu model ở đâu, tham số train là gì."""
    cac_job = []
    for ten in TEN_CAC_JOB:
        rieng = CAU_HINH_RIENG.get(ten, {})
        backbone = rieng.get("BACKBONE_NAME", BACKBONE_NAME)
        model_dir = os.path.join(MODEL_ROOT, ten.lower())
        cac_job.append({
            "ten": ten,
            "dataset_dirs": [os.path.join(DATASET_ROOT, ten, "train"),
                             os.path.join(DATASET_ROOT, ten, "val")],
            "model_dir": model_dir,
            "model_out": os.path.join(
                model_dir, f"edge_classifier_fewshot_{backbone}_{ten.lower()}.pt"),
            "backbone_name": backbone,
            "augment_config": rieng.get("AUGMENT_CONFIG", AUGMENT_CONFIG),
            "augment_copies": rieng.get("AUGMENT_COPIES", AUGMENT_COPIES),
            "k_folds": rieng.get("K_FOLDS", K_FOLDS),
            "head_epochs": rieng.get("HEAD_EPOCHS", HEAD_EPOCHS),
            "head_lr": rieng.get("HEAD_LR", HEAD_LR),
            "head_dropout": rieng.get("HEAD_DROPOUT", HEAD_DROPOUT),
            "patience": rieng.get("EARLY_STOP_PATIENCE", EARLY_STOP_PATIENCE),
        })
    return cac_job


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


def list_images(dataset_dirs):
    """Gộp ảnh từ nhiều thư mục (vd train/ + val/) thành 1 danh sách chung.
    Trả về: image_paths (list[str]), labels (list[int]), class_names (list[str])."""
    class_names = set()
    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {d}")
        for entry in os.scandir(d):
            if entry.is_dir():
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


@torch.no_grad()
def precompute_all_embeddings(backbone, image_paths, augment_transform, clean_transform, augment_copies):
    """
    Với MỖI ảnh gốc, tính sẵn:
      - augmented_emb[i]: tensor (augment_copies, dim) - dùng khi ảnh này rơi vào phần TRAIN của 1 fold
      - clean_emb[i]:      tensor (1, dim)               - dùng khi ảnh này rơi vào phần VAL của 1 fold
    Việc này chỉ làm 1 LẦN cho toàn bộ dataset, sau đó K-Fold chỉ việc
    "chọn lại" các embedding có sẵn theo từng fold - không phải extract lại.
    """
    augmented_emb, clean_emb = [], []

    for path in image_paths:
        image = Image.open(path).convert("RGB")

        aug_tensors = torch.stack([augment_transform(image) for _ in range(augment_copies)]).to(DEVICE)
        aug_emb = extract_embedding(backbone, aug_tensors)
        augmented_emb.append(aug_emb.cpu())

        clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
        c_emb = extract_embedding(backbone, clean_tensor)
        clean_emb.append(c_emb.cpu())

    return augmented_emb, clean_emb


def stratified_k_fold_indices(labels, k, seed=42):
    """Chia index thành k fold, đảm bảo tỉ lệ mỗi class đồng đều giữa các fold
    (stratified) - viết tay, không phụ thuộc thư viện ngoài."""
    labels = np.array(labels)
    rng = np.random.RandomState(seed)
    fold_assignment = np.zeros(len(labels), dtype=int)

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        # Chia đều idx_c thành k phần, phân bổ lần lượt fold 0,1,2,...,k-1
        for i, idx in enumerate(idx_c):
            fold_assignment[idx] = i % k

    return fold_assignment


def make_head(embedding_dim, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)).to(DEVICE)


def train_head(train_emb, train_labels, val_emb, val_labels, embedding_dim, num_classes,
                epochs, lr, dropout, patience, print_epochs=False, fold_label=""):
    """Train 1 classifier head trên embedding đã cache. Dùng chung cho cả
    K-Fold (train/val nội bộ mỗi fold) lẫn train model sản xuất cuối cùng.
    Trả về thêm val_probs (xác suất từng class cho từng ảnh val) để phục vụ
    báo cáo chi tiết. Nếu print_epochs=True, in ra tiến trình từng epoch."""
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


def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names,
                           model_dir, ten_job=""):
    import csv

    num_classes = len(class_names)
    n = len(image_paths)

    # --- Confusion matrix (hàng = thật, cột = dự đoán) ---
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(n):
        confusion[labels_np[i], oof_pred[i]] += 1

    print("\n" + "=" * 100)
    print(f"BÁO CÁO CHI TIẾT{' - ' + ten_job if ten_job else ''} "
          f"(out-of-fold - đánh giá trên TOÀN BỘ dữ liệu, không thiên vị)")
    print("=" * 100)

    print("\nConfusion matrix (hàng = thật, cột = dự đoán):")
    header = "".ljust(15) + "".join(f"{c[:10]:>12s}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = row_name.ljust(15) + "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes))
        print(row_str)

    # --- Accuracy + confidence trung bình theo từng class ---
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

    # --- Bảng chi tiết từng ảnh ---
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

    # --- Xuất CSV để mở bằng Excel ---
    csv_path = os.path.join(model_dir, "kfold_detailed_report.csv")
    os.makedirs(model_dir, exist_ok=True)
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

    return overall_acc


# ==========================================================================
# CHẠY 1 JOB = TRAIN 1 MODEL (nội dung y hệt main() của bản gốc)
# ==========================================================================
def chay_mot_job(job, backbone_cache):
    ten = job["ten"]
    backbone_name = job["backbone_name"]

    print("\n" + "#" * 100)
    print(f"# JOB: {ten}  |  backbone: {backbone_name}")
    print(f"# Dữ liệu: {job['dataset_dirs']}")
    print(f"# Model ra: {job['model_out']}")
    print("#" * 100)

    image_paths, labels, class_names = list_images(job["dataset_dirs"])
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class: {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")

    augment_transform, clean_transform = build_transforms(job["augment_config"])

    # Backbone nạp 1 lần rồi dùng lại cho mọi job (nó bị đóng băng nên an toàn)
    if backbone_name not in backbone_cache:
        print(f"Đang nạp backbone {backbone_name}...")
        backbone_cache[backbone_name] = build_backbone(backbone_name)
    backbone, embedding_dim = backbone_cache[backbone_name]
    print(f"Embedding dimension: {embedding_dim}")

    print(f"\nĐang extract embedding cho toàn bộ {len(image_paths)} ảnh "
          f"({job['augment_copies']} bản augment/ảnh, chỉ chạy 1 lần)...")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, image_paths, augment_transform, clean_transform, job["augment_copies"]
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION - đánh giá độ ổn định thật sự của model
    # --------------------------------------------------------------------
    k_folds = job["k_folds"]
    fold_assignment = stratified_k_fold_indices(labels, k_folds, seed=RANDOM_SEED)
    labels_np = np.array(labels)

    print(f"\n=== K-Fold Cross Validation (K={k_folds}) - {ten} ===")
    fold_accs, fold_losses = [], []

    oof_pred = [None] * len(image_paths)
    oof_probs = [None] * len(image_paths)
    oof_fold = [None] * len(image_paths)

    for fold in range(k_folds):
        val_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]

        train_emb = torch.cat([augmented_emb_list[i] for i in train_idx], dim=0).to(DEVICE)
        train_lbl = torch.tensor(
            [labels_np[i] for i in train_idx for _ in range(job["augment_copies"])], dtype=torch.long
        ).to(DEVICE)

        val_emb = torch.cat([clean_emb_list[i] for i in val_idx], dim=0).to(DEVICE)
        val_lbl = torch.tensor([labels_np[i] for i in val_idx], dtype=torch.long).to(DEVICE)

        print(f"\n--- {ten} | Fold {fold+1}/{k_folds} "
              f"({len(train_idx)} ảnh train, {len(val_idx)} ảnh val) ---")
        _, val_acc, val_loss, val_probs = train_head(
            train_emb, train_lbl, val_emb, val_lbl, embedding_dim, num_classes,
            job["head_epochs"], job["head_lr"], job["head_dropout"], job["patience"],
            print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"{ten} Fold {fold+1}/{k_folds}"
        )

        fold_accs.append(val_acc)
        fold_losses.append(val_loss)

        for local_i, global_i in enumerate(val_idx):
            probs_row = val_probs[local_i]
            oof_pred[global_i] = int(probs_row.argmax().item())
            oof_probs[global_i] = probs_row.tolist()
            oof_fold[global_i] = fold

        print(f"{ten} | Fold {fold+1}/{k_folds}: {len(train_idx)} ảnh train, "
              f"{len(val_idx)} ảnh val -> Val acc: {val_acc:.2%} | Val loss: {val_loss:.4f}")

    fold_accs = np.array(fold_accs)
    print(f"\n>>> [{ten}] Kết quả K-Fold: Accuracy trung bình = {fold_accs.mean():.2%} "
          f"(độ lệch chuẩn ±{fold_accs.std():.2%})")

    # --------------------------------------------------------------------
    # BÁO CÁO CHI TIẾT - tổng hợp từ kết quả out-of-fold của TOÀN BỘ ảnh
    # --------------------------------------------------------------------
    overall_acc = print_detailed_report(image_paths, labels_np, oof_pred, oof_probs,
                                        oof_fold, class_names, job["model_dir"], ten)

    # --------------------------------------------------------------------
    # Train model SẢN XUẤT cuối cùng trên TOÀN BỘ dữ liệu (không giữ val)
    # --------------------------------------------------------------------
    print(f"\n=== [{ten}] Train model cuối cùng trên toàn bộ {len(image_paths)} ảnh (để deploy) ===")
    all_train_emb = torch.cat(augmented_emb_list, dim=0).to(DEVICE)
    all_train_lbl = torch.tensor(
        [labels_np[i] for i in range(len(image_paths)) for _ in range(job["augment_copies"])],
        dtype=torch.long
    ).to(DEVICE)

    final_epochs = max(50, int(job["head_epochs"] * 0.5))

    final_head, _, _, _ = train_head(
        all_train_emb, all_train_lbl, None, None, embedding_dim, num_classes,
        final_epochs, job["head_lr"], job["head_dropout"], patience=final_epochs,
        print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"{ten} Model cuối cùng"
    )

    full_model = build_full_model(backbone_name)
    full_model.classifier = final_head
    full_model = full_model.to(DEVICE)
    full_model.eval()

    os.makedirs(job["model_dir"], exist_ok=True)
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
        "backbone_name": backbone_name,
        "job_name": ten,
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
    }, job["model_out"])
    print(f"Đã lưu model cuối cùng tại: {job['model_out']}")

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
    model_size_mb = os.path.getsize(job["model_out"]) / (1024 * 1024)

    print(f"\n--- [{ten}] Thông số edge ---")
    print(f"Backbone: {backbone_name}")
    print(f"Latency trung bình (1 ảnh): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")
    print(f"K-Fold accuracy (độ tin cậy thật): {fold_accs.mean():.2%} ± {fold_accs.std():.2%}")

    return {
        "ten": ten,
        "so_anh": len(image_paths),
        "class_names": class_names,
        "so_anh_moi_class": dict(zip(class_names, counts.tolist())),
        "kfold_mean": float(fold_accs.mean()),
        "kfold_std": float(fold_accs.std()),
        "oof_acc": float(overall_acc),
        "latency_ms": avg_latency_ms,
        "model_out": job["model_out"],
    }


# ==========================================================================
# MAIN - chạy lần lượt tất cả các job
# ==========================================================================
def main():
    print(f"Đang chạy trên: {DEVICE}")
    cac_job = tao_danh_sach_job()
    print(f"Số model sẽ train: {len(cac_job)} ({', '.join(j['ten'] for j in cac_job)})")

    backbone_cache = {}
    ket_qua, that_bai = [], []
    t_tong = time.time()

    for i, job in enumerate(cac_job, start=1):
        t0 = time.time()
        print(f"\n\n>>>>>> [{i}/{len(cac_job)}] Bắt đầu job {job['ten']} <<<<<<")
        try:
            kq = chay_mot_job(job, backbone_cache)
            kq["thoi_gian_s"] = time.time() - t0
            ket_qua.append(kq)
            print(f">>>>>> Xong job {job['ten']} sau {kq['thoi_gian_s']:.1f}s <<<<<<")
        except Exception as e:
            that_bai.append((job["ten"], repr(e)))
            print(f"!!!!!! Job {job['ten']} LỖI: {e}")
            if not BO_QUA_JOB_LOI:
                raise

    # ------------------------------------------------------------------
    # BẢNG TỔNG HỢP - so sánh các model với nhau
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("TỔNG HỢP TOÀN BỘ JOB")
    print("=" * 100)
    print(f"{'Job':<10}{'Số ảnh':>8}{'Phân bố class':>28}{'K-Fold acc':>16}"
          f"{'OOF acc':>10}{'Latency':>11}{'Thời gian':>11}")
    print("-" * 100)
    for kq in ket_qua:
        phan_bo = ", ".join(f"{k}:{v}" for k, v in kq["so_anh_moi_class"].items())
        print(f"{kq['ten']:<10}{kq['so_anh']:>8}{phan_bo:>28}"
              f"{kq['kfold_mean']:>11.2%}±{kq['kfold_std']:<4.1%}"
              f"{kq['oof_acc']:>10.2%}{kq['latency_ms']:>9.1f}ms{kq['thoi_gian_s']:>10.1f}s")
    print("-" * 100)

    if ket_qua:
        tb = np.mean([k["kfold_mean"] for k in ket_qua])
        print(f"Accuracy K-Fold trung bình các job: {tb:.2%}")
        yeu = [k for k in ket_qua if k["kfold_mean"] < tb - 0.05]
        if yeu:
            mo_ta = ", ".join(f"{k['ten']} ({k['kfold_mean']:.1%})" for k in yeu)
            print(f"Job kém hơn hẳn mặt bằng chung: {mo_ta}")

    print(f"\nTổng thời gian: {time.time()-t_tong:.1f}s | "
          f"Thành công {len(ket_qua)}/{len(cac_job)} job")
    if that_bai:
        print(f"\nCác job LỖI ({len(that_bai)}):")
        for ten, loi in that_bai:
            print(f"  - {ten}: {loi}")


if __name__ == "__main__":
    main()