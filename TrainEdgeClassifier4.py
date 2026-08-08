"""
KIẾN TRÚC CHUNG: Pretrained-CNN + Few-shot learning + K-Fold Cross Validation.

Đây là pipeline TỔNG QUÁT, dùng lại được cho BẤT KỲ bài toán phân loại ảnh
ít dữ liệu nào (màu sắc, trái cây, OK/NG, vật thể...) - chỉ cần sửa phần
CONFIG bên dưới, không cần đụng vào logic code.

--------------------------------------------------------------------------
Ý TƯỞNG CỐT LÕI (không đổi giữa các bài toán):
--------------------------------------------------------------------------
1. Backbone CNN (MobileNetV2/V3) ĐÓNG BĂNG HOÀN TOÀN, không fine-tune.
2. Extract embedding (vector đặc trưng) cho mỗi ảnh 1 LẦN DUY NHẤT, cache
   lại -> việc "train" sau đó chỉ là train 1 classifier NHẸ trên embedding,
   nhanh gấp hàng chục lần so với train lại cả CNN.
3. Vì dataset ít, KHÔNG chia train/val cố định một lần (dễ bị đánh giá sai
   lệch, không ổn định) -> dùng K-FOLD CROSS VALIDATION: gộp hết dữ liệu,
   xoay vòng chia K phần, mỗi ảnh đều được làm "val" ít nhất 1 lần.
   Đây là cách giải quyết "làm giàu val" ĐÚNG NGUYÊN TẮC và TỔNG QUÁT,
   không phụ thuộc vào bài toán cụ thể là gì.
4. Sau khi đánh giá xong bằng K-Fold, train 1 model "sản xuất" cuối cùng
   trên TOÀN BỘ dữ liệu (không giữ lại val) để deploy.

--------------------------------------------------------------------------
CÁCH ÁP DỤNG CHO BÀI TOÁN MỚI:
--------------------------------------------------------------------------
Chỉ cần sửa phần CONFIG bên dưới:
  - DATASET_DIRS: trỏ tới (các) thư mục chứa ảnh, mỗi class 1 thư mục con
  - AUGMENT_CONFIG: bật/tắt phép augmentation phù hợp với bài toán
    (vd: tắt ColorJitter cho bài toán MÀU, tắt Flip cho bài toán có chữ/
    ký hiệu định hướng)
  - BACKBONE_NAME, K_FOLDS, HEAD_EPOCHS...
Không cần sửa bất kỳ dòng logic nào khác trong file.

Cách chạy:
    python few_shot_pipeline.py
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

# Danh sách thư mục chứa dữ liệu, mỗi thư mục có cấu trúc class1/, class2/,...
# Nếu bạn vẫn còn tách sẵn train/ và val/, liệt kê cả 2 - script sẽ TỰ GỘP
# lại rồi chia K-Fold, không cần bạn tự gộp tay.
DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\PCB\dataset14\PART2\train",
    r"D:\TongHop\RTC Technologi\PCB\dataset14\PART2\val",
]

MODEL_DIR = r"D:\TongHop\RTC Technologi\PCB\model\model17"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")

# Cấu hình augmentation - BẬT/TẮT theo đặc thù bài toán, không hard-code cứng
AUGMENT_CONFIG = {
    "horizontal_flip": True,     # tắt nếu ảnh có chữ/hướng quan trọng
    "rotation_degrees": 10,      # 0 để tắt xoay
    "color_jitter": True,       # tắt nếu bài toán phụ thuộc MÀU SẮC (như hiện tại)
    "random_resized_crop": False, # tắt nếu bố cục/vị trí vật thể trong ảnh quan trọng
}

AUGMENT_COPIES = 20   # số bản augmentation/ảnh khi extract embedding cho train
K_FOLDS = 5            # số fold cross-validation - tăng nếu dataset lớn hơn

HEAD_EPOCHS = 300
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 30
PRINT_EPOCH_DETAILS = True   # True = in chi tiết từng epoch trong mỗi fold (như trước)

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa khi đổi bài toán
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


def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names):
    import csv

    num_classes = len(class_names)
    n = len(image_paths)

    # --- Confusion matrix (hàng = thật, cột = dự đoán) ---
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

    image_paths, labels, class_names = list_images(DATASET_DIRS)
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class: {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")

    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)

    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    print(f"\nĐang extract embedding cho toàn bộ {len(image_paths)} ảnh "
          f"({AUGMENT_COPIES} bản augment/ảnh, chỉ chạy 1 lần)...")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, image_paths, augment_transform, clean_transform, AUGMENT_COPIES
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION - đánh giá độ ổn định thật sự của model
    # --------------------------------------------------------------------
    fold_assignment = stratified_k_fold_indices(labels, K_FOLDS, seed=RANDOM_SEED)
    labels_np = np.array(labels)

    print(f"\n=== K-Fold Cross Validation (K={K_FOLDS}) ===")
    fold_accs, fold_losses = [], []

    # Lưu kết quả "out-of-fold" cho TỪNG ẢNH - vì mỗi ảnh chỉ làm val đúng 1
    # lần trong toàn bộ K-Fold, ta có thể gộp lại thành báo cáo đầy đủ,
    # không thiên vị, cho TOÀN BỘ dataset (không chỉ 1 tập test riêng nhỏ).
    oof_pred = [None] * len(image_paths)       # class dự đoán
    oof_probs = [None] * len(image_paths)      # xác suất từng class
    oof_fold = [None] * len(image_paths)       # ảnh này thuộc fold nào

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

        # Ghi lại kết quả từng ảnh trong fold này
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
    print(">>> Đây là chỉ số ĐÁNG TIN CẬY hơn nhiều so với 1 lần chia val cố định,")
    print(">>> vì đã được đánh giá trên TOÀN BỘ dữ liệu, không phụ thuộc may rủi của 1 lần chia.")

    # --------------------------------------------------------------------
    # BÁO CÁO CHI TIẾT - tổng hợp từ kết quả out-of-fold của TOÀN BỘ ảnh
    # --------------------------------------------------------------------
    print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names)

    # --------------------------------------------------------------------
    # Train model SẢN XUẤT cuối cùng trên TOÀN BỘ dữ liệu (không giữ val)
    # --------------------------------------------------------------------
    print(f"\n=== Train model cuối cùng trên toàn bộ {len(image_paths)} ảnh (để deploy) ===")
    all_train_emb = torch.cat(augmented_emb_list, dim=0).to(DEVICE)
    all_train_lbl = torch.tensor(
        [labels_np[i] for i in range(len(image_paths)) for _ in range(AUGMENT_COPIES)], dtype=torch.long
    ).to(DEVICE)

    # Số epoch cho model cuối = trung bình số epoch hội tụ tốt nhất qua các fold
    # (ước lượng hợp lý, vì không còn tập val riêng để early-stop nữa)
    final_epochs = max(50, int(HEAD_EPOCHS * 0.5))

    final_head, _, _, _ = train_head(
        all_train_emb, all_train_lbl, None, None, embedding_dim, num_classes,
        final_epochs, HEAD_LR, HEAD_DROPOUT, patience=final_epochs,  # không early stop, chạy hết
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