"""
KIẾN TRÚC CHUNG: Pretrained-CNN (+ tuỳ chọn fine-tune 1 phần) + Few-shot
learning + K-Fold Cross Validation.

BẢN SỬA so với bản gốc, giải quyết 4 vấn đề đã trao đổi:

  [Mục 2] Augmentation bớt "hung hãn" hơn với các lỗi nhỏ/tinh vi:
          RandomResizedCrop scale hẹp lại (mặc định 0.9-1.0 thay vì
          0.8-1.0), rotation giảm mặc định (10° thay vì 20°) - đều có thể
          chỉnh lại qua CONFIG.

  [Mục 3] Cho phép "hé mở" (fine-tune) vài block cuối của backbone thay vì
          đóng băng 100% - bật bằng UNFREEZE_LAST_BLOCK=True. Khi bật,
          pipeline tự chuyển sang train END-TO-END (chậm hơn embedding
          cache, nhưng có khả năng tách các lỗi tinh vi tốt hơn linear
          probe thuần).

  [Mục 4] Model "sản xuất" cuối cùng giờ được trích 1 phần dữ liệu làm
          validation riêng (FINAL_MODEL_VAL_RATIO) để có early-stopping
          thực sự, thay vì train mù 1 số epoch cố định không giám sát.

  [Mục 5] SỬA LỖI báo cáo Grad-CAM: trước đây heatmap được tính từ model
          "sản xuất" (đã học thuộc toàn bộ ảnh) nhưng lại gắn nhãn dự đoán
          của model K-Fold (chưa từng thấy ảnh đó) - 2 model khác nhau bị
          trộn lẫn. Giờ Grad-CAM cho mỗi ảnh dùng ĐÚNG model của fold đã
          tạo ra oof_pred[i] cho ảnh đó.

  [Mục 6] Thêm CALIBRATE_TEMPERATURE: sau K-Fold, khớp 1 hệ số nhiệt độ T
          (temperature scaling) trên toàn bộ logit out-of-fold để % hiển
          thị phản ánh đúng độ tin cậy thực tế hơn - lưu T vào checkpoint,
          các script inference (gradcam_heatmap.py, script chụp+phân loại)
          nên chia logit cho T trước khi softmax nếu muốn dùng % đã hiệu
          chỉnh (xem hàm calibrated_softmax() ở cuối file để tái sử dụng).

Cách áp dụng cho bài toán mới: chỉ sửa phần CONFIG bên dưới.

Cách chạy:
    python few_shot_pipeline_v2.py
"""

import os
import time
import random
import csv

import cv2
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
    r"D:\TongHop\RTC Technologi\PCB\cropVung\roi_1_split\train",
    r"D:\TongHop\RTC Technologi\PCB\cropVung\roi_1_split\val",
]

MODEL_DIR = r"D:\TongHop\RTC Technologi\PCB\model\model3"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")

# --- [Mục 2] Augmentation - đã giảm mức độ "phá" đặc trưng nhỏ/tinh vi ---
AUGMENT_CONFIG = {
    "horizontal_flip": True,       # tắt nếu ảnh có chữ/hướng quan trọng
    "rotation_degrees": 0,        # giảm từ 20 -> 10, đặt 0 để tắt hẳn
    "color_jitter": False,         # tắt nếu bài toán phụ thuộc MÀU SẮC
    "random_resized_crop": False,   # tắt nếu vị trí lỗi gần rìa ảnh hay bị cắt mất
    "random_resized_crop_scale": (0.9, 1.0),   # thu hẹp từ (0.8,1.0) -> (0.9,1.0)
}

AUGMENT_COPIES = 20
K_FOLDS = 10

HEAD_EPOCHS = 300
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 200
PRINT_EPOCH_DETAILS = True

# --- [Mục 3] Fine-tune 1 phần backbone thay vì đóng băng 100% ---
UNFREEZE_LAST_BLOCK = False   # True = "hé mở" 2 block cuối backbone để fine-tune
BACKBONE_LR = 1e-4            # LR riêng cho phần backbone được mở (nên nhỏ hơn HEAD_LR nhiều)
FINETUNE_EPOCHS = 80          # số epoch khi train kiểu fine-tune (chậm hơn embedding-cache nên ít epoch hơn)
FINETUNE_PATIENCE = 40

# --- [Mục 4] Validation riêng cho model sản xuất cuối cùng (early-stopping thật) ---
FINAL_MODEL_VAL_RATIO = 0.15   # trích ~15% dữ liệu (chia đều mỗi class) làm val cho model cuối

# --- [Mục 6] Hiệu chỉnh xác suất (temperature scaling) ---
CALIBRATE_TEMPERATURE = True
TEMPERATURE_LR = 0.01
TEMPERATURE_STEPS = 200

RANDOM_SEED = 42

# --- Báo cáo Grad-CAM sau khi train xong ---
GENERATE_GRADCAM_REPORT = True
GRADCAM_REPORT_DIR = os.path.join(MODEL_DIR, "gradcam_report")
GRADCAM_MAX_PER_CLASS = 30

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
        scale = augment_cfg.get("random_resized_crop_scale", (0.9, 1.0))
        aug_ops.append(transforms.RandomResizedCrop(224, scale=scale))
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


def _pretrained_weights(backbone_name):
    if backbone_name == "mobilenet_v2":
        return models.MobileNet_V2_Weights.IMAGENET1K_V1
    elif backbone_name == "mobilenet_v3_large":
        return models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    elif backbone_name == "mobilenet_v3_small":
        return models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")


def build_backbone(backbone_name):
    """Backbone ĐÓNG BĂNG HOÀN TOÀN - dùng cho path embedding-cache (nhanh)."""
    if backbone_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=_pretrained_weights(backbone_name))
        embedding_dim = backbone.last_channel
    elif backbone_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=_pretrained_weights(backbone_name))
        embedding_dim = 960
    elif backbone_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=_pretrained_weights(backbone_name))
        embedding_dim = 576
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, embedding_dim


def build_full_model(backbone_name, num_classes=None, dropout=0.3, pretrained=True):
    """Model đầy đủ (backbone + classifier). Nếu pretrained=False, khởi tạo
    kiến trúc rỗng (dùng lúc NẠP checkpoint đã train, không cần tải lại
    trọng số ImageNet)."""
    weights = _pretrained_weights(backbone_name) if pretrained else None

    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        embedding_dim = model.last_channel
        if num_classes is not None:
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        embedding_dim = 960
        if num_classes is not None:
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        embedding_dim = 576
        if num_classes is not None:
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    for p in model.features.parameters():
        p.requires_grad = False
    return model, embedding_dim


# --------------------------------------------------------------------------
# [Mục 3] Mở khoá (fine-tune) vài block cuối của backbone
# --------------------------------------------------------------------------

def unfreeze_last_block(model, n_blocks=2):
    """Cho phép train n_blocks CUỐI CÙNG trong model.features (mặc định cả
    2 backbone MobileNetV2/V3 đều có cấu trúc features = Sequential các
    block, nên cách này dùng chung được cho cả 3 biến thể)."""
    children = list(model.features.children())
    n_unfreeze = min(n_blocks, len(children))
    for module in children[-n_unfreeze:]:
        for p in module.parameters():
            p.requires_grad = True
    return model


def build_finetune_model(backbone_name, num_classes, dropout):
    model, embedding_dim = build_full_model(backbone_name, num_classes, dropout, pretrained=True)
    unfreeze_last_block(model)
    return model.to(DEVICE), embedding_dim


@torch.no_grad()
def extract_embedding(backbone, image_tensor):
    x = backbone.features(image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


# --------------------------------------------------------------------------
# Precompute: embedding (path nhanh) HOẶC tensor ảnh thô (path fine-tune)
# --------------------------------------------------------------------------

@torch.no_grad()
def precompute_all_embeddings(backbone, image_paths, augment_transform, clean_transform, augment_copies):
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


def precompute_all_tensors(image_paths, augment_transform, clean_transform, augment_copies):
    """Giống precompute_all_embeddings nhưng giữ lại TENSOR ẢNH THÔ (chưa
    qua backbone) - dùng cho path fine-tune, vì backbone không còn đóng
    băng hoàn toàn nên phải forward lại mỗi epoch (không cache được
    embedding cố định)."""
    augmented_t, clean_t = [], []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        aug_tensors = torch.stack([augment_transform(image) for _ in range(augment_copies)])
        augmented_t.append(aug_tensors)  # giữ trên CPU, chuyển GPU theo batch lúc train
        clean_tensor = clean_transform(image).unsqueeze(0)
        clean_t.append(clean_tensor)
    return augmented_t, clean_t


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


def stratified_holdout_indices(labels, val_ratio, seed=42):
    """[Mục 4] Trích ra 1 tập validation cố định (chia đều theo class) để
    early-stopping cho model sản xuất cuối cùng."""
    labels = np.array(labels)
    rng = np.random.RandomState(seed)
    val_mask = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * val_ratio))) if len(idx_c) > 1 else 0
        val_mask[idx_c[:n_val]] = True
    return val_mask


def make_head(embedding_dim, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)).to(DEVICE)


def train_head(train_emb, train_labels, val_emb, val_labels, embedding_dim, num_classes,
                epochs, lr, dropout, patience, print_epochs=False, fold_label=""):
    """Train 1 classifier head trên embedding đã cache (path nhanh, backbone
    đóng băng 100%). Trả về thêm best_val_logits (RAW, trước softmax) để
    phục vụ [Mục 6] hiệu chỉnh nhiệt độ."""
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
    best_val_logits = None
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
                best_val_logits = val_outputs.detach().cpu()
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
    return head, best_val_acc, best_val_loss, best_val_probs, best_val_logits


# --------------------------------------------------------------------------
# [Mục 3] Train kiểu FINE-TUNE end-to-end (chậm hơn, dùng khi UNFREEZE_LAST_BLOCK=True)
# --------------------------------------------------------------------------

def train_finetune(train_aug_tensors, train_labels_expanded, val_clean_tensors, val_labels,
                    backbone_name, num_classes, dropout, epochs, head_lr, backbone_lr,
                    patience, print_epochs=False, fold_label=""):
    """Giống train_head() nhưng forward qua CẢ backbone (2 block cuối được
    mở khoá) thay vì chỉ 1 lớp Linear trên embedding có sẵn."""
    model, embedding_dim = build_finetune_model(backbone_name, num_classes, dropout)

    class_weights = train_aug_tensors.shape[0] / (
        num_classes * np.bincount(train_labels_expanded.cpu().numpy(), minlength=num_classes)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    has_val = val_clean_tensors is not None and val_clean_tensors.shape[0] > 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_probs = None
    best_val_logits = None
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        outputs = model(train_aug_tensors)
        loss = criterion(outputs, train_labels_expanded)
        loss.backward()
        optimizer.step()
        train_acc = (outputs.argmax(dim=1) == train_labels_expanded).float().mean().item()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_clean_tensors)
                val_loss = criterion(val_outputs, val_labels).item()
                val_probs = F.softmax(val_outputs, dim=1)
                val_acc = (val_probs.argmax(dim=1) == val_labels).float().mean().item()
            scheduler.step(val_acc)

            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_probs = val_probs.cpu()
                best_val_logits = val_outputs.detach().cpu()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if print_epochs:
                epoch_time_ms = (time.time() - epoch_start) * 1000
                print(f"  [{fold_label}] Epoch {epoch+1:3d}/{epochs} | Train loss: {loss.item():.4f} "
                      f"| Train acc: {train_acc:.2%} | {epoch_time_ms:.1f}ms")

    model.load_state_dict(best_state)
    return model, best_val_acc, best_val_loss, best_val_probs, best_val_logits


# --------------------------------------------------------------------------
# [Mục 6] Hiệu chỉnh nhiệt độ (temperature scaling)
# --------------------------------------------------------------------------

def fit_temperature(oof_logits, oof_labels, lr=TEMPERATURE_LR, steps=TEMPERATURE_STEPS):
    """Khớp 1 hệ số nhiệt độ T sao cho softmax(logits / T) khớp tốt nhất
    với nhãn thật (tối thiểu hoá NLL) - chuẩn "temperature scaling" phổ
    biến để hiệu chỉnh xác suất, không đổi thứ tự xếp hạng class, chỉ làm
    con số % đáng tin hơn."""
    logits = oof_logits.to(DEVICE)
    labels = oof_labels.to(DEVICE)
    log_temperature = torch.zeros(1, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        optimizer.zero_grad()
        T = torch.exp(log_temperature)
        loss = criterion(logits / T, labels)
        loss.backward()
        optimizer.step()

    T_final = torch.exp(log_temperature).item()
    return T_final


def calibrated_softmax(logits, temperature):
    """Hàm tiện ích để TÁI SỬ DỤNG trong các script inference khác (script
    chụp ảnh + phân loại, gradcam_heatmap.py...): thay vì
    F.softmax(output, dim=1), dùng calibrated_softmax(output, temperature)
    với temperature lấy từ checkpoint["temperature"]."""
    return F.softmax(logits / temperature, dim=1)


# --------------------------------------------------------------------------
# Báo cáo chi tiết
# --------------------------------------------------------------------------

def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names,
                           temperature=None):
    num_classes = len(class_names)
    n = len(image_paths)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(n):
        confusion[labels_np[i], oof_pred[i]] += 1

    print("\n" + "=" * 100)
    print("BÁO CÁO CHI TIẾT (out-of-fold - đánh giá trên TOÀN BỘ dữ liệu, không thiên vị)")
    if temperature is not None:
        print(f"(Xác suất bên dưới đã được HIỆU CHỈNH bằng temperature scaling, T={temperature:.3f})")
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


# --------------------------------------------------------------------------
# GRAD-CAM REPORT - [Mục 5] SỬA: dùng đúng model của từng fold
# --------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output


def _overlay_heatmap(img_bgr, cam, alpha=0.45):
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)


def generate_gradcam_report(fold_models, image_paths, labels_np, oof_pred, oof_probs, oof_fold,
                             class_names, report_dir, max_per_class, gradcam_transform):
    """[Mục 5] fold_models: dict {fold_index: model_of_that_fold}. Mỗi ảnh
    dùng ĐÚNG model của fold đã sinh ra oof_pred[i] cho ảnh đó - khớp chính
    xác với model đã tạo ra dự đoán, không lẫn sang model sản xuất cuối."""
    os.makedirs(report_dir, exist_ok=True)
    n = len(image_paths)
    selected = []

    for c in range(len(class_names)):
        idx_c = [i for i in range(n) if labels_np[i] == c]
        wrong = [i for i in idx_c if oof_pred[i] != c]
        correct = [i for i in idx_c if oof_pred[i] == c]
        chosen = wrong[:max_per_class]
        if len(chosen) < max_per_class:
            chosen += correct[:max_per_class - len(chosen)]
        selected.extend(chosen)

    print(f"\nĐang sinh Grad-CAM cho {len(selected)} ảnh (ưu tiên ảnh đoán sai, "
          f"mỗi ảnh dùng ĐÚNG model của fold đã đánh giá nó)...")
    rows = []
    gradcam_cache = {}   # cache GradCAM object theo fold, tránh tạo lại nhiều lần

    for i in selected:
        path = image_paths[i]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        fold_idx = oof_fold[i]
        model_this_fold = fold_models[fold_idx]

        if fold_idx not in gradcam_cache:
            target_layer = model_this_fold.features[-1]
            gradcam_cache[fold_idx] = GradCAM(model_this_fold, target_layer)
        gradcam = gradcam_cache[fold_idx]

        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        input_tensor = gradcam_transform(pil_img).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        true_cls = class_names[labels_np[i]]
        pred_cls = class_names[oof_pred[i]]
        status = "dung" if oof_pred[i] == labels_np[i] else "SAI"

        cam, output = gradcam.compute(input_tensor, oof_pred[i])
        overlay = _overlay_heatmap(img_bgr, cam)

        label_text = f"that={true_cls} doan={pred_cls} ({status}) fold={fold_idx+1}"
        cv2.putText(overlay, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(overlay, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        fname = os.path.basename(path)
        out_name = f"{status}_{true_cls}_as_{pred_cls}_{fname}"
        out_path = os.path.join(report_dir, out_name)
        cv2.imwrite(out_path, overlay)

        rows.append([fname, true_cls, pred_cls, status, fold_idx + 1, out_path])

    csv_path = os.path.join(report_dir, "gradcam_report_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Class_that", "Class_doan", "Dung_Sai", "Fold", "Anh_gradcam"])
        writer.writerows(rows)

    print(f"Đã lưu {len(rows)} ảnh Grad-CAM + tóm tắt CSV tại: {report_dir}")


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================
def main():
    print(f"Đang chạy trên: {DEVICE}")
    print(f"Backbone: {BACKBONE_NAME} | Fine-tune backbone: {UNFREEZE_LAST_BLOCK}")

    image_paths, labels, class_names = list_images(DATASET_DIRS)
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class: {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")
    if counts.min() < K_FOLDS:
        print(f"  CẢNH BÁO: class ít ảnh nhất chỉ có {counts.min()} ảnh, nhỏ hơn K_FOLDS={K_FOLDS}. "
              f"Một số fold sẽ không có đủ ảnh của class đó trong tập validation, khiến "
              f"acc từng fold dao động mạnh. Cân nhắc giảm K_FOLDS.")

    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)
    labels_np = np.array(labels)
    fold_assignment = stratified_k_fold_indices(labels, K_FOLDS, seed=RANDOM_SEED)

    oof_pred = [None] * len(image_paths)
    oof_probs = [None] * len(image_paths)
    oof_logits_list = [None] * len(image_paths)
    oof_fold = [None] * len(image_paths)
    fold_models = {}   # [Mục 5] lưu lại model của TỪNG fold để dùng đúng model khi sinh Grad-CAM

    fold_accs, fold_losses = [], []
    embedding_dim = None

    if not UNFREEZE_LAST_BLOCK:
        # ------------------- PATH NHANH: embedding-cache -------------------
        backbone, embedding_dim = build_backbone(BACKBONE_NAME)
        print(f"Embedding dimension: {embedding_dim}")
        print(f"\nĐang extract embedding cho toàn bộ {len(image_paths)} ảnh "
              f"({AUGMENT_COPIES} bản augment/ảnh, chỉ chạy 1 lần)...")
        t0 = time.time()
        augmented_emb_list, clean_emb_list = precompute_all_embeddings(
            backbone, image_paths, augment_transform, clean_transform, AUGMENT_COPIES
        )
        print(f"  -> Xong sau {time.time()-t0:.1f}s")

        print(f"\n=== K-Fold Cross Validation (K={K_FOLDS}) ===")
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
            head, val_acc, val_loss, val_probs, val_logits = train_head(
                train_emb, train_lbl, val_emb, val_lbl, embedding_dim, num_classes,
                HEAD_EPOCHS, HEAD_LR, HEAD_DROPOUT, EARLY_STOP_PATIENCE,
                print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"Fold {fold+1}/{K_FOLDS}"
            )
            fold_accs.append(val_acc)
            fold_losses.append(val_loss)

            # [Mục 5] Ráp lại model đầy đủ (backbone đóng băng + head của fold này) để dùng cho Grad-CAM
            fold_full_model, _ = build_full_model(BACKBONE_NAME, num_classes, HEAD_DROPOUT, pretrained=True)
            fold_full_model.classifier = head
            fold_full_model = fold_full_model.to(DEVICE).eval()
            fold_models[fold] = fold_full_model

            for local_i, global_i in enumerate(val_idx):
                probs_row = val_probs[local_i]
                oof_pred[global_i] = int(probs_row.argmax().item())
                oof_probs[global_i] = probs_row.tolist()
                oof_logits_list[global_i] = val_logits[local_i]
                oof_fold[global_i] = fold

            print(f"Fold {fold+1}/{K_FOLDS}: {len(train_idx)} ảnh train, {len(val_idx)} ảnh val "
                  f"-> Val acc: {val_acc:.2%} | Val loss: {val_loss:.4f}")

    else:
        # ------------------- PATH FINE-TUNE: end-to-end -------------------
        print(f"\nĐang chuẩn bị tensor ảnh thô (fine-tune mode - không cache embedding)...")
        t0 = time.time()
        augmented_t_list, clean_t_list = precompute_all_tensors(
            image_paths, augment_transform, clean_transform, AUGMENT_COPIES
        )
        print(f"  -> Xong sau {time.time()-t0:.1f}s")

        print(f"\n=== K-Fold Cross Validation FINE-TUNE (K={K_FOLDS}) ===")
        for fold in range(K_FOLDS):
            val_idx = np.where(fold_assignment == fold)[0]
            train_idx = np.where(fold_assignment != fold)[0]

            train_tensors = torch.cat([augmented_t_list[i] for i in train_idx], dim=0).to(DEVICE)
            train_lbl = torch.tensor(
                [labels_np[i] for i in train_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long
            ).to(DEVICE)
            val_tensors = torch.cat([clean_t_list[i] for i in val_idx], dim=0).to(DEVICE)
            val_lbl = torch.tensor([labels_np[i] for i in val_idx], dtype=torch.long).to(DEVICE)

            print(f"\n--- Fold {fold+1}/{K_FOLDS} ({len(train_idx)} ảnh train, {len(val_idx)} ảnh val) ---")
            model, val_acc, val_loss, val_probs, val_logits = train_finetune(
                train_tensors, train_lbl, val_tensors, val_lbl,
                BACKBONE_NAME, num_classes, HEAD_DROPOUT, FINETUNE_EPOCHS, HEAD_LR, BACKBONE_LR,
                FINETUNE_PATIENCE, print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"Fold {fold+1}/{K_FOLDS}"
            )
            fold_accs.append(val_acc)
            fold_losses.append(val_loss)
            fold_models[fold] = model.eval()

            for local_i, global_i in enumerate(val_idx):
                probs_row = val_probs[local_i]
                oof_pred[global_i] = int(probs_row.argmax().item())
                oof_probs[global_i] = probs_row.tolist()
                oof_logits_list[global_i] = val_logits[local_i]
                oof_fold[global_i] = fold

            print(f"Fold {fold+1}/{K_FOLDS}: {len(train_idx)} ảnh train, {len(val_idx)} ảnh val "
                  f"-> Val acc: {val_acc:.2%} | Val loss: {val_loss:.4f}")

    fold_accs = np.array(fold_accs)
    print(f"\n>>> Kết quả K-Fold: Accuracy trung bình = {fold_accs.mean():.2%} "
          f"(độ lệch chuẩn ±{fold_accs.std():.2%})")

    # --------------------------------------------------------------------
    # [Mục 6] Hiệu chỉnh nhiệt độ trên toàn bộ logit out-of-fold
    # --------------------------------------------------------------------
    temperature = None
    if CALIBRATE_TEMPERATURE:
        oof_logits_tensor = torch.stack(oof_logits_list, dim=0)
        oof_labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        temperature = fit_temperature(oof_logits_tensor, oof_labels_tensor)
        print(f"\n[Hiệu chỉnh nhiệt độ] T = {temperature:.3f} "
              f"(T > 1 nghĩa là model gốc đang QUÁ TỰ TIN, T < 1 nghĩa là đang QUÁ RỤT RÈ)")

        calibrated_probs = calibrated_softmax(oof_logits_tensor, temperature).numpy()
        oof_probs_calibrated = [row.tolist() for row in calibrated_probs]
    else:
        oof_probs_calibrated = oof_probs

    # --------------------------------------------------------------------
    # BÁO CÁO CHI TIẾT (in cả bản CHƯA và ĐÃ hiệu chỉnh để đối chiếu)
    # --------------------------------------------------------------------
    print("\n\n########## BÁO CÁO VỚI XÁC SUẤT GỐC (chưa hiệu chỉnh) ##########")
    print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names)

    if CALIBRATE_TEMPERATURE:
        print("\n\n########## BÁO CÁO VỚI XÁC SUẤT ĐÃ HIỆU CHỈNH NHIỆT ĐỘ ##########")
        print_detailed_report(image_paths, labels_np, oof_pred, oof_probs_calibrated, oof_fold,
                               class_names, temperature=temperature)

    # --------------------------------------------------------------------
    # [Mục 4] Train model SẢN XUẤT cuối cùng - có validation riêng để early-stop
    # --------------------------------------------------------------------
    print(f"\n=== Train model cuối cùng (để deploy) ===")
    val_mask = stratified_holdout_indices(labels, FINAL_MODEL_VAL_RATIO, seed=RANDOM_SEED)
    train_final_idx = np.where(~val_mask)[0]
    val_final_idx = np.where(val_mask)[0]
    print(f"Chia dữ liệu cho model cuối: {len(train_final_idx)} ảnh train, "
          f"{len(val_final_idx)} ảnh validation (giữ lại để early-stopping thật).")

    if not UNFREEZE_LAST_BLOCK:
        all_train_emb = torch.cat([augmented_emb_list[i] for i in train_final_idx], dim=0).to(DEVICE)
        all_train_lbl = torch.tensor(
            [labels_np[i] for i in train_final_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long
        ).to(DEVICE)
        final_val_emb = torch.cat([clean_emb_list[i] for i in val_final_idx], dim=0).to(DEVICE)
        final_val_lbl = torch.tensor([labels_np[i] for i in val_final_idx], dtype=torch.long).to(DEVICE)

        final_head, final_val_acc, final_val_loss, _, _ = train_head(
            all_train_emb, all_train_lbl, final_val_emb, final_val_lbl, embedding_dim, num_classes,
            HEAD_EPOCHS, HEAD_LR, HEAD_DROPOUT, EARLY_STOP_PATIENCE,
            print_epochs=PRINT_EPOCH_DETAILS, fold_label="Model cuối cùng"
        )
        print(f"Model cuối cùng - Val acc (giữ riêng, chưa từng train): {final_val_acc:.2%} "
              f"| Val loss: {final_val_loss:.4f}")

        full_model, _ = build_full_model(BACKBONE_NAME, num_classes, HEAD_DROPOUT, pretrained=True)
        full_model.classifier = final_head
        full_model = full_model.to(DEVICE)
        full_model.eval()
    else:
        all_train_tensors = torch.cat([augmented_t_list[i] for i in train_final_idx], dim=0).to(DEVICE)
        all_train_lbl = torch.tensor(
            [labels_np[i] for i in train_final_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long
        ).to(DEVICE)
        final_val_tensors = torch.cat([clean_t_list[i] for i in val_final_idx], dim=0).to(DEVICE)
        final_val_lbl = torch.tensor([labels_np[i] for i in val_final_idx], dtype=torch.long).to(DEVICE)

        full_model, final_val_acc, final_val_loss, _, _ = train_finetune(
            all_train_tensors, all_train_lbl, final_val_tensors, final_val_lbl,
            BACKBONE_NAME, num_classes, HEAD_DROPOUT, FINETUNE_EPOCHS, HEAD_LR, BACKBONE_LR,
            FINETUNE_PATIENCE, print_epochs=PRINT_EPOCH_DETAILS, fold_label="Model cuối cùng"
        )
        full_model.eval()
        print(f"Model cuối cùng - Val acc (giữ riêng, chưa từng train): {final_val_acc:.2%} "
              f"| Val loss: {final_val_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
        "final_model_val_acc": float(final_val_acc),
        "temperature": float(temperature) if temperature is not None else 1.0,
        "unfreeze_last_block": UNFREEZE_LAST_BLOCK,
    }, MODEL_OUT)
    print(f"Đã lưu model cuối cùng tại: {MODEL_OUT}")
    if temperature is not None:
        print(f"  (Đã lưu kèm temperature={temperature:.3f} trong checkpoint - dùng "
              f"calibrated_softmax(logits, checkpoint['temperature']) lúc inference "
              f"để có % hiệu chỉnh thay vì F.softmax thường.)")

    # --------------------------------------------------------------------
    # [Mục 5] BÁO CÁO GRAD-CAM - dùng ĐÚNG model của từng fold
    # --------------------------------------------------------------------
    if GENERATE_GRADCAM_REPORT:
        generate_gradcam_report(
            fold_models, image_paths, labels_np, oof_pred, oof_probs, oof_fold,
            class_names, GRADCAM_REPORT_DIR, GRADCAM_MAX_PER_CLASS, clean_transform,
        )

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
    print(f"Backbone: {BACKBONE_NAME} (fine-tune: {UNFREEZE_LAST_BLOCK})")
    print(f"Latency trung bình (1 ảnh): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")
    print(f"K-Fold accuracy (độ tin cậy thật): {fold_accs.mean():.2%} ± {fold_accs.std():.2%}")
    print(f"Val accuracy model cuối (giữ riêng, chưa từng train): {final_val_acc:.2%}")


if __name__ == "__main__":
    main()