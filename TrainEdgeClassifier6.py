"""
KIẾN TRÚC CHUNG: Pretrained-CNN + Few-shot learning + K-Fold Cross Validation.

Đây là pipeline TỔNG QUÁT, dùng lại được cho BẤT KỲ bài toán phân loại ảnh
ít dữ liệu nào (màu sắc, trái cây, OK/NG, vật thể...) - chỉ cần sửa phần
CONFIG bên dưới, không cần đụng vào logic code.

--------------------------------------------------------------------------
Ý TƯỞNG CỐT LÕI (không đổi giữa các bài toán):
--------------------------------------------------------------------------
1. Backbone CNN ĐÓNG BĂNG HOÀN TOÀN, không fine-tune. Hỗ trợ:
     - "mobilenet_v2" / "mobilenet_v3_large" / "mobilenet_v3_small": nhẹ,
       nhanh, phù hợp chạy edge.
     - "wide_resnet50_2": NẶNG hơn nhiều (~8-10 lần), nhưng cho embedding
       chất lượng cao hơn hẳn cho bài toán phát hiện bất thường/khiếm
       khuyết cục bộ - đây cũng chính là backbone anomalib mặc định dùng
       cho PatchCore, đã được kiểm chứng tốt trong domain PCB defect.
2. Extract embedding (vector đặc trưng) cho mỗi ảnh 1 LẦN DUY NHẤT, cache
   lại -> việc "train" sau đó chỉ là train 1 classifier NHẸ trên embedding,
   nhanh gấp hàng chục lần so với train lại cả CNN.
3. Vì dataset ít, KHÔNG chia train/val cố định một lần (dễ bị đánh giá sai
   lệch, không ổn định) -> dùng K-FOLD CROSS VALIDATION.
4. Sau khi đánh giá xong bằng K-Fold, train 1 model "sản xuất" cuối cùng
   trên TOÀN BỘ dữ liệu (không giữ lại val) để deploy.

--------------------------------------------------------------------------
CÁCH ÁP DỤNG CHO BÀI TOÁN MỚI:
--------------------------------------------------------------------------
Chỉ cần sửa phần CONFIG bên dưới:
  - DATASET_DIRS: trỏ tới (các) thư mục chứa ảnh, mỗi class 1 thư mục con
  - AUGMENT_CONFIG: bật/tắt phép augmentation phù hợp với bài toán
  - BACKBONE_NAME, K_FOLDS, HEAD_EPOCHS...
Không cần sửa bất kỳ dòng logic nào khác trong file.

Cách chạy:
    python few_shot_pipeline.py
"""

import os
import time
import random

import numpy as np
import cv2
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

MODEL_DIR = r"D:\TongHop\RTC Technologi\PCB\model\model11"
# "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small" | "wide_resnet50_2"
BACKBONE_NAME = "wide_resnet50_2"
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")

AUGMENT_CONFIG = {
    "horizontal_flip": True,     # tắt nếu pipeline upstream đã đồng bộ hướng (SIFT)
    "rotation_degrees": 10,
    "color_jitter": True,        # bật nếu bài toán KHÔNG phụ thuộc màu sắc tuyệt đối
    "random_resized_crop": False,
}

AUGMENT_COPIES = 20
K_FOLDS = 5

HEAD_EPOCHS = 300
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 200
PRINT_EPOCH_DETAILS = True

# Số ảnh MỖI CLASS sẽ được chọn ngẫu nhiên để lưu heatmap minh hoạ sau khi
# train xong model cuối cùng - giúp XEM TRỰC QUAN model đang "nhìn" vào
# đâu để ra quyết định, không cần chạy riêng script Grad-CAM/camera.
SO_ANH_MINH_HOA_MOI_CLASS = 8
THU_MUC_HEATMAP_MAU = os.path.join(MODEL_DIR, "heatmap_mau")

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Các backbone thuộc họ ResNet (kiến trúc conv1/bn1/relu/maxpool/layer1-4,
# không có .features/.classifier như MobileNet) - dùng để rẽ nhánh logic
# 1 chỗ duy nhất, dễ thêm resnet18/resnet34/resnet50... sau này nếu cần.
HO_RESNET = {"wide_resnet50_2", "resnet18", "resnet34", "resnet50"}

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
    """Trả về (backbone đã đóng băng, embedding_dim)."""
    if backbone_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        embedding_dim = backbone.last_channel
    elif backbone_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        embedding_dim = 960
    elif backbone_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        embedding_dim = 576
    elif backbone_name == "wide_resnet50_2":
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        embedding_dim = 2048
    elif backbone_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        embedding_dim = 512
    elif backbone_name == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        embedding_dim = 512
    elif backbone_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        embedding_dim = 2048
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, embedding_dim


def spatial_features(backbone, backbone_name, x):
    """Lấy feature map KHÔNG GIAN (trước bước Global Average Pooling) -
    dùng chung cho cả họ MobileNet (.features) và họ ResNet (không có
    .features, phải tự forward qua từng khối con)."""
    if backbone_name in HO_RESNET:
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        return x
    else:
        return backbone.features(x)


def build_full_model(backbone_name):
    """Trả về model ĐẦY ĐỦ (chưa gắn head) dùng để lưu/deploy cuối cùng.
    Với MobileNet, phần classifier sẽ được thay bằng head mới ở nơi gọi.
    Với ResNet, phần fc sẽ được thay bằng head mới ở nơi gọi."""
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
    elif backbone_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
    elif backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
    elif backbone_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
    elif backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model


def gan_head_vao_model(model, backbone_name, head):
    """Gắn classifier head (đã train) vào đúng vị trí tuỳ kiến trúc:
    MobileNet dùng .classifier, ResNet dùng .fc."""
    if backbone_name in HO_RESNET:
        model.fc = head
    else:
        model.classifier = head
    return model


@torch.no_grad()
def extract_embedding(backbone, backbone_name, image_tensor):
    x = spatial_features(backbone, backbone_name, image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


@torch.no_grad()
def precompute_all_embeddings(backbone, backbone_name, image_paths,
                               augment_transform, clean_transform, augment_copies):
    augmented_emb, clean_emb = [], []

    for path in image_paths:
        image = Image.open(path).convert("RGB")

        aug_tensors = torch.stack([augment_transform(image) for _ in range(augment_copies)]).to(DEVICE)
        aug_emb = extract_embedding(backbone, backbone_name, aug_tensors)
        augmented_emb.append(aug_emb.cpu())

        clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
        c_emb = extract_embedding(backbone, backbone_name, clean_tensor)
        clean_emb.append(c_emb.cpu())

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


class ChuanHoaEmbedding(nn.Module):
    """Lớp chuẩn hoá (trừ mean, chia std) được NHÚNG THẲNG vào đầu head,
    để khi model hoàn chỉnh (backbone + head) forward 1 ảnh bất kỳ lúc
    inference, bước chuẩn hoá TỰ ĐỘNG chạy đúng vị trí - không cần script
    inference phải tự làm lại thủ công (dễ quên/làm sai lệch).
    mean/std được đăng ký làm buffer -> tự động lưu/load cùng state_dict
    của model, không cần lưu riêng."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, embedding_dim))
        self.register_buffer("std", torch.ones(1, embedding_dim))

    def gan_gia_tri(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x):
        return (x - self.mean) / self.std


def make_head(embedding_dim, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)).to(DEVICE)


def tinh_chuan_hoa_embedding(embedding_tensor):
    """Tính mean/std của embedding trên TẬP TRAIN, dùng để chuẩn hoá input
    trước khi đưa vào Linear head. Bắt buộc phải lưu lại 2 giá trị này
    trong checkpoint và áp dụng LẠI ĐÚNG y hệt ở bước inference - nếu
    không, embedding ảnh mới sẽ lệch thang đo so với lúc train, khiến
    Linear layer (vốn không giới hạn output) dễ "nổ" điểm số một cách
    giả tạo, gây hiện tượng tự tin sai (overconfident wrong prediction)
    trên ảnh ngoài phân bố train."""
    mean = embedding_tensor.mean(dim=0, keepdim=True)
    std = embedding_tensor.std(dim=0, keepdim=True) + 1e-6
    return mean, std


def ap_dung_chuan_hoa(embedding_tensor, mean, std):
    return (embedding_tensor - mean) / std


def train_head(train_emb, train_labels, val_emb, val_labels, embedding_dim, num_classes,
                epochs, lr, dropout, patience, print_epochs=False, fold_label="",
                weight_decay=1e-4):
    head = make_head(embedding_dim, num_classes, dropout)

    class_weights = train_emb.shape[0] / (
        num_classes * np.bincount(train_labels.cpu().numpy(), minlength=num_classes)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # weight_decay > 0: ép trọng số Linear không phát triển quá lớn, giảm
    # nguy cơ ngoại suy quá tự tin (overconfident) khi gặp embedding lệch
    # nhẹ so với phân bố train (vd ảnh live qua pipeline camera thực tế).
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
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


def luu_anh_heatmap_mau(image_paths, labels_np, class_names, backbone, backbone_name,
                         clean_transform, final_head_day_du, thu_muc_luu,
                         so_luong_moi_class=8, seed=RANDOM_SEED):
    """Chọn ngẫu nhiên tối đa `so_luong_moi_class` ảnh MỖI CLASS, tính CAM
    cổ điển (Class Activation Mapping - công thức đóng, KHÔNG phải
    Grad-CAM xấp xỉ) và lưu ảnh overlay heatmap ra đĩa để xem trực quan
    model đang chú ý vùng nào khi ra quyết định.

    CAM cổ điển CHÍNH XÁC TUYỆT ĐỐI (không phải xấp xỉ) với kiến trúc
    GAP + Linear vì:
        logit_c = sum_k [ w_eff[c,k] * f_k(x,y) ]  (lấy trung bình theo x,y)
    Trong đó f_k(x,y) là feature map k tại vị trí (x,y) TRƯỚC bước Global
    Average Pooling, và w_eff là trọng số Linear đã gộp sẵn với bước
    chuẩn hoá (ChuanHoaEmbedding) ở đầu head - vì (emb-mean)/std rồi nhân
    W tương đương với nhân (W/std) trực tiếp lên emb gốc.
    """
    os.makedirs(thu_muc_luu, exist_ok=True)

    lop_chuan_hoa = final_head_day_du[0]
    linear = final_head_day_du[2]
    std = lop_chuan_hoa.std.detach().squeeze(0)        # (embedding_dim,)
    W = linear.weight.detach()                          # (num_classes, embedding_dim)
    W_eff = W / std.unsqueeze(0)                         # (num_classes, embedding_dim)

    backbone.eval()
    final_head_day_du.eval()

    rng = random.Random(seed)
    idx_theo_class = {i: [] for i in range(len(class_names))}
    for i, lb in enumerate(labels_np):
        idx_theo_class[int(lb)].append(i)

    idx_can_ve = []
    for c, idxs in idx_theo_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        idx_can_ve.extend(idxs[:so_luong_moi_class])

    print(f"\nĐang lưu {len(idx_can_ve)} ảnh minh hoạ CAM vào: {thu_muc_luu}")

    with torch.no_grad():
        for i in idx_can_ve:
            path = image_paths[i]
            true_label = int(labels_np[i])

            image = Image.open(path).convert("RGB")
            tensor = clean_transform(image).unsqueeze(0).to(DEVICE)

            feat_map = spatial_features(backbone, backbone_name, tensor)   # (1, C, h, w)
            emb = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)       # (1, C)

            logits = final_head_day_du(emb)
            probs = F.softmax(logits, dim=1)[0]
            pred_label = int(probs.argmax().item())
            confidence = probs[pred_label].item()

            # CAM theo đúng class MODEL DỰ ĐOÁN - cho biết model đang dựa
            # vào vùng nào để đưa ra quyết định đó (dù đúng hay sai).
            cam = torch.einsum("c,chw->hw", W_eff[pred_label], feat_map[0])
            cam = F.relu(cam).cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()

            anh_bgr = cv2.imread(path)
            if anh_bgr is None:
                continue
            anh_bgr = cv2.resize(anh_bgr, (224, 224))  # khớp đúng kích thước model nhìn thấy

            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap, 0.45, anh_bgr, 0.55, 0)

            dung_sai = "DUNG" if pred_label == true_label else "SAI"
            nhan = (f"That:{class_names[true_label]} Doan:{class_names[pred_label]} "
                    f"({confidence*100:.0f}%) {dung_sai}")
            cv2.putText(overlay, nhan, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ten_file_goc = os.path.splitext(os.path.basename(path))[0]
            ten_luu = f"{dung_sai}_that_{class_names[true_label]}_doan_{class_names[pred_label]}_{ten_file_goc}.png"
            cv2.imwrite(os.path.join(thu_muc_luu, ten_luu), overlay)

    print(f"  -> Đã lưu xong. Mở thư mục '{thu_muc_luu}' để xem.")


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
    if BACKBONE_NAME in HO_RESNET:
        print("  (Backbone họ ResNet nặng hơn MobileNet khá nhiều - bước này "
              "có thể chậm hơn hẳn so với trước, đặc biệt nếu chạy trên CPU.)")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, BACKBONE_NAME, image_paths, augment_transform, clean_transform, AUGMENT_COPIES
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # Tính mean/std để CHUẨN HOÁ embedding - bắt buộc phải làm với backbone
    # cho embedding lớn/chưa chuẩn hoá tốt (như wide_resnet50_2), nếu
    # không Linear head rất dễ ngoại suy sai lệch tự tin trên ảnh live
    # (xem giải thích chi tiết ở hàm tinh_chuan_hoa_embedding).
    # --------------------------------------------------------------------
    all_clean_emb_for_norm = torch.cat(clean_emb_list, dim=0)
    chuan_hoa_mean, chuan_hoa_std = tinh_chuan_hoa_embedding(all_clean_emb_for_norm)
    print(f"Đã tính mean/std chuẩn hoá embedding từ {all_clean_emb_for_norm.shape[0]} ảnh "
          f"(sẽ lưu vào checkpoint để dùng lại y hệt lúc inference).")

    augmented_emb_list = [ap_dung_chuan_hoa(e, chuan_hoa_mean, chuan_hoa_std) for e in augmented_emb_list]
    clean_emb_list = [ap_dung_chuan_hoa(e, chuan_hoa_mean, chuan_hoa_std) for e in clean_emb_list]

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
    # Train model SẢN XUẤT cuối cùng trên TOÀN BỘ dữ liệu (không giữ val)
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

    # "Nướng" bước chuẩn hoá embedding vào ĐẦU head trước khi gắn vào
    # model hoàn chỉnh - để khi deploy, model(anh) tự động chuẩn hoá
    # đúng như lúc train, không phụ thuộc script inference phải tự làm
    # đúng thủ công (nguồn gốc gây lệch train/inference trước đây).
    lop_chuan_hoa = ChuanHoaEmbedding(embedding_dim).to(DEVICE)
    lop_chuan_hoa.gan_gia_tri(chuan_hoa_mean.to(DEVICE), chuan_hoa_std.to(DEVICE))
    final_head_day_du = nn.Sequential(lop_chuan_hoa, *final_head).to(DEVICE)

    full_model = gan_head_vao_model(full_model, BACKBONE_NAME, final_head_day_du)

    # --------------------------------------------------------------------
    # Lưu ảnh minh hoạ CAM (heatmap) để XEM TRỰC QUAN model học được gì
    # --------------------------------------------------------------------
    luu_anh_heatmap_mau(
        image_paths, labels_np, class_names, backbone, BACKBONE_NAME,
        clean_transform, final_head_day_du, THU_MUC_HEATMAP_MAU,
        so_luong_moi_class=SO_ANH_MINH_HOA_MOI_CLASS,
    )
    full_model = full_model.to(DEVICE)
    full_model.eval()

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "kfold_val_acc_mean": float(fold_accs.mean()),
        "kfold_val_acc_std": float(fold_accs.std()),
        "chuan_hoa_mean": chuan_hoa_mean.cpu(),
        "chuan_hoa_std": chuan_hoa_std.cpu(),
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