"""
Train một classifier nhẹ (MobileNetV2) bằng transfer learning.
PHIÊN BẢN CẢI TIẾN: thêm class weights, fine-tuning 2 giai đoạn,
learning rate scheduler, và early stopping.

Cách chạy:
    python train_edge_classifier_v2.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os

# ----------------------------
# 1. Cấu hình
# ----------------------------
DATA_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset"          # thư mục chứa train/ và val/
BATCH_SIZE = 8

# --- Cấu hình 2 giai đoạn train ---
EPOCHS_PHASE1 = 20        # giai đoạn 1: freeze backbone, chỉ train classifier
EPOCHS_PHASE2 = 20        # giai đoạn 2: mở khóa vài layer cuối, fine-tune nhẹ
LR_PHASE1 = 1e-3          # learning rate lúc chỉ train classifier
LR_PHASE2 = 1e-5          # learning rate nhỏ hơn nhiều khi fine-tune backbone
UNFREEZE_LAST_N_BLOCKS = 3  # số block cuối của backbone sẽ được mở khóa ở giai đoạn 2

# --- Early stopping ---
EARLY_STOP_PATIENCE = 50   # dừng sớm nếu val_acc không cải thiện sau N epoch liên tiếp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thư mục lưu model - tự tạo nếu chưa tồn tại
MODEL_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "edge_classifier.pt")

print(f"Đang chạy trên: {DEVICE}")

# ----------------------------
# 2. Data augmentation + loader
# ----------------------------
# Augmentation giúp "nhân bản" dữ liệu ít ỏi của bạn thành nhiều biến thể
# Lưu ý: nếu ảnh của bạn có chữ/ký hiệu định hướng quan trọng, cân nhắc bỏ
# RandomHorizontalFlip vì nó có thể làm sai lệch ý nghĩa của ảnh.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Các class phát hiện được: {class_names}")

# ----------------------------
# 3. Tính class weights (xử lý mất cân bằng dữ liệu)
# ----------------------------
# Đếm số ảnh mỗi class trong tập train
class_counts = np.bincount(
    [label for _, label in train_dataset.samples], minlength=num_classes
)
print(f"Số ảnh mỗi class: {dict(zip(class_names, class_counts))}")

total_samples = class_counts.sum()
# Class càng ít ảnh thì weight càng cao -> loss phạt nặng hơn khi đoán sai class đó
class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {dict(zip(class_names, class_weights.round(3)))}")

# ----------------------------
# 4. Model: MobileNetV2 pretrained (transfer learning)
# ----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Đóng băng toàn bộ backbone -> chỉ train phần classifier cuối (giai đoạn 1)
for param in model.features.parameters():
    param.requires_grad = False

# Thay lớp classifier cuối cho đúng số class của bạn
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model = model.to(DEVICE)

# ----------------------------
# 5. Loss (có class weights) + optimizer + scheduler
# ----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1
)

# Giảm learning rate khi val_acc ngừng cải thiện, giúp hội tụ mượt hơn về cuối
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ----------------------------
# 6. Các hàm hỗ trợ
# ----------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def unfreeze_last_n_blocks(model, n):
    """Mở khóa n block cuối cùng của backbone MobileNetV2 để fine-tune."""
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    print(f"Đã mở khóa {n} block cuối của backbone để fine-tune.")


# ----------------------------
# 7. Vòng lặp train chính (2 giai đoạn + early stopping)
# ----------------------------
best_val_acc = 0.0
epochs_no_improve = 0
total_epochs_run = 0

# Ghép 2 giai đoạn thành 1 danh sách để dùng chung 1 vòng lặp
phases = [
    ("Giai đoạn 1 (freeze backbone)", EPOCHS_PHASE1, None),
    ("Giai đoạn 2 (fine-tune backbone)", EPOCHS_PHASE2, UNFREEZE_LAST_N_BLOCKS),
]

for phase_name, phase_epochs, unfreeze_n in phases:
    print(f"\n=== {phase_name} ===")

    # Nếu là giai đoạn 2: mở khóa vài layer cuối + tạo lại optimizer với LR nhỏ hơn
    if unfreeze_n is not None:
        unfreeze_last_n_blocks(model, unfreeze_n)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

    for epoch in range(phase_epochs):
        total_epochs_run += 1
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{phase_name}] Epoch {epoch+1}/{phase_epochs} "
              f"| Train loss: {train_loss:.4f} | Val acc: {val_acc:.2%} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
            }, MODEL_OUT)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Val acc không cải thiện sau {EARLY_STOP_PATIENCE} epoch liên tiếp "
                  f"-> dừng sớm ở epoch {total_epochs_run}.")
            break
    else:
        continue
    break  # thoát cả 2 vòng lặp nếu early stopping đã kích hoạt

print(f"\nXong! Model tốt nhất (val acc={best_val_acc:.2%}) đã lưu tại: {MODEL_OUT}")

# ----------------------------
# 8. Đo các chỉ số "edge" cơ bản
# ----------------------------
# Load lại checkpoint tốt nhất trước khi đo, vì model hiện tại có thể là
# epoch cuối cùng (chưa chắc là bản tốt nhất do early stopping / overfit nhẹ)
checkpoint = torch.load(MODEL_OUT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# Đo latency
n_runs = 50
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(dummy_input)
avg_latency_ms = (time.time() - start) / n_runs * 1000

# Đo kích thước model
model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)

print(f"\n--- Thông số edge ---")
print(f"Latency trung bình (1 ảnh): {avg_latency_ms:.2f} ms")
print(f"Kích thước file model: {model_size_mb:.2f} MB")