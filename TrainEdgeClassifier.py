"""
Train một classifier nhẹ (MobileNetV2) bằng transfer learning.
Mục tiêu: hiểu pipeline train -> eval -> export cho edge.

Cách chạy:
    python train_edge_classifier.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os

# ----------------------------
# 1. Cấu hình
# ----------------------------
DATA_DIR = r"D:\TongHop\RTC Technologi\RD\Tutorial\dataset"          # thư mục chứa train/ và val/
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "edge_classifier.pt"

print(f"Đang chạy trên: {DEVICE}")

# ----------------------------
# 2. Data augmentation + loader
# ----------------------------
# Augmentation giúp "nhân bản" dữ liệu ít ỏi của bạn thành nhiều biến thể
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
# 3. Model: MobileNetV2 pretrained (transfer learning)
# ----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Đóng băng toàn bộ backbone -> chỉ train phần classifier cuối
for param in model.features.parameters():
    param.requires_grad = False

# Thay lớp classifier cuối cho đúng số class của bạn
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model = model.to(DEVICE)

# ----------------------------
# 4. Loss + optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
# Chỉ optimize các tham số chưa bị freeze (nhanh hơn nhiều)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# ----------------------------
# 5. Training loop
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


best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "class_names": class_names,
        }, MODEL_OUT)

print(f"\nXong! Model tốt nhất (val acc={best_val_acc:.2%}) đã lưu tại: {MODEL_OUT}")

# ----------------------------
# 6. Đo các chỉ số "edge" cơ bản
# ----------------------------
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