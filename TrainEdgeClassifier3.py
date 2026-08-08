"""
Train classifier bằng kỹ thuật "Pretrained-CNN + Few-shot learning".

Ý tưởng cốt lõi (khác với train_edge_classifier_v2.py):
- Backbone (MobileNetV2) ĐÓNG BĂNG HOÀN TOÀN, không bao giờ fine-tune.
- Vì backbone không đổi, ta CHỈ CẦN forward qua backbone 1 LẦN DUY NHẤT để
  lấy ra "embedding" (vector đặc trưng) cho mỗi ảnh, rồi CACHE lại.
- Sau đó, việc "train" chỉ là train một classifier RẤT NHẸ (vài trăm tham
  số) trên các embedding đã cache sẵn -> mỗi epoch chỉ mất ~0.01-0.1s
  thay vì vài giây/ảnh, vì không phải forward qua CNN nặng nữa.
- Để bù lại việc có ít ảnh gốc, ta tạo nhiều bản augmentation cho mỗi ảnh
  TRƯỚC khi extract embedding (chỉ tốn thời gian 1 lần, không lặp lại mỗi
  epoch) -> tăng "hiệu quả" số lượng mẫu train mà vẫn cực nhanh.

Kết quả checkpoint lưu ra có CÙNG CẤU TRÚC với train_edge_classifier_v2.py
(mobilenet_v2 + classifier[1] tùy chỉnh) -> dùng chung được với
test_edge_classifier.py / test_model_report.py, không cần sửa gì thêm.

Cách chạy:
    python train_few_shot_classifier.py
"""

import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# 1. Cấu hình
# ----------------------------
DATA_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset"  # chứa train/ và val/

MODEL_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "edge_classifier_fewshot.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Số bản augmentation tạo thêm cho MỖI ảnh train, để "nhân bản" dữ liệu ít ỏi
# trước khi extract embedding. Ví dụ 20 -> 1 ảnh gốc thành 20 embedding khác nhau.
AUGMENT_COPIES = 20

# Classifier head - train rất nhanh vì chỉ là 1-2 lớp Linear nhỏ trên embedding
HEAD_EPOCHS = 300         # tăng lên nhiều vì mỗi epoch chỉ ~3ms, không tốn kém gì
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 30  # tăng patience vì theo dõi val_loss (mượt hơn val_acc)

print(f"Đang chạy trên: {DEVICE}")

# ----------------------------
# 2. Augmentation để tạo thêm "biến thể" trước khi extract embedding
# ----------------------------
# LƯU Ý: vì đây là bài toán phân loại MÀU nên KHÔNG dùng ColorJitter
# (sẽ làm méo chính đặc trưng cần phân biệt). Nếu bài toán của bạn không
# phụ thuộc màu, có thể thêm ColorJitter lại.
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# Transform "sạch" - không augmentation, dùng cho ảnh gốc và val/test
clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ----------------------------
# 3. Hàm extract embedding bằng backbone đóng băng
# ----------------------------
@torch.no_grad()
def extract_embedding(backbone, image_tensor):
    """image_tensor: (B, 3, 224, 224) -> trả về (B, 1280) cho MobileNetV2."""
    x = backbone.features(image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def build_embedding_cache(backbone, dataset, transform, augment_copies=1):
    """
    Với mỗi ảnh trong dataset, tạo `augment_copies` bản biến thể (hoặc 1 bản
    sạch nếu augment_copies=1), extract embedding qua backbone đóng băng,
    rồi cache lại thành tensor (N, 1280) + label (N,).
    """
    all_embeddings = []
    all_labels = []

    for img_path, label in dataset.samples:
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        for _ in range(augment_copies):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            emb = extract_embedding(backbone, img_tensor)
            all_embeddings.append(emb.cpu())
            all_labels.append(label)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return embeddings, labels


# ----------------------------
# 4. Prototype classifier (baseline few-shot cổ điển, KHÔNG cần train)
# ----------------------------
def evaluate_prototype_classifier(train_emb, train_labels, val_emb, val_labels, num_classes):
    """
    Kỹ thuật few-shot kinh điển: tính "prototype" (vector trung bình) của mỗi
    class trong không gian embedding, rồi phân loại ảnh mới bằng cách xem nó
    gần prototype nào nhất (cosine similarity). Không cần train gradient nào.
    Dùng để so sánh/đối chiếu với classifier head có train.
    """
    prototypes = []
    for c in range(num_classes):
        class_embs = train_emb[train_labels == c]
        prototype = class_embs.mean(dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)  # (num_classes, 1280)

    val_norm = F.normalize(val_emb, dim=1)
    proto_norm = F.normalize(prototypes, dim=1)
    similarity = val_norm @ proto_norm.T  # (N_val, num_classes)
    preds = similarity.argmax(dim=1)

    acc = (preds == val_labels).float().mean().item()
    return acc


# ----------------------------
# 5. Main
# ----------------------------
def main():
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"))

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Các class phát hiện được: {class_names}")

    class_counts = np.bincount(
        [label for _, label in train_dataset.samples], minlength=num_classes
    )
    print(f"Số ảnh gốc mỗi class (train): {dict(zip(class_names, class_counts))}")

    # --- Tạo backbone đóng băng để extract embedding ---
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # --- Bước 1: Extract & cache embedding (chỉ làm 1 lần) ---
    print(f"\nĐang extract embedding cho tập train "
          f"({AUGMENT_COPIES} bản/ảnh, chỉ chạy 1 lần)...")
    t0 = time.time()
    train_emb, train_labels = build_embedding_cache(
        backbone, train_dataset, augment_transform, augment_copies=AUGMENT_COPIES
    )
    print(f"  -> {train_emb.shape[0]} embedding train, xong sau {time.time()-t0:.1f}s")

    print("Đang extract embedding cho tập val (không augmentation)...")
    t0 = time.time()
    val_emb, val_labels = build_embedding_cache(
        backbone, val_dataset, clean_transform, augment_copies=1
    )
    print(f"  -> {val_emb.shape[0]} embedding val, xong sau {time.time()-t0:.1f}s")

    train_emb, train_labels = train_emb.to(DEVICE), train_labels.to(DEVICE)
    val_emb, val_labels = val_emb.to(DEVICE), val_labels.to(DEVICE)

    # --- So sánh nhanh với Prototype classifier (không train gì cả) ---
    proto_acc = evaluate_prototype_classifier(
        train_emb.cpu(), train_labels.cpu(), val_emb.cpu(), val_labels.cpu(), num_classes
    )
    print(f"\n[Baseline - Prototype classifier, không train]: Val acc = {proto_acc:.2%}")

    # --- Bước 2: Train classifier head NHẸ trên embedding đã cache ---
    emb_dim = train_emb.shape[1]  # 1280 với MobileNetV2
    head = nn.Sequential(
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(emb_dim, num_classes),
    ).to(DEVICE)

    class_weights = train_emb.shape[0] / (num_classes * np.bincount(train_labels.cpu().numpy(), minlength=num_classes))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print(f"\n=== Train classifier head trên {train_emb.shape[0]} embedding đã cache ===")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_head_state = None
    epochs_no_improve = 0

    for epoch in range(HEAD_EPOCHS):
        epoch_start = time.time()

        head.train()
        optimizer.zero_grad()
        outputs = head(train_emb)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        train_acc = (outputs.argmax(dim=1) == train_labels).float().mean().item()

        head.eval()
        with torch.no_grad():
            val_outputs = head(val_emb)
            val_loss = criterion(val_outputs, val_labels).item()
            val_acc = (val_outputs.argmax(dim=1) == val_labels).float().mean().item()

        scheduler.step(val_acc)
        epoch_time = time.time() - epoch_start

        # QUAN TRỌNG: theo dõi val_loss thay vì val_acc để chọn "best".
        # Lý do: val_acc dễ bão hòa ở 100% rất sớm (nhất là với val set nhỏ),
        # trong khi val_loss vẫn tiếp tục giảm sau đó -> nghĩa là model vẫn
        # đang "tự tin hơn" dù accuracy không tăng thêm được nữa. Nếu chỉ
        # nhìn val_acc, model sẽ bị early-stop quá sớm, đúng nhưng thiếu
        # chắc chắn (confidence thấp) như trường hợp bạn gặp phải.
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
            status = "Best"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{HEAD_EPOCHS} | Train loss: {loss.item():.4f} "
              f"| Train acc: {train_acc:.2%} | Val loss: {val_loss:.4f} "
              f"| Val acc: {val_acc:.2%} | {status:<5} | {epoch_time*1000:.1f}ms")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Val loss không cải thiện sau {EARLY_STOP_PATIENCE} epoch -> dừng sớm.")
            break

    print(f"\nXong! Head tốt nhất đạt val acc = {best_val_acc:.2%} "
          f"(so với Prototype baseline không train: {proto_acc:.2%})")

    # --- Bước 3: Ghép head đã train vào model đầy đủ, lưu checkpoint ---
    # Cấu trúc GIỐNG HỆT train_edge_classifier_v2.py -> tương thích với
    # test_edge_classifier.py / test_model_report.py, không cần sửa gì.
    full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in full_model.features.parameters():
        p.requires_grad = False
    full_model.classifier = head  # gán head đã train vào đúng vị trí classifier
    full_model.classifier.load_state_dict(best_head_state)
    full_model = full_model.to(DEVICE)
    full_model.eval()

    torch.save({
        "model_state": full_model.state_dict(),
        "class_names": class_names,
    }, MODEL_OUT)
    print(f"Đã lưu model tại: {MODEL_OUT}")

    # ----------------------------
    # 6. Đo thông số edge (latency tổng: backbone + head, vì lúc inference
    # thực tế vẫn phải forward qua backbone cho ảnh mới)
    # ----------------------------
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = full_model(dummy_input)
    avg_latency_ms = (time.time() - start) / n_runs * 1000

    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)

    print(f"\n--- Thông số edge ---")
    print(f"Latency trung bình (1 ảnh, backbone + head): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    main()