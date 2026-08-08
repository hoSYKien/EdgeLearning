"""
KIẾN TRÚC: Frozen pretrained CNN -> cache embedding -> classifier tuyến tính nhẹ
           -> đánh giá bằng Repeated Stratified K-Fold Cross Validation.

CHẾ ĐỘ NHIỀU MODEL: khai báo DATASET_DIRS_LIST gồm N bộ dữ liệu (vd part1..part5),
script train N model ĐỘC LẬP trong 1 lần chạy, rồi in bảng SO SÁNH cuối cùng.
Backbone (phần nặng nhất) chỉ load 1 lần và dùng chung cho mọi part.

Pipeline TỔNG QUÁT cho bài toán phân loại ảnh ÍT DỮ LIỆU (few-shot):
OK/NG, defect PCB, phân loại màu, trái cây, vật thể...
Đổi bài toán = chỉ sửa phần CONFIG, KHÔNG cần đụng logic.

--------------------------------------------------------------------------
Ý TƯỞNG CỐT LÕI
--------------------------------------------------------------------------
1. Backbone CNN ĐÓNG BĂNG HOÀN TOÀN (eval mode, requires_grad=False).
2. Extract embedding 1 lần duy nhất, cache cả trong RAM lẫn ra ĐĨA
   -> đổi hyperparameter/K-Fold không phải extract lại (vài giây thay vì vài phút).
3. Dataset nhỏ -> KHÔNG chia train/val cố định. Dùng REPEATED Stratified K-Fold:
   K fold x N lần lặp với seed khác nhau. Rẻ gần như miễn phí vì embedding
   đã cache, và cho ước lượng ổn định hơn nhiều so với K-Fold 1 lần.
4. Head là classifier TUYẾN TÍNH, mặc định dùng LogisticRegression (lbfgs)
   -> hội tụ tới nghiệm tối ưu toàn cục, KHÔNG có epoch/lr/early-stop để chỉnh sai.
5. Sau khi đánh giá, train head cuối trên TOÀN BỘ dữ liệu để deploy.

--------------------------------------------------------------------------
NHỮNG ĐIỂM ĐÃ SỬA SO VỚI BẢN TRƯỚC (đọc kỹ, ảnh hưởng tới con số bạn tin)
--------------------------------------------------------------------------
[A] Backbone generic thật sự: hỗ trợ cả họ MobileNet (.features) lẫn họ
    ResNet (.children()[:-2]). embedding_dim SUY RA bằng dummy forward,
    không hard-code 960/576.
[B] BỎ early stopping trên chính tập val rồi báo cáo lại tập val đó.
    Đây là nguồn optimistic bias lớn nhất của bản cũ (chọn max của ~300 số
    đo nhiễu). Giờ head hội tụ xác định, không "chọn epoch đẹp nhất".
[C] Guard chia-cho-0 ở class_weights (bản cũ -> inf -> NaN loss âm thầm).
[D] Đo latency ĐÚNG: có warm-up + torch.cuda.synchronize(). Bản cũ đo trên
    GPU bất đồng bộ nên con số nhỏ hơn thực tế nhiều lần. Báo cáo p50/p95.
[E] Metric thật cho bài toán defect: per-class precision/recall/F1,
    khoảng tin cậy Wilson, quét ngưỡng theo recall mục tiêu, và phân tích
    "abstain" (bao nhiêu % ảnh phải chuyển người kiểm tra tay).
[F] Checkpoint lưu ĐỦ preprocess (size, mean, std, l2_normalize) để code
    inference không phải đoán rồi sai âm thầm.
[G] Deploy bằng nn.Module tường minh, không hack model.classifier / model.fc.
    Kèm export ONNX.
[H] Cảnh báo ảnh TRÙNG giữa các thư mục (gộp train/+val/ dễ gây leak).
[I] Extract embedding qua DataLoader nhiều worker + AMP -> nhanh hơn nhiều.

Cách chạy:
    python few_shot_pipeline.py
"""

import copy
import csv
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ==========================================================================
# CONFIG - CHỈ SỬA Ở ĐÂY KHI ÁP DỤNG CHO BÀI TOÁN MỚI
# ==========================================================================

# DANH SÁCH CÁC BỘ DỮ LIỆU - MỖI PHẦN TỬ = 1 MODEL RIÊNG.
# Mỗi phần tử là list các thư mục sẽ được GỘP lại rồi chia K-Fold.
# (Nếu bạn còn tách sẵn train/ và val/, cứ liệt kê cả 2 - script tự gộp.)
# Tên model lấy từ thư mục cha chung, vd ...\crop7\part1\train -> "part1".
DATASET_DIRS_LIST = [
    [
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART1\train",
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART1\val",
    ],
    [
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART2\train",
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART2\val",
    ],
    [
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART3\train",
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART3\val",
    ],
    [
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART4\train",
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART4\val",
    ],
    [
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART5\train",
        r"D:\TongHop\RTC Technologi\PCB\dataset14\PART5\val",
    ],
]

# Ghi đè cấu hình cho RIÊNG một số part. Bỏ trống {} = mọi part dùng chung config.
# Dùng khi một part khó hơn hẳn các part khác. Ví dụ:
#   PART_OVERRIDES = {
#       "part3": {"logreg_C": 0.1, "k_folds": 4},
#       "part5": {"augment": {"horizontal_flip": False, "rotation_degrees": 0,
#                             "color_jitter": True, "random_resized_crop": False}},
#   }
# Lưu ý: "augment" ghi đè TOÀN BỘ dict, phải liệt kê đủ các khoá.
PART_OVERRIDES = {}

# Thư mục gốc chứa kết quả. Mỗi part được lưu vào MODEL_ROOT/<tên part>/
MODEL_ROOT = r"D:\TongHop\RTC Technologi\PCB\model\model15"

# Nếu 1 part lỗi (thiếu thư mục, thiếu ảnh...) thì bỏ qua và chạy tiếp part sau,
# thay vì dừng cả mẻ. Lỗi được tổng hợp lại ở bảng cuối.
CONTINUE_ON_PART_ERROR = True

# "mobilenet_v2" | "mobilenet_v3_small" | "mobilenet_v3_large"
# "resnet18" | "resnet50" | "wide_resnet50_2" | "efficientnet_b0" | "convnext_tiny"
BACKBONE_NAME = "wide_resnet50_2"

IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Augmentation - bật/tắt theo đặc thù bài toán
AUGMENT_CONFIG = {
    "horizontal_flip": True,       # tắt nếu ảnh có chữ/ký hiệu định hướng
    "vertical_flip": False,
    "rotation_degrees": 10,        # 0 = tắt
    "color_jitter": True,          # PCB: giữ True để bù chênh lệch ánh sáng chụp.
                                   # TẮT nếu bài toán phân loại theo MÀU SẮC.
    "random_resized_crop": False,  # tắt nếu vị trí/bố cục vật thể quan trọng
}

AUGMENT_COPIES = 20      # số bản augment/ảnh, cache sẵn (augmentation offline)
L2_NORMALIZE = True      # L2-norm embedding trước classifier - thường tăng acc rõ

# --- Head ---
HEAD_TYPE = "logreg"     # "logreg" (khuyến nghị) | "torch"
LOGREG_C = 1.0           # nghịch đảo cường độ regularization. Nhỏ hơn = regularize mạnh hơn
LOGREG_MAX_ITER = 2000

TORCH_HEAD_EPOCHS = 200  # chỉ dùng khi HEAD_TYPE == "torch"
TORCH_HEAD_LR = 1e-3
TORCH_HEAD_DROPOUT = 0.3
TORCH_HEAD_BATCH = 256
TORCH_HEAD_WEIGHT_DECAY = 1e-4

# --- Đánh giá ---
K_FOLDS = 5
N_REPEATS = 5            # lặp lại K-Fold với seed khác nhau. Rẻ vì đã cache embedding.

# Class "lỗi/NG" để phân tích ngưỡng. None = tự đoán từ tên class.
DEFECT_CLASS_NAME = None

# --- Vận hành ---
CACHE_EMBEDDINGS = True  # cache ra đĩa, key = hash(file list + config)
NUM_WORKERS = 0          # Windows: 0 hoặc 2 an toàn nhất. Linux: 4-8 nhanh hơn.
BATCH_IMAGES = 16        # số ảnh GỐC mỗi batch khi extract (mỗi ảnh -> AUGMENT_COPIES bản)
USE_AMP = True           # float16 khi extract trên CUDA, nhanh ~2x
EXPORT_ONNX = True
VERBOSE_FOLDS = False    # True = in chi tiết từng fold

# --- Heatmap (CAM) - xem model NHÌN VÀO ĐÂU để ra quyết định ---
EXPORT_HEATMAPS = True
HEATMAP_MAX_IMAGES = 12   # tổng số ảnh trong 1 file heatmap
HEATMAP_MAX_WRONG = 8     # trong đó, tối đa bao nhiêu ảnh SAI (ưu tiên xem trước)
HEATMAP_TILE = 160        # kích thước 1 ô ảnh, pixel
HEATMAP_ALPHA = 0.5       # độ đậm của lớp màu phủ lên ảnh gốc

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa khi đổi bài toán
# ==========================================================================

BACKBONE_REGISTRY = {
    "mobilenet_v2": (models.mobilenet_v2, "MobileNet_V2_Weights"),
    "mobilenet_v3_small": (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),
    "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
}

DEFECT_NAME_HINTS = ("ng", "nok", "defect", "bad", "fail", "loi", "lỗi", "error", "reject")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_part_names(dirs_list):
    """Suy ra tên model từ thư mục cha chung của mỗi nhóm.
    [...\\crop7\\part1\\train, ...\\crop7\\part1\\val] -> "part1"
    Nếu trùng tên hoặc không suy ra được -> đặt "dataset_1", "dataset_2"..."""
    names, used = [], set()
    for i, dirs in enumerate(dirs_list):
        try:
            parents = {os.path.basename(os.path.dirname(os.path.normpath(d))) for d in dirs}
            name = parents.pop() if len(parents) == 1 else os.path.basename(
                os.path.normpath(os.path.commonpath([os.path.abspath(d) for d in dirs])))
        except Exception:
            name = ""
        if not name or name in used:
            name = f"dataset_{i + 1}"
        used.add(name)
        names.append(name)
    return names


def build_cfg(part_name):
    """Config hiệu lực cho 1 part = config chung + phần ghi đè riêng của part."""
    cfg = {
        "augment": dict(AUGMENT_CONFIG),
        "augment_copies": AUGMENT_COPIES,
        "l2_normalize": L2_NORMALIZE,
        "image_size": IMAGE_SIZE,
        "batch_images": BATCH_IMAGES,
        "head_type": HEAD_TYPE,
        "logreg_C": LOGREG_C,
        "logreg_max_iter": LOGREG_MAX_ITER,
        "torch_epochs": TORCH_HEAD_EPOCHS,
        "torch_lr": TORCH_HEAD_LR,
        "torch_dropout": TORCH_HEAD_DROPOUT,
        "torch_batch": TORCH_HEAD_BATCH,
        "torch_weight_decay": TORCH_HEAD_WEIGHT_DECAY,
        "k_folds": K_FOLDS,
        "n_repeats": N_REPEATS,
        "defect_class": DEFECT_CLASS_NAME,
    }
    ov = PART_OVERRIDES.get(part_name, {})
    unknown = set(ov) - set(cfg)
    if unknown:
        raise ValueError(f"PART_OVERRIDES['{part_name}'] có khoá không hợp lệ: {sorted(unknown)}. "
                         f"Khoá hợp lệ: {sorted(cfg)}")
    cfg.update(ov)
    return cfg


# --------------------------------------------------------------------------
# Dữ liệu
# --------------------------------------------------------------------------
def build_transforms(cfg, size, mean, std):
    aug_ops = [transforms.Resize((size, size))]
    if cfg.get("random_resized_crop"):
        aug_ops = [transforms.RandomResizedCrop(size, scale=(0.8, 1.0))]
    if cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if cfg.get("vertical_flip"):
        aug_ops.append(transforms.RandomVerticalFlip())
    if cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(cfg["rotation_degrees"]))
    if cfg.get("color_jitter"):
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    tail = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    clean_ops = [transforms.Resize((size, size))] + tail
    return transforms.Compose(aug_ops + tail), transforms.Compose(clean_ops)


def list_images(dataset_dirs):
    """Gộp ảnh từ nhiều thư mục thành 1 danh sách chung.
    Trả về: image_paths, labels, class_names."""
    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {d}")

    class_names = set()
    for d in dataset_dirs:
        for entry in os.scandir(d):
            if entry.is_dir():
                class_names.add(entry.name)
    class_names = sorted(class_names)
    if not class_names:
        raise RuntimeError(f"Không có thư mục con (class) nào trong: {dataset_dirs}")

    # Cảnh báo tên class chỉ khác nhau hoa/thường -> gần như chắc chắn là lỗi gõ
    lowered = {}
    for c in class_names:
        lowered.setdefault(c.lower(), []).append(c)
    for low, group in lowered.items():
        if len(group) > 1:
            print(f"  [CẢNH BÁO] Các class chỉ khác hoa/thường, đang bị coi là KHÁC NHAU: {group}")

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    image_paths, labels = [], []
    for d in dataset_dirs:
        for cls in class_names:
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [CHÚ Ý] Class '{cls}' không có trong {d}")
                continue
            n_before = len(image_paths)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])
            if len(image_paths) == n_before:
                print(f"  [CHÚ Ý] Thư mục rỗng (không có ảnh hợp lệ): {cls_dir}")

    if not image_paths:
        raise RuntimeError("Không tìm thấy ảnh nào.")
    return image_paths, labels, class_names


def warn_duplicate_images(image_paths):
    """Gộp train/ và val/ rất dễ có ảnh TRÙNG -> cùng 1 ảnh rơi vào 2 fold
    khác nhau -> rò rỉ dữ liệu -> accuracy ảo. Phải kiểm tra."""
    seen, dups = {}, []
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError as e:
            print(f"  [CẢNH BÁO] Không đọc được {p}: {e}")
            continue
        if h in seen:
            dups.append((p, seen[h]))
        else:
            seen[h] = p

    if dups:
        print(f"\n  [CẢNH BÁO NGHIÊM TRỌNG] Phát hiện {len(dups)} ảnh TRÙNG NỘI DUNG.")
        print("  Ảnh trùng rơi vào 2 fold khác nhau sẽ làm accuracy CAO GIẢ TẠO.")
        for a, b in dups[:10]:
            print(f"    {os.path.basename(a)}  ==  {os.path.basename(b)}")
        if len(dups) > 10:
            print(f"    ... và {len(dups) - 10} cặp nữa")
    else:
        print("  Không có ảnh trùng nội dung. OK.")
    return dups


class EmbedExtractDataset(Dataset):
    """Mỗi item trả về (AUGMENT_COPIES bản augment, 1 bản clean) của 1 ảnh gốc."""

    def __init__(self, paths, aug_tf, clean_tf, copies):
        self.paths, self.aug_tf, self.clean_tf, self.copies = paths, aug_tf, clean_tf, copies

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Không mở được ảnh {self.paths[i]}: {e}") from e
        aug = torch.stack([self.aug_tf(img) for _ in range(self.copies)])
        return aug, self.clean_tf(img)


def _worker_init(worker_id):
    s = RANDOM_SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------
def build_trunk(backbone_name, pretrained=True):
    """Trả về (trunk, embedding_dim). Trunk xuất feature map (B, D, H, W).

    Xử lý được cả 2 họ kiến trúc:
      - có .features (mobilenet, efficientnet, convnext, vgg...)
      - không có .features (resnet: conv1/bn1/layer1..4) -> cắt bỏ avgpool + fc
    embedding_dim SUY RA bằng dummy forward, không hard-code.
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone không hỗ trợ: {backbone_name}. "
            f"Chọn một trong: {sorted(BACKBONE_REGISTRY)}"
        )
    ctor, weights_enum_name = BACKBONE_REGISTRY[backbone_name]
    weights = getattr(models, weights_enum_name).IMAGENET1K_V1 if pretrained else None
    model = ctor(weights=weights)

    if hasattr(model, "features"):
        trunk = model.features
    else:
        trunk = nn.Sequential(*list(model.children())[:-2])

    trunk = trunk.to(DEVICE).eval()
    for p in trunk.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dim = trunk(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)).shape[1]
    return trunk, int(dim)


def pool_embed(feat, l2_normalize):
    x = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    return F.normalize(x, dim=1) if l2_normalize else x


class FewShotClassifier(nn.Module):
    """Model deploy tường minh: trunk (frozen) -> GAP -> [L2] -> Linear.

    Thay cho cách hack `model.classifier = head` của bản cũ - cách đó SAI với
    ResNet (ResNet dùng .fc, gán .classifier tạo attribute rác và forward vẫn
    chạy qua fc 1000-class gốc).
    """

    def __init__(self, trunk, embedding_dim, num_classes, l2_normalize):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(embedding_dim, num_classes)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, x):
        return self.head(pool_embed(self.trunk(x), self.l2_normalize))


# --------------------------------------------------------------------------
# Extract + cache embedding
# --------------------------------------------------------------------------
def _cache_key(image_paths, backbone_name, aug_cfg, copies, size, mean, std, seed):
    payload = {
        "files": [(os.path.abspath(p), os.path.getsize(p)) for p in image_paths],
        "backbone": backbone_name,
        "aug": aug_cfg,
        "copies": copies,
        "size": size,
        "mean": mean,
        "std": std,
        "seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


@torch.no_grad()
def precompute_all_embeddings(trunk, image_paths, aug_tf, clean_tf, copies, cfg, cache_path=None):
    """Với MỖI ảnh gốc tính sẵn:
      - aug_emb[i]:   (copies, D) - dùng khi ảnh rơi vào phần TRAIN của fold
      - clean_emb[i]: (D,)        - dùng khi ảnh rơi vào phần VAL của fold

    LƯU Ý VỀ AUGMENTATION: đây là augmentation OFFLINE - `copies` bản được cố
    định một lần, không sinh mới mỗi epoch. Đổi tốc độ lấy tính đa dạng.
    Tăng copies từ 20 lên 50 tốn 2.5x thời gian mà lợi ích giảm dần rất nhanh.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"  -> Nạp embedding từ cache: {cache_path}")
        return data["aug"], data["clean"]

    ds = EmbedExtractDataset(image_paths, aug_tf, clean_tf, copies)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_images"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=_worker_init if NUM_WORKERS > 0 else None,
    )

    amp_on = USE_AMP and DEVICE.type == "cuda"
    aug_out, clean_out = [], []
    done, total = 0, len(image_paths)
    t0 = time.time()

    for aug_batch, clean_batch in loader:
        b = aug_batch.shape[0]
        aug_flat = aug_batch.reshape(b * copies, *aug_batch.shape[2:]).to(DEVICE, non_blocking=True)
        clean_batch = clean_batch.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_on):
            a = pool_embed(trunk(aug_flat), cfg["l2_normalize"])
            c = pool_embed(trunk(clean_batch), cfg["l2_normalize"])

        aug_out.append(a.float().cpu().numpy().reshape(b, copies, -1))
        clean_out.append(c.float().cpu().numpy())

        done += b
        print(f"    {done}/{total} ảnh  ({time.time() - t0:.1f}s)", end="\r")

    print(" " * 60, end="\r")
    aug_emb = np.concatenate(aug_out, axis=0)      # (N, copies, D)
    clean_emb = np.concatenate(clean_out, axis=0)  # (N, D)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, aug=aug_emb, clean=clean_emb)
        print(f"  -> Đã lưu cache embedding: {cache_path}")
    return aug_emb, clean_emb


# --------------------------------------------------------------------------
# K-Fold
# --------------------------------------------------------------------------
def stratified_fold_assignment(labels, k, seed):
    """Chia index thành k fold, giữ tỉ lệ class đồng đều (stratified)."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    assign = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        offset = rng.randint(k)  # xoay điểm bắt đầu để các repeat khác nhau thật sự
        for i, idx in enumerate(idx_c):
            assign[idx] = (i + offset) % k
    return assign


# --------------------------------------------------------------------------
# Head
# --------------------------------------------------------------------------
def _check_all_classes_present(y, num_classes, context):
    counts = np.bincount(y, minlength=num_classes)
    missing = np.where(counts == 0)[0]
    if len(missing) > 0:
        raise ValueError(
            f"{context}: class index {missing.tolist()} KHÔNG có mẫu nào trong tập train.\n"
            f"  -> Giảm K_FOLDS, hoặc bổ sung ảnh cho class đó.\n"
            f"  (Bản code cũ sẽ tạo class_weight = inf -> NaN loss âm thầm ở đây.)"
        )
    return counts


def fit_head_logreg(X, y, num_classes, C, max_iter):
    """LogisticRegression: hội tụ tới nghiệm tối ưu toàn cục, không có
    epoch/lr/early-stop để chỉnh sai. Trả về (W, b) dạng numpy."""
    from sklearn.linear_model import LogisticRegression

    _check_all_classes_present(y, num_classes, "fit_head_logreg")
    clf = LogisticRegression(
        C=C, max_iter=max_iter, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X, y)

    if clf.coef_.shape[0] == 1:  # sklearn trả 1 hàng khi nhị phân
        # Tách đôi ĐỐI XỨNG: softmax([-z/2, +z/2]) == [1-sigmoid(z), sigmoid(z)]
        # -> xác suất y hệt cách đặt [0, z], nhưng KHÔNG để hàng nào toàn 0.
        # Quan trọng cho heatmap: hàng toàn 0 sẽ cho CAM phẳng lì, vô nghĩa.
        W = np.vstack([-clf.coef_[0] / 2, clf.coef_[0] / 2])
        b = np.array([-clf.intercept_[0] / 2, clf.intercept_[0] / 2])
    else:
        W, b = clf.coef_, clf.intercept_

    # sắp lại theo đúng thứ tự class index (clf.classes_ có thể không đủ/không sắp)
    W_full = np.zeros((num_classes, X.shape[1]))
    b_full = np.zeros(num_classes)
    for row, cls in enumerate(clf.classes_):
        W_full[cls], b_full[cls] = W[row], b[row]
    return W_full.astype(np.float32), b_full.astype(np.float32)


def fit_head_torch(X, y, num_classes, epochs, lr, dropout, batch_size, weight_decay, verbose=False):
    """Head torch, minibatch + shuffle, số epoch CỐ ĐỊNH, KHÔNG early-stop.

    Bản cũ chạy full-batch 1 step/epoch (300 epoch = 300 bước cập nhật -> thường
    chưa hội tụ) và early-stop trên chính tập val rồi báo cáo lại tập val đó
    (-> optimistic bias). Cả hai vấn đề đã bỏ.
    """
    counts = _check_all_classes_present(y, num_classes, "fit_head_torch")
    class_w = torch.tensor(len(y) / (num_classes * counts), dtype=torch.float32, device=DEVICE)

    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y, dtype=torch.long, device=DEVICE)

    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(X.shape[1], num_classes)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=class_w)

    n = len(yt)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = crit(head(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"      epoch {epoch+1:3d}/{epochs} | loss {tot/n:.4f}")

    lin = head[1]
    return lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy()


def fit_head(X, y, num_classes, cfg, verbose=False):
    if cfg["head_type"] == "logreg":
        return fit_head_logreg(X, y, num_classes, cfg["logreg_C"], cfg["logreg_max_iter"])
    if cfg["head_type"] == "torch":
        return fit_head_torch(X, y, num_classes, cfg["torch_epochs"], cfg["torch_lr"],
                              cfg["torch_dropout"], cfg["torch_batch"],
                              cfg["torch_weight_decay"], verbose)
    raise ValueError(f"head_type không hợp lệ: {cfg['head_type']}")


def predict_probs(W, b, X):
    logits = X @ W.T + b
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def wilson_interval(k, n, z=1.96):
    """Khoảng tin cậy 95% cho tỉ lệ. Với n nhỏ (few-shot) thì đây là cách
    đúng, chứ không phải p ± 1.96*sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_and_prf(cm, class_names):
    """Với dữ liệu mất cân bằng + bài toán defect, accuracy gần như vô dụng.
    Cái cần nhìn là precision/recall/F1 theo từng class."""
    n_cls = len(class_names)
    w = max(12, max(len(c) for c in class_names) + 2)

    print("\nConfusion matrix (hàng = THẬT, cột = ĐOÁN):")
    print(" " * 16 + "".join(f"{c[:w-2]:>{w}}" for c in class_names) + f"{'Tổng':>10}")
    for i, name in enumerate(class_names):
        print(f"{name[:15]:<16}" + "".join(f"{cm[i, j]:>{w}d}" for j in range(n_cls))
              + f"{cm[i].sum():>10d}")

    print("\nPrecision / Recall / F1 theo từng class:")
    print(f"{'Class':<16}{'Support':>9}{'Precision':>11}{'Recall':>10}{'F1':>9}"
          f"{'Recall CI 95%':>22}")
    print("-" * 77)
    f1s, precs, recs = [], [], []
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lo, hi = wilson_interval(tp, support)
        precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"{class_names[c][:15]:<16}{support:>9d}{prec:>10.2%}{rec:>10.2%}{f1:>9.3f}"
              f"{f'[{lo:.1%} - {hi:.1%}]':>22}")

    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total else 0.0
    lo, hi = wilson_interval(correct, total)
    print("-" * 77)
    print(f"{'Macro avg':<16}{total:>9d}{np.mean(precs):>10.2%}{np.mean(recs):>10.2%}"
          f"{np.mean(f1s):>9.3f}")
    print(f"{'Accuracy':<16}{total:>9d}{'':>10}{acc:>10.2%}")
    print(f"  -> Khoảng tin cậy 95% của accuracy: [{lo:.2%} - {hi:.2%}]")
    if total < 200:
        print(f"  -> CẢNH BÁO: chỉ {total} mẫu. Khoảng tin cậy rộng như trên nghĩa là")
        print(f"     chênh lệch vài % giữa các cấu hình gần như KHÔNG có ý nghĩa thống kê.")
    return {"accuracy": acc, "acc_ci": (lo, hi), "macro_f1": float(np.mean(f1s)),
            "macro_precision": float(np.mean(precs)), "macro_recall": float(np.mean(recs)),
            "per_class_recall": recs, "per_class_precision": precs, "per_class_f1": f1s}


def resolve_defect_class(class_names, configured):
    if configured is not None:
        if configured not in class_names:
            print(f"  [CHÚ Ý] DEFECT_CLASS_NAME='{configured}' không có trong {class_names}")
            return None
        return class_names.index(configured)
    for i, c in enumerate(class_names):
        if c.strip().lower() in DEFECT_NAME_HINTS:
            return i
    return None


def threshold_analysis(probs, y_true, defect_idx, class_names):
    """Với kiểm tra PCB, chi phí BỎ SÓT NG khác hẳn báo nhầm OK.
    argmax luôn trả lời -> không kiểm soát được đánh đổi này.
    Bảng dưới cho biết: muốn recall NG đạt X% thì phải đặt ngưỡng bao nhiêu
    và phải chịu bao nhiêu báo động giả."""
    name = class_names[defect_idx]
    score = probs[:, defect_idx]
    is_defect = (y_true == defect_idx)
    n_def, n_ok = int(is_defect.sum()), int((~is_defect).sum())
    if n_def == 0 or n_ok == 0:
        return

    print("\n" + "-" * 77)
    print(f"PHÂN TÍCH NGƯỠNG cho class lỗi = '{name}'  ({n_def} lỗi / {n_ok} không lỗi)")
    print(f"{'Recall mục tiêu':<18}{'Ngưỡng':>10}{'Recall':>10}{'Precision':>12}"
          f"{'Bỏ sót':>9}{'Báo nhầm':>11}")
    print("-" * 77)
    for target in (0.90, 0.95, 0.98, 0.99, 1.00):
        cand = sorted(score[is_defect])
        need = math.ceil(target * n_def)
        if need > n_def:
            continue
        thr = cand[n_def - need]  # ngưỡng thấp nhất đạt được recall mục tiêu
        pred = score >= thr
        tp = int((pred & is_defect).sum())
        fp = int((pred & ~is_defect).sum())
        fn = n_def - tp
        rec = tp / n_def
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"{target:<18.0%}{thr:>10.4f}{rec:>10.2%}{prec:>12.2%}{fn:>9d}{fp:>11d}")

    print("\nPHÂN TÍCH 'ABSTAIN' - chuyển ảnh không chắc chắn cho người kiểm tra tay:")
    print(f"{'Độ chính xác cần':<20}{'Ngưỡng tin cậy':>16}{'Tự động xử lý':>16}"
          f"{'Cần review tay':>16}")
    print("-" * 77)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true)
    order = np.argsort(-conf)                       # tự tin nhất -> ít tự tin nhất
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    for target in (0.98, 0.99, 0.995, 1.00):
        ok = np.where(cum_acc >= target)[0]
        if len(ok) == 0:
            print(f"{target:<20.1%}{'-':>16}{'không đạt được':>16}{'':>16}")
            continue
        cut = int(ok.max()) + 1                     # coverage lớn nhất còn đạt mục tiêu
        print(f"{target:<20.1%}{conf[order[cut-1]]:>16.4f}{cut/len(y_true):>15.1%}"
              f"{1 - cut/len(y_true):>16.1%}")


def export_detail_csv(path, image_paths, y_true, pred, probs, fold_of, class_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["File", "Duong_dan", "Fold", "Class_that", "Class_doan", "Dung_Sai",
                     "Do_tin_cay"] + [f"Xac_suat_{c}" for c in class_names])
        rank = np.array([probs[i][pred[i]] if pred[i] != y_true[i] else -1.0
                         for i in range(len(y_true))])
        order = np.argsort(-rank)
        for i in order:  # ảnh SAI với độ tin cậy cao nhất lên đầu - đáng xem nhất
            wr.writerow([
                os.path.basename(image_paths[i]), image_paths[i], fold_of[i] + 1,
                class_names[y_true[i]], class_names[pred[i]],
                "Dung" if pred[i] == y_true[i] else "SAI",
                f"{probs[i][pred[i]]:.4f}",
            ] + [f"{p:.4f}" for p in probs[i]])
    print(f"\nĐã xuất báo cáo chi tiết: {path}")


def print_worst_mistakes(image_paths, y_true, pred, probs, class_names, top_n=15):
    wrong = [i for i in range(len(y_true)) if pred[i] != y_true[i]]
    if not wrong:
        print("\nKhông có ảnh nào bị dự đoán sai (out-of-fold).")
        return
    wrong.sort(key=lambda i: -probs[i][pred[i]])
    print(f"\n{len(wrong)} ảnh SAI - {min(top_n, len(wrong))} ca tự tin nhất (đáng kiểm tra nhãn):")
    print(f"{'File':<34}{'Thật':<14}{'Đoán':<14}{'Tin cậy':>9}")
    print("-" * 71)
    for i in wrong[:top_n]:
        print(f"{os.path.basename(image_paths[i])[:33]:<34}{class_names[y_true[i]][:13]:<14}"
              f"{class_names[pred[i]][:13]:<14}{probs[i][pred[i]]:>9.1%}")


# --------------------------------------------------------------------------
# Heatmap / CAM
# --------------------------------------------------------------------------
def _jet(x):
    """Colormap kiểu jet: xanh dương (thấp) -> xanh lá -> vàng -> đỏ (cao).
    Tự cài bằng numpy để KHÔNG phải phụ thuộc matplotlib."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


@torch.no_grad()
def compute_cam(trunk, W, img_tensor, l2_normalize):
    """Class Activation Map (Zhou et al. 2016).

    Kiến trúc ở đây là trunk -> GAP -> Linear, đúng dạng CAM áp dụng được
    TRỰC TIẾP, không cần gradient:
        logit_c = W[c] . GAP(A) = (1/HW) * sum_{h,w} sum_d W[c,d] * A[d,h,w]
    -> CAM_c[h,w] = sum_d W[c,d] * A[d,h,w] chính là phần đóng góp của từng
    vị trí không gian vào logit của class c.

    L2-normalize chỉ chia embedding cho một số dương -> KHÔNG đổi hình dạng
    bản đồ nhiệt, chỉ đổi thang giá trị. Nên CAM vẫn đúng.

    Trả về (num_classes, h, w) - giá trị THÔ, có thể âm (bằng chứng NGƯỢC lại).
    """
    feat = trunk(img_tensor.unsqueeze(0).to(DEVICE))[0]        # (D, h, w)
    Wt = torch.as_tensor(W, dtype=feat.dtype, device=feat.device)  # (C, D)
    cam = torch.einsum("cd,dhw->chw", Wt, feat)
    return cam.float().cpu().numpy()


def _cam_to_tile(cam_2d, base_img, tile, alpha):
    """Chuẩn hoá 1 bản đồ CAM rồi phủ màu lên ảnh gốc."""
    lo, hi = float(cam_2d.min()), float(cam_2d.max())
    norm = (cam_2d - lo) / (hi - lo) if hi > lo else np.zeros_like(cam_2d)
    heat = Image.fromarray(_jet(norm)).resize((tile, tile), Image.BICUBIC)
    return Image.blend(base_img, heat, alpha), (lo, hi)


def _pick_heatmap_samples(y_true, pred, probs, max_total, max_wrong, num_classes):
    """Ưu tiên ảnh SAI với độ tin cậy cao nhất - đó là chỗ model học nhầm,
    xem heatmap ở đây mới có giá trị chẩn đoán. Sau đó bù thêm vài ảnh ĐÚNG
    của mỗi class để có mốc so sánh 'trông thế nào là bình thường'."""
    wrong = [i for i in range(len(y_true)) if pred[i] != y_true[i]]
    wrong.sort(key=lambda i: -probs[i][pred[i]])
    chosen = wrong[:max_wrong]

    per_class = max(1, (max_total - len(chosen)) // max(num_classes, 1))
    for c in range(num_classes):
        ok = [i for i in range(len(y_true)) if y_true[i] == c and pred[i] == c]
        ok.sort(key=lambda i: -probs[i][pred[i]])
        chosen += ok[:per_class]

    seen, out = set(), []
    for i in chosen:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:max_total]


def _resolve_font(size):
    """Font mặc định của PIL là bitmap ASCII -> chữ có dấu tiếng Việt sẽ ra
    ô vuông. Tìm một font TrueType có sẵn trên máy trước.
    Trả về (font, hỗ_trợ_unicode)."""
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",           # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size), True
            except Exception:
                pass
    for name in ("DejaVuSans.ttf", "arial.ttf"):  # để PIL tự tìm trong system font
        try:
            return ImageFont.truetype(name, size), True
        except Exception:
            pass
    try:
        return ImageFont.load_default(size), False
    except TypeError:                              # Pillow < 10.1
        return ImageFont.load_default(), False


def _deaccent(text):
    """Bỏ dấu tiếng Việt - chỉ dùng khi không tìm được font TrueType nào,
    để nhãn còn đọc được thay vì thành một dãy ô vuông."""
    import unicodedata

    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(ch))


def export_heatmaps(path, trunk, W, image_paths, y_true, pred, probs, class_names,
                    clean_tf, cfg, part_name):
    """Xuất 1 file PNG: mỗi hàng = 1 ảnh, các cột = ảnh gốc + CAM của TỪNG class.

    Đọc bảng này thế nào:
      - Cột của class ĐÚNG: vùng đỏ nên nằm trên khuyết tật thật. Nếu đỏ nằm
        ở mép ảnh / nền / vết bẩn không liên quan -> model đang bắt nhầm
        đặc trưng, thêm dữ liệu cũng khó cứu.
      - So sánh cột 'đoán' với cột 'thật' ở các hàng SAI để biết model bị
        cái gì đánh lừa.
    """
    from PIL import ImageDraw

    idxs = _pick_heatmap_samples(y_true, pred, probs, HEATMAP_MAX_IMAGES,
                                 HEATMAP_MAX_WRONG, len(class_names))
    if not idxs:
        return None

    tile, alpha = HEATMAP_TILE, HEATMAP_ALPHA
    n_col = 1 + len(class_names)
    pad, head_h, cap_h = 6, 26, 22
    grid_w = n_col * tile + (n_col + 1) * pad
    grid_h = head_h + len(idxs) * (tile + cap_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font, uni = _resolve_font(14)
    font_s, _ = _resolve_font(12)
    txt = (lambda x: x) if uni else _deaccent
    if not uni:
        print("  [CHÚ Ý] Không tìm thấy font TrueType -> nhãn trong ảnh sẽ bị bỏ dấu.")

    titles = ["Ảnh gốc"] + [f"CAM: {c}" for c in class_names]
    for j, t in enumerate(titles):
        draw.text((pad + j * (tile + pad) + 4, 6), txt(t), fill=(0, 0, 0), font=font)

    flat_maps = 0
    for row, i in enumerate(idxs):
        y0 = head_h + row * (tile + cap_h + pad)
        img = Image.open(image_paths[i]).convert("RGB")
        base = img.resize((tile, tile), Image.BICUBIC)
        canvas.paste(base, (pad, y0))

        cam = compute_cam(trunk, W, clean_tf(img), cfg["l2_normalize"])
        for c in range(len(class_names)):
            overlay, (lo, hi) = _cam_to_tile(cam[c], base, tile, alpha)
            canvas.paste(overlay, (pad + (c + 1) * (tile + pad), y0))
            if hi - lo < 1e-6:
                flat_maps += 1
            # viền đỏ cho cột model ĐOÁN, viền xanh cho cột NHÃN THẬT
            box = [pad + (c + 1) * (tile + pad) - 2, y0 - 2,
                   pad + (c + 1) * (tile + pad) + tile + 1, y0 + tile + 1]
            if c == pred[i]:
                draw.rectangle(box, outline=(220, 30, 30), width=2)
            elif c == y_true[i]:
                draw.rectangle(box, outline=(30, 140, 220), width=2)

        ok = pred[i] == y_true[i]
        cap = (f"{os.path.basename(image_paths[i])[:38]}  |  thật: {class_names[y_true[i]]}"
               f"  |  đoán: {class_names[pred[i]]} ({probs[i][pred[i]]:.0%})  |  "
               f"{'ĐÚNG' if ok else 'SAI'}")
        draw.text((pad + 2, y0 + tile + 4), txt(cap), font=font_s,
                  fill=(20, 120, 20) if ok else (200, 20, 20))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)
    n_wrong = sum(1 for i in idxs if pred[i] != y_true[i])
    print(f"Đã xuất heatmap ({len(idxs)} ảnh, {n_wrong} ảnh SAI): {path}")
    print("  Viền ĐỎ = cột class model đoán, viền XANH = cột class nhãn thật.")
    if flat_maps:
        print(f"  [CHÚ Ý] {flat_maps} bản đồ gần như phẳng -> vùng ảnh đó không có "
              f"đặc trưng nào nổi bật với model.")
    return path


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
@torch.no_grad()
def benchmark_latency(model, device, n_warmup=20, n_runs=100):
    """Bản cũ đo SAI: CUDA chạy bất đồng bộ, vòng lặp chỉ enqueue lệnh rồi
    time.time() trả về ngay -> số nhỏ hơn thực tế nhiều lần. Phải có warm-up
    và synchronize()."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"mean": t.mean(), "p50": np.percentile(t, 50), "p95": np.percentile(t, 95)}


# ==========================================================================
# CHẠY 1 PART
# ==========================================================================
def run_one_part(part_name, dataset_dirs, trunk, embedding_dim, part_idx, part_total):
    """Train + đánh giá + lưu model cho MỘT bộ dữ liệu.
    Trả về dict tóm tắt để dựng bảng so sánh cuối cùng."""
    cfg = build_cfg(part_name)
    model_dir = os.path.join(MODEL_ROOT, part_name)
    os.makedirs(model_dir, exist_ok=True)
    model_out = os.path.join(model_dir, f"fewshot_{BACKBONE_NAME}_{part_name}.pt")

    print("\n" + "#" * 77)
    print(f"# PART {part_idx}/{part_total}: {part_name}")
    print("#" * 77)
    if PART_OVERRIDES.get(part_name):
        print(f"Config ghi đè riêng cho part này: {PART_OVERRIDES[part_name]}")

    # --- Dữ liệu ---
    image_paths, labels, class_names = list_images(dataset_dirs)
    labels_np = np.array(labels)
    num_classes = len(class_names)
    counts = np.bincount(labels_np, minlength=num_classes)
    n = len(image_paths)
    print(f"\nSố ảnh: {n}   |   Class: {dict(zip(class_names, counts.tolist()))}")

    k_folds, n_repeats = cfg["k_folds"], cfg["n_repeats"]
    if counts.min() < k_folds:
        raise ValueError(
            f"[{part_name}] Class '{class_names[int(counts.argmin())]}' chỉ có {counts.min()} ảnh "
            f"< k_folds={k_folds}. Giảm K_FOLDS, hoặc đặt PART_OVERRIDES['{part_name}'] "
            f"= {{'k_folds': {int(counts.min())}}}, hoặc bổ sung ảnh."
        )
    if counts.max() / max(counts.min(), 1) > 5:
        print(f"  [CHÚ Ý] Mất cân bằng {counts.max()/counts.min():.1f}:1 -> nhìn F1/recall, "
              f"đừng nhìn accuracy.")

    print("Kiểm tra ảnh trùng nội dung...")
    n_dups = len(warn_duplicate_images(image_paths))

    # --- Embedding (backbone dùng chung, chỉ extract lại phần ảnh của part này) ---
    aug_tf, clean_tf = build_transforms(cfg["augment"], cfg["image_size"], NORM_MEAN, NORM_STD)
    copies = cfg["augment_copies"]

    cache_path = None
    if CACHE_EMBEDDINGS:
        key = _cache_key(image_paths, BACKBONE_NAME, cfg["augment"], copies,
                         cfg["image_size"], NORM_MEAN, NORM_STD, RANDOM_SEED)
        cache_path = os.path.join(MODEL_ROOT, "emb_cache", f"{part_name}_{key}.npz")

    print(f"Extract embedding cho {n} ảnh ({copies} bản augment/ảnh)...")
    t0 = time.time()
    aug_emb, clean_emb = precompute_all_embeddings(
        trunk, image_paths, aug_tf, clean_tf, copies, cfg, cache_path
    )
    print(f"  -> Xong sau {time.time() - t0:.1f}s   |   aug {aug_emb.shape}, clean {clean_emb.shape}")

    # --- Repeated Stratified K-Fold ---
    print(f"\n--- Repeated Stratified K-Fold: K={k_folds} x {n_repeats} lần "
          f"= {k_folds * n_repeats} lượt train ---")

    prob_sum = np.zeros((n, num_classes))
    fold_of = np.zeros(n, dtype=int)
    repeat_accs, all_fold_accs, train_accs = [], [], []

    for rep in range(n_repeats):
        assign = stratified_fold_assignment(labels_np, k_folds, seed=RANDOM_SEED + rep * 1000)
        rep_pred = np.zeros(n, dtype=int)

        for fold in range(k_folds):
            val_idx = np.where(assign == fold)[0]
            tr_idx = np.where(assign != fold)[0]

            X_tr = aug_emb[tr_idx].reshape(-1, embedding_dim)
            y_tr = np.repeat(labels_np[tr_idx], copies)
            X_va, y_va = clean_emb[val_idx], labels_np[val_idx]

            W, b = fit_head(X_tr, y_tr, num_classes, cfg)
            p_va = predict_probs(W, b, X_va)

            prob_sum[val_idx] += p_va
            rep_pred[val_idx] = p_va.argmax(axis=1)
            if rep == 0:
                fold_of[val_idx] = fold

            acc = float((p_va.argmax(axis=1) == y_va).mean())
            all_fold_accs.append(acc)
            train_accs.append(float((predict_probs(W, b, X_tr).argmax(axis=1) == y_tr).mean()))
            if VERBOSE_FOLDS:
                print(f"    Repeat {rep+1} Fold {fold+1}: {len(tr_idx)} train / "
                      f"{len(val_idx)} val -> val acc {acc:.2%}")

        rep_acc = float((rep_pred == labels_np).mean())
        repeat_accs.append(rep_acc)
        print(f"  Lần lặp {rep+1}/{n_repeats}: OOF accuracy = {rep_acc:.2%}")

    repeat_accs = np.array(repeat_accs)
    all_fold_accs = np.array(all_fold_accs)
    tr_acc = float(np.mean(train_accs))
    gap = tr_acc - all_fold_accs.mean()

    print(f"\n>>> OOF accuracy: {repeat_accs.mean():.2%} ± {repeat_accs.std():.2%} "
          f"(std qua {n_repeats} lần lặp)")
    print(f">>> Accuracy từng fold: {all_fold_accs.mean():.2%} ± {all_fold_accs.std():.2%} "
          f"(qua {len(all_fold_accs)} fold)")
    print(f">>> Train accuracy TB: {tr_acc:.2%}   (chênh lệch train-val: {gap:+.2%})")
    if gap > 0.15:
        print(f"    -> Train >> val: OVERFIT. Đặt PART_OVERRIDES['{part_name}'] "
              f"= {{'logreg_C': 0.1}} để regularize mạnh hơn.")
    elif gap < -0.05:
        print("    -> Train < val: augmentation ĐANG QUÁ MẠNH, tạo ảnh khó hơn ảnh thật.")
        print("       Giảm rotation_degrees / tắt color_jitter / tắt random_resized_crop.")
    else:
        print("    -> Chênh lệch hợp lý.")

    # --- Báo cáo ---
    probs_avg = prob_sum / n_repeats
    pred = probs_avg.argmax(axis=1)
    cm = confusion_matrix(labels_np, pred, num_classes)

    print(f"\n--- Báo cáo out-of-fold: {part_name} ---")
    m = print_confusion_and_prf(cm, class_names)

    defect_idx = resolve_defect_class(class_names, cfg["defect_class"])
    defect_recall = None
    if defect_idx is not None:
        defect_recall = m["per_class_recall"][defect_idx]
        threshold_analysis(probs_avg, labels_np, defect_idx, class_names)
    else:
        print("\n  [CHÚ Ý] Không xác định được class 'lỗi' -> bỏ qua phân tích ngưỡng.")
        print("  Đặt DEFECT_CLASS_NAME = '<tên class NG>' để bật phần này.")

    print_worst_mistakes(image_paths, labels_np, pred, probs_avg, class_names)
    export_detail_csv(os.path.join(model_dir, f"oof_report_{part_name}.csv"),
                      image_paths, labels_np, pred, probs_avg, fold_of, class_names)

    # --- Model sản xuất ---
    print(f"\n--- Train model cuối cùng cho '{part_name}' trên toàn bộ {n} ảnh ---")
    X_all = aug_emb.reshape(-1, embedding_dim)
    y_all = np.repeat(labels_np, copies)
    W, b = fit_head(X_all, y_all, num_classes, cfg, verbose=VERBOSE_FOLDS)
    final_train_acc = float((predict_probs(W, b, X_all).argmax(axis=1) == y_all).mean())
    print(f"Train accuracy model cuối: {final_train_acc:.2%}  (TB các fold: {tr_acc:.2%})")

    model = FewShotClassifier(trunk, embedding_dim, num_classes, cfg["l2_normalize"])
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(W))
        model.head.bias.copy_(torch.from_numpy(b))
    model = model.to(DEVICE).eval()

    torch.save({
        "model_state": model.state_dict(),
        "part_name": part_name,
        "class_names": class_names,
        "backbone_name": BACKBONE_NAME,
        "embedding_dim": embedding_dim,
        "head_type": cfg["head_type"],
        # Đủ thông tin để code inference KHÔNG phải đoán preprocess.
        "preprocess": {
            "image_size": cfg["image_size"],
            "resize": "Resize((H,W)) - KHÔNG giữ tỉ lệ",
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
            "l2_normalize": cfg["l2_normalize"],
            "channel_order": "RGB",
        },
        "metrics": {
            "oof_accuracy": m["accuracy"],
            "oof_acc_ci95": list(m["acc_ci"]),
            "macro_f1": m["macro_f1"],
            "oof_acc_mean_over_repeats": float(repeat_accs.mean()),
            "oof_acc_std_over_repeats": float(repeat_accs.std()),
            "fold_acc_mean": float(all_fold_accs.mean()),
            "fold_acc_std": float(all_fold_accs.std()),
            "train_val_gap": gap,
            "confusion_matrix": cm.tolist(),
            "n_images": n,
            "class_counts": counts.tolist(),
            "duplicate_pairs": n_dups,
        },
        "cfg": cfg,
        "dataset_dirs": list(dataset_dirs),
        "torch_version": torch.__version__,
    }, model_out)
    print(f"Đã lưu model: {model_out}")

    if EXPORT_ONNX:
        export_onnx(model, model_out, cfg["image_size"])

    # --- Heatmap: CAM lấy từ W của MODEL CUỐI, còn nhãn đúng/sai là kết quả
    # OUT-OF-FOLD (không thiên vị). Hai thứ này khớp nhau ở hầu hết ảnh; chỗ
    # lệch nhau chính là ảnh nằm sát ranh giới quyết định - rất đáng xem.
    heatmap_path = None
    if EXPORT_HEATMAPS:
        try:
            heatmap_path = export_heatmaps(
                os.path.join(model_dir, f"heatmap_{part_name}.png"),
                trunk, W, image_paths, labels_np, pred, probs_avg, class_names,
                clean_tf, cfg, part_name)
        except Exception as e:
            print(f"  [CHÚ Ý] Xuất heatmap thất bại ({type(e).__name__}: {e}). Bỏ qua.")

    return {
        "part": part_name,
        "n_images": n,
        "classes": class_names,
        "class_counts": counts.tolist(),
        "oof_acc": m["accuracy"],
        "acc_ci": m["acc_ci"],
        "macro_f1": m["macro_f1"],
        "defect_recall": defect_recall,
        "acc_std_repeats": float(repeat_accs.std()),
        "train_val_gap": gap,
        "duplicates": n_dups,
        "model_path": model_out,
        "heatmap_path": heatmap_path,
        "model": model,
        "cfg": cfg,
    }


def export_onnx(model, model_out, image_size):
    onnx_path = model_out.replace(".pt", ".onnx")
    dummy = torch.randn(1, 3, image_size, image_size, device=DEVICE)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=13)
    # torch>=2.5 mặc định dùng exporter dynamo (cần onnxscript). Nếu thiếu,
    # thử lại với exporter TorchScript cũ.
    last = ""
    for attempt in ({"dynamo": False}, {}):
        try:
            torch.onnx.export(model, dummy, onnx_path, **kwargs, **attempt)
            print(f"Đã export ONNX: {onnx_path}")
            return
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    print(f"  [CHÚ Ý] Export ONNX thất bại: {last}")
    if "onnxscript" in last:
        print("     -> Chạy: pip install onnxscript   (torch mới cần gói này để export)")
    print("     Bỏ qua cũng được, file .pt vẫn dùng bình thường.")


# ==========================================================================
# BẢNG SO SÁNH CÁC PART
# ==========================================================================
def print_summary(results, failures):
    print("\n" + "=" * 100)
    print("BẢNG SO SÁNH TẤT CẢ CÁC MODEL")
    print("=" * 100)

    if results:
        print(f"{'Part':<14}{'Ảnh':>6}{'Class':>7}{'OOF acc':>10}{'KTC 95%':>20}"
              f"{'Macro F1':>10}{'Recall NG':>11}{'Gap tr-val':>12}{'Trùng':>7}")
        print("-" * 100)
        for r in results:
            ci = f"[{r['acc_ci'][0]:.1%}-{r['acc_ci'][1]:.1%}]"
            dr = f"{r['defect_recall']:.1%}" if r["defect_recall"] is not None else "-"
            print(f"{r['part'][:13]:<14}{r['n_images']:>6}{len(r['classes']):>7}"
                  f"{r['oof_acc']:>10.2%}{ci:>20}{r['macro_f1']:>10.3f}{dr:>11}"
                  f"{r['train_val_gap']:>+12.1%}{r['duplicates']:>7}")

        accs = np.array([r["oof_acc"] for r in results])
        print("-" * 100)
        print(f"{'TRUNG BÌNH':<14}{sum(r['n_images'] for r in results):>6}"
              f"{'':>7}{accs.mean():>10.2%}")

        # Cảnh báo các part lệch hẳn so với phần còn lại
        if len(accs) >= 3:
            med = float(np.median(accs))
            weak = [r for r in results if r["oof_acc"] < med - 0.10]
            if weak:
                print(f"\n  [CHÚ Ý] Part yếu hơn trung vị (={med:.1%}) trên 10 điểm %: "
                      f"{', '.join(r['part'] for r in weak)}")
                print("  -> Kiểm tra: số ảnh có đủ không, nhãn có sạch không, "
                      "hay vùng crop này vốn khó hơn.")

        # Tập class không giống nhau giữa các part là điều CẦN biết trước khi ghép
        # các model lại thành 1 hệ thống inference.
        class_sets = {tuple(r["classes"]) for r in results}
        if len(class_sets) > 1:
            print("\n  [CHÚ Ý] Các part KHÔNG cùng tập class:")
            for r in results:
                print(f"    {r['part']:<14} {r['classes']}")
            print("  -> Code inference phải đọc class_names TỪNG model, không dùng chung.")

        dup_total = sum(r["duplicates"] for r in results)
        if dup_total:
            print(f"\n  [CẢNH BÁO] Tổng {dup_total} cặp ảnh trùng trong các part. "
                  f"Accuracy ở trên đang CAO GIẢ TẠO cho những part đó.")

    if failures:
        print(f"\n{'-' * 100}")
        print(f"{len(failures)} PART LỖI (không train được):")
        for name, err in failures:
            print(f"  {name:<14} {err}")

    # Xuất CSV tổng hợp
    if results:
        path = os.path.join(MODEL_ROOT, "summary_all_parts.csv")
        os.makedirs(MODEL_ROOT, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["Part", "So_anh", "Cac_class", "So_anh_moi_class", "OOF_accuracy",
                         "KTC95_duoi", "KTC95_tren", "Macro_F1", "Recall_lop_loi",
                         "Std_qua_cac_lan_lap", "Gap_train_val", "Cap_anh_trung", "Model_path",
                         "Heatmap_path"])
            for r in results:
                wr.writerow([r["part"], r["n_images"], "|".join(r["classes"]),
                             "|".join(map(str, r["class_counts"])), f"{r['oof_acc']:.4f}",
                             f"{r['acc_ci'][0]:.4f}", f"{r['acc_ci'][1]:.4f}",
                             f"{r['macro_f1']:.4f}",
                             f"{r['defect_recall']:.4f}" if r["defect_recall"] is not None else "",
                             f"{r['acc_std_repeats']:.4f}", f"{r['train_val_gap']:.4f}",
                             r["duplicates"], r["model_path"],
                             r.get("heatmap_path") or ""])
        print(f"\nĐã xuất bảng tổng hợp: {path}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    set_seed(RANDOM_SEED)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    part_names = derive_part_names(DATASET_DIRS_LIST)

    print("=" * 100)
    print(f"Device: {DEVICE}   |   Backbone: {BACKBONE_NAME}   |   Head: {HEAD_TYPE}")
    print(f"Số model sẽ train: {len(DATASET_DIRS_LIST)}  ->  {', '.join(part_names)}")
    print("=" * 100)

    # Backbone load MỘT LẦN và dùng chung cho mọi part - đây là phần nặng nhất.
    # An toàn vì trunk đã frozen + eval, không part nào sửa được nó.
    t_all = time.time()
    trunk, embedding_dim = build_trunk(BACKBONE_NAME)
    print(f"Đã load backbone 1 lần, dùng chung cho tất cả part. "
          f"Embedding dim = {embedding_dim}")

    results, failures = [], []
    for i, (name, dirs) in enumerate(zip(part_names, DATASET_DIRS_LIST), start=1):
        try:
            results.append(run_one_part(name, dirs, trunk, embedding_dim, i, len(DATASET_DIRS_LIST)))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((name, msg))
            print(f"\n  [LỖI] Part '{name}' thất bại -> {msg}")
            if not CONTINUE_ON_PART_ERROR:
                raise
            print("  Bỏ qua part này, chạy tiếp part sau "
                  "(đặt CONTINUE_ON_PART_ERROR = False nếu muốn dừng ngay).")

    print_summary(results, failures)

    # --- Benchmark: kiến trúc giống nhau nên chỉ cần đo 1 lần ---
    if results:
        print("\n" + "=" * 100)
        print("THÔNG SỐ EDGE (mọi part cùng kiến trúc -> chỉ đo 1 lần)")
        print("=" * 100)
        model = results[0]["model"]
        size_mb = os.path.getsize(results[0]["model_path"]) / (1024 * 1024)
        print(f"Kích thước 1 file model : {size_mb:.2f} MB "
              f"(x{len(results)} part = {size_mb * len(results):.1f} MB tổng)")

        devices = [torch.device("cpu")] if DEVICE.type == "cpu" else [DEVICE, torch.device("cpu")]
        for dev in devices:
            # deepcopy khi đo trên device khác, tránh kéo trunk DÙNG CHUNG sang CPU
            target = model if dev == DEVICE else copy.deepcopy(model)
            s = benchmark_latency(target, dev)
            print(f"Latency 1 ảnh trên {str(dev):<5}: mean {s['mean']:.2f} ms | "
                  f"p50 {s['p50']:.2f} ms | p95 {s['p95']:.2f} ms")
            if dev != DEVICE:
                del target
        n_parts = len(results)
        s_cpu = benchmark_latency(copy.deepcopy(model), torch.device("cpu"), n_runs=30)
        print(f"\nChạy TUẦN TỰ cả {n_parts} model trên 1 ảnh (CPU): "
              f"~{s_cpu['mean'] * n_parts:.0f} ms")
        print("  -> Nếu quá chậm cho nhịp dây chuyền: gộp {n} crop thành 1 batch, hoặc "
              "dùng chung backbone và chỉ chạy {n} head (nhanh gần bằng 1 model).".format(n=n_parts))
        if DEVICE.type == "cuda":
            print("  -> Thiết bị edge thường là CPU/ARM: hãy tin con số CPU, không phải GPU.")

    print(f"\nTổng thời gian: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()