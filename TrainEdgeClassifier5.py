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
4. Sau khi đánh giá xong bằng K-Fold, train 1 model "sản xuất" cuối cùng
   trên TOÀN BỘ dữ liệu (không giữ lại val) để deploy.

--------------------------------------------------------------------------
MỚI: MULTI-CROP CHO LỖI NHỎ, VỊ TRÍ THAY ĐỔI (thay cho "chia đôi ảnh")
--------------------------------------------------------------------------
Vấn đề gốc rễ khi lỗi nhỏ + nằm ở góc/rìa ảnh:
  (a) Global Average Pooling ở cuối backbone làm LOÃNG tín hiệu nhỏ (lỗi
      chỉ chiếm vài % diện tích, bị hòa lẫn vào phần feature "bình thường").
  (b) Resize cả ảnh về 224x224 làm mất chi tiết nếu ảnh gốc lớn (>800px).
  (c) RandomResizedCrop trước đây được áp SAU khi đã resize về 224 -> "zoom"
      vào ảnh đã mất chi tiết, không cứu vãn được gì, thậm chí có thể vô tình
      cắt mất hẳn vùng lỗi ở góc trong khi vẫn giữ nhãn NG (nhiễu nhãn).

Vì VỊ TRÍ LỖI THAY ĐỔI giữa các ảnh (không cố định 1 góc), KHÔNG dùng cách
chia đôi/chia vùng cố định rồi gán nhãn theo vùng (dễ gây nhiễu nhãn nếu lỗi
rơi vào "nửa sai"). Thay vào đó:

  1. Với EMBEDDING "SẠCH" (dùng cho val trong K-Fold, và cho production sau
     này): cắt 5 vùng chồng lấn (4 góc + giữa) TỪ ẢNH GỐC ĐỘ PHÂN GIẢI CAO
     (trước khi resize), mỗi vùng resize riêng về 224x224, trích 5 embedding
     -> lấy MAX theo từng chiều -> gộp thành 1 embedding duy nhất đại diện
     cho "vùng khả nghi nhất" trong ảnh. Nhãn ảnh giữ nguyên là nhãn gốc của
     CẢ ảnh, không tách theo vùng -> không phát sinh nhiễu nhãn.
  2. Với EMBEDDING AUGMENT (dùng cho train): mỗi bản augment lấy 1 vùng cắt
     ngẫu nhiên (vị trí + kích thước ngẫu nhiên quanh crop_scale) TỪ ẢNH GỐC,
     rồi mới áp flip/rotation/color-jitter/resize. Việc này vừa tăng đa dạng
     augmentation, vừa buộc model thấy nhiều "khung nhìn zoom gần" khác nhau
     thay vì luôn thấy toàn ảnh đã bị nén nhỏ.

QUAN TRỌNG khi deploy: nếu MULTI_CROP_CONFIG["enable"] = True lúc train, thì
code inference lúc production CŨNG PHẢI làm đúng bước 5-crop + max-pool này
trước khi đưa ảnh vào model - nếu không sẽ bị lệch phân phối train/inference.
Config multi-crop đã được lưu kèm trong checkpoint (key "multi_crop_config")
để code inference đọc lại và tái tạo đúng pipeline tiền xử lý.

--------------------------------------------------------------------------
CÁCH ÁP DỤNG CHO BÀI TOÁN MỚI:
--------------------------------------------------------------------------
Chỉ cần sửa phần CONFIG bên dưới:
  - DATASET_DIRS: trỏ tới (các) thư mục chứa ảnh, mỗi class 1 thư mục con
  - AUGMENT_CONFIG: bật/tắt phép augmentation phù hợp với bài toán
  - MULTI_CROP_CONFIG: bật/tắt + chỉnh crop_scale cho bài toán lỗi nhỏ
  - BACKBONE_NAME, K_FOLDS, HEAD_EPOCHS...
Không cần sửa bất kỳ dòng logic nào khác trong file.

Cách chạy:
    python few_shot_pipeline_multicrop.py
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw

# ==========================================================================
# CONFIG - SỬA Ở ĐÂY KHI ÁP DỤNG CHO BÀI TOÁN MỚI
# ==========================================================================

DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\HZT.Bottom\train",
    r"D:\TongHop\RTC Technologi\HZT.Bottom\val",
]

MODEL_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\model\model2"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"

# Cấu hình augmentation - BẬT/TẮT theo đặc thù bài toán, không hard-code cứng
AUGMENT_CONFIG = {
    "horizontal_flip": True,     # tắt nếu ảnh có chữ/hướng quan trọng
    "rotation_degrees": 20,      # 0 để tắt xoay
    "color_jitter": False,       # tắt nếu bài toán phụ thuộc MÀU SẮC
    # Chỉ còn tác dụng khi MULTI_CROP_CONFIG["enable"] = False (xem bên dưới)
    "random_resized_crop": True,
    "random_resized_crop_scale": (0.8, 1.0),
}


# --------------------------------------------------------------------------
# MULTI-CROP: dành cho lỗi NHỎ, VỊ TRÍ THAY ĐỔI giữa các ảnh, ảnh gốc LỚN
# (thay thế cho ý tưởng "chia đôi ảnh" - xem giải thích ở docstring đầu file)
# Chạy trực tiếp trên ảnh gốc (không còn bước ROI crop nữa - đã bỏ vì cắt
# mất vùng có lỗi thật).
# --------------------------------------------------------------------------
MULTI_CROP_CONFIG = {
    "enable": True,
    # Quay lại "custom" - CUSTOM_REGIONS bên dưới giờ đã ĐẢM BẢO phủ kín 100%
    # ảnh (không còn rủi ro bỏ sót vùng như bản cũ), đồng thời vẫn giữ đúng
    # 4 vùng domain knowledge (trái/giữa/phải/dưới) thay vì chia hình học vô
    # nghĩa như "grid". Kết hợp được cả 2 mục tiêu: phủ sóng đủ + đúng ngữ
    # nghĩa vùng lỗi.
    "mode": "custom",
    "crop_scale": 0.6,   # chỉ dùng khi mode="grid"
    "positions": ["top_left", "top_right", "bottom_left", "bottom_right", "center"],
    # Độ ngẫu nhiên (+/-) quanh mỗi vùng khi tạo bản augment cho train, để
    # tăng đa dạng vị trí/kích thước thay vì luôn đúng khung cố định.
    "train_jitter": 0.15,
    "min_source_size": 400,
}

# Toạ độ % (0.0-1.0, dạng x0,y0,x1,y1) - CHIA KIỂU TILING: hàng trên chia 3
# cột (trái 1/4 - giữa 1/2 - phải 1/4), hàng dưới chiếm trọn chiều rộng.
# Phủ kín 100% ảnh, KHÔNG chồng lấn, không hở - khác bản cũ (đo từ nét khoanh
# tay, chỉ phủ ~20-30% diện tích, bỏ sót phần lớn ảnh).
CUSTOM_REGIONS = {
    "left_wall":   (0.00, 0.00, 0.25, 2 / 3),   # gần kẹp trái
    "top_bump":    (0.25, 0.00, 0.75, 2 / 3),   # gờ giữa
    "right_wall":  (0.75, 0.00, 1.00, 2 / 3),   # gần kẹp phải
    "bottom_pins": (0.00, 2 / 3, 1.00, 1.00),   # gần chân pin
}

_multicrop_tag = "_multicrop" if MULTI_CROP_CONFIG["enable"] else ""
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}{_multicrop_tag}.pt")

# --------------------------------------------------------------------------
# CUTOUT: che ngẫu nhiên 1 vùng khi tạo bản augment cho TRAIN, để phá shortcut
# đã xác nhận qua Grad-CAM (model bám vào 1 vùng cố định ở giữa bất kể lỗi
# thật nằm ở đâu). Với 1 tỉ lệ % bản augment, vùng "dễ" đó bị che -> model
# buộc phải tìm đặc trưng khác mới đoán đúng NG được. CHỈ áp cho train, KHÔNG
# áp cho val/clean embedding (giữ nguyên để đánh giá đúng thực tế).
# --------------------------------------------------------------------------
CUTOUT_CONFIG = {
    "enable": True,
    "prob": 0.2,          # xác suất 1 bản augment bị áp cutout
    "target": "center",   # "center" = che đúng vùng shortcut đã xác nhận | "random" = che vị trí ngẫu nhiên
    "region_scale": 0.35, # vùng che chiếm ~35% chiều rộng/cao của ảnh đang augment
    "position_jitter": 0.2,  # xê dịch nhẹ quanh target để không che đúng 1 khung cố định tuyệt đối
}


AUGMENT_COPIES = 20
K_FOLDS = 5

# Kích thước ảnh đưa vào backbone. Mặc định 224 (đúng chuẩn pretrained
# ImageNet). Tăng lên (vd 384, 640 - nên chia hết cho 32 để khớp stride của
# MobileNet) giúp giữ chi tiết cho lỗi nhỏ, đặc biệt khi kết hợp multi-crop
# ở trên - cái giá phải trả CHỈ rơi vào bước extract embedding
# (1 lần duy nhất), KHÔNG ảnh hưởng tốc độ vòng lặp train head (chạy trên
# embedding đã cache). Với GPU + dataset nhỏ, tăng thoải mái không cần lo.
INPUT_SIZE = 640
assert INPUT_SIZE % 32 == 0, "INPUT_SIZE nên chia hết cho 32 để khớp stride của MobileNet"

HEAD_EPOCHS = 300
HEAD_LR = 1e-3
HEAD_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 30
PRINT_EPOCH_DETAILS = True

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa khi đổi bài toán
# ==========================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def build_transforms(augment_cfg, skip_random_resized_crop=False):
    """QUAN TRỌNG: Resize((INPUT_SIZE,INPUT_SIZE)) được đặt CUỐI CÙNG (ngay
    trước ToTensor), không phải đầu tiên như bản gốc. Lý do: nếu Resize chạy
    trước, mọi phép crop sau đó (RandomResizedCrop, hay vùng crop multi-crop
    truyền vào đây) đều thao tác trên ảnh ĐÃ MẤT chi tiết -> không tận dụng
    được ảnh gốc độ phân giải cao. Đặt Resize cuối cùng đảm bảo crop luôn
    lấy từ dữ liệu gốc.
    """
    aug_ops = []
    if augment_cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(augment_cfg["rotation_degrees"]))
    if augment_cfg.get("random_resized_crop") and not skip_random_resized_crop:
        aug_ops.append(transforms.RandomResizedCrop(
            INPUT_SIZE, scale=augment_cfg.get("random_resized_crop_scale", (0.8, 1.0))
        ))
    if augment_cfg.get("color_jitter"):
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    aug_ops += [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    clean_ops = [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(aug_ops), transforms.Compose(clean_ops)


# --------------------------------------------------------------------------
# MULTI-CROP: cắt vùng từ ảnh PIL gốc (độ phân giải đầy đủ), trước mọi resize
# --------------------------------------------------------------------------
def _crop_box(w, h, cw, ch, position):
    boxes = {
        "top_left": (0, 0),
        "top_right": (w - cw, 0),
        "bottom_left": (0, h - ch),
        "bottom_right": (w - cw, h - ch),
        "center": ((w - cw) // 2, (h - ch) // 2),
        "top_center": ((w - cw) // 2, 0),
        "bottom_center": ((w - cw) // 2, h - ch),
        "left_center": (0, (h - ch) // 2),
        "right_center": (w - cw, (h - ch) // 2),
    }
    x, y = boxes[position]
    x = max(0, min(x, w - cw))
    y = max(0, min(y, h - ch))
    return (x, y, x + cw, y + ch)


def apply_cutout(image, cutout_cfg):
    """Che 1 vùng chữ nhật (điền màu trung bình ảnh, tránh tạo cạnh tương
    phản giả) với xác suất cutout_cfg['prob']. Dùng để phá shortcut model
    đã học bám vào 1 vùng cố định (xác nhận qua Grad-CAM) - target='center'
    che đúng vùng đó, target='random' che vị trí bất kỳ."""
    if not cutout_cfg.get("enable") or random.random() > cutout_cfg.get("prob", 0):
        return image

    img = image.copy()
    w, h = img.size
    scale = cutout_cfg.get("region_scale", 0.35)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))

    if cutout_cfg.get("target") == "random":
        x = random.randint(0, max(w - cw, 0))
        y = random.randint(0, max(h - ch, 0))
    else:  # "center"
        x = (w - cw) // 2
        y = (h - ch) // 2
        jitter = cutout_cfg.get("position_jitter", 0.0)
        jx, jy = int(cw * jitter), int(ch * jitter)
        x = max(0, min(w - cw, x + random.randint(-jx, jx)))
        y = max(0, min(h - ch, y + random.randint(-jy, jy)))

    arr = np.asarray(img)
    fill_color = tuple(int(v) for v in arr.reshape(-1, arr.shape[-1]).mean(axis=0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + cw, y + ch], fill=fill_color)
    return img


def get_custom_region_boxes(image_size, custom_regions):
    """Trả về list (ten_vung, (x1,y1,x2,y2)) theo toạ độ pixel, dịch từ
    CUSTOM_REGIONS (dạng fraction 0-1) sang kích thước ảnh thực tế."""
    w, h = image_size
    boxes = []
    for name, (fx0, fy0, fx1, fy1) in custom_regions.items():
        x1, y1 = int(fx0 * w), int(fy0 * h)
        x2, y2 = int(fx1 * w), int(fy1 * h)
        boxes.append((name, (x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))))
    return boxes


def jitter_region_box(fx0, fy0, fx1, fy1, jitter):
    """Xê dịch/nới nhẹ 1 box (dạng fraction 0-1) để tăng đa dạng augmentation,
    KHÔNG đổi vùng nào cả - dùng cho từng vùng riêng lẻ, không phải chọn ngẫu
    nhiên 1 trong nhiều vùng."""
    bw, bh = fx1 - fx0, fy1 - fy0
    nx0 = max(0.0, fx0 - random.uniform(0, bw * jitter))
    ny0 = max(0.0, fy0 - random.uniform(0, bh * jitter))
    nx1 = min(1.0, fx1 + random.uniform(0, bw * jitter))
    ny1 = min(1.0, fy1 + random.uniform(0, bh * jitter))
    return nx0, ny0, nx1, ny1


def crop_jittered_custom_region(image, frac_box, jitter):
    """Crop 1 vùng CUSTOM cụ thể (đã biết trước, KHÔNG chọn ngẫu nhiên), có
    jitter nhẹ quanh biên. Dùng để tạo augmented view của vùng đó."""
    w, h = image.size
    fx0, fy0, fx1, fy1 = jitter_region_box(*frac_box, jitter)
    box = (int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h))
    return image.crop((box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1)))


def get_fixed_crop_regions(image, crop_scale, positions):
    """Trả về danh sách ảnh PIL đã cắt tại các vị trí cố định (vd 4 góc + giữa).
    Dùng cho embedding "sạch" (val / production) - deterministic, không ngẫu nhiên."""
    w, h = image.size
    cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
    return [image.crop(_crop_box(w, h, cw, ch, pos)) for pos in positions]


def crop_jittered_grid_region(image, crop_scale, jitter, position):
    """Crop 1 vị trí cụ thể trong kiểu 'grid' (top_left, center...), có jitter
    nhẹ quanh biên. Tương đương crop_jittered_custom_region nhưng cho mode grid."""
    w, h = image.size
    cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
    x1, y1, x2, y2 = _crop_box(w, h, cw, ch, position)
    frac_box = (x1 / w, y1 / h, x2 / w, y2 / h)
    fx0, fy0, fx1, fy1 = jitter_region_box(*frac_box, jitter)
    box = (int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h))
    return image.crop((box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1)))


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
def precompute_all_embeddings(backbone, image_paths, augment_transform, clean_transform,
                               augment_copies, multi_crop_cfg, custom_regions):
    """
    augmented_emb[i]: tensor (augment_copies, dim) - MỖI HÀNG đã là max-pool
                       qua toàn bộ các vùng (không phải embedding của 1 vùng
                       đơn lẻ nữa - xem giải thích trong hàm bên dưới).
    clean_emb[i]:      tensor (1, dim) - max-pool qua toàn bộ các vùng, không augmentation.

    Nếu multi_crop_cfg["enable"] = True VÀ ảnh đủ lớn, có 2 chế độ:
      - mode="custom": dùng đúng CUSTOM_REGIONS (domain knowledge thực tế,
        đã tự phủ kín 100% ảnh - xem CONFIG)
      - mode="grid": 4 góc + giữa theo crop_scale (kiểu cũ, tự động)
    Ngược lại: giữ nguyên hành vi gốc (dùng cả ảnh, resize toàn bộ).
    """
    augmented_emb, clean_emb = [], []
    mc_enable = multi_crop_cfg["enable"]
    mode = multi_crop_cfg.get("mode", "grid")
    crop_scale = multi_crop_cfg["crop_scale"]
    positions = multi_crop_cfg["positions"]
    jitter = multi_crop_cfg["train_jitter"]
    min_size = multi_crop_cfg["min_source_size"]

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        use_multi_crop = mc_enable and min(w, h) >= min_size

        # ---- augmented embeddings (train) ----
        # QUAN TRỌNG: mỗi bản augment giờ CŨNG max-pool qua toàn bộ các vùng,
        # y hệt cách tính embedding "sạch" bên dưới - trước đây mỗi bản augment
        # chỉ lấy 1 vùng đơn lẻ (không max-pool), khiến train thấy 1 phân phối
        # (embedding của 1 vùng) còn val/inference lại thấy phân phối khác hẳn
        # (max của nhiều vùng) -> model học lệch, đây là nguyên nhân sâu xa của
        # nhiều dự đoán sai trước đó. Đa dạng augmentation vẫn giữ được nhờ
        # jitter riêng cho từng vùng + flip/rotate/cutout ở mỗi bản copy.
        aug_final_embs = []
        for _ in range(augment_copies):
            if use_multi_crop and mode == "custom":
                regions_for_copy = [crop_jittered_custom_region(image, box, jitter)
                                     for box in custom_regions.values()]
            elif use_multi_crop:
                regions_for_copy = [crop_jittered_grid_region(image, crop_scale, jitter, pos)
                                     for pos in positions]
            else:
                regions_for_copy = [image]
            regions_for_copy = [apply_cutout(r, CUTOUT_CONFIG) for r in regions_for_copy]  # chỉ áp cho train
            copy_tensors = torch.stack([augment_transform(r) for r in regions_for_copy]).to(DEVICE)
            copy_embs = extract_embedding(backbone, copy_tensors)            # (num_vung, dim)
            aug_final_embs.append(copy_embs.max(dim=0, keepdim=True).values)  # (1, dim)
        aug_emb = torch.cat(aug_final_embs, dim=0)   # (augment_copies, dim)
        augmented_emb.append(aug_emb.cpu())

        # ---- clean embedding (val) ----
        if use_multi_crop and mode == "custom":
            boxes = get_custom_region_boxes((w, h), custom_regions)
            regions = [image.crop(box) for _, box in boxes]
            clean_tensors = torch.stack([clean_transform(r) for r in regions]).to(DEVICE)
            region_embs = extract_embedding(backbone, clean_tensors)
            c_emb = region_embs.max(dim=0, keepdim=True).values
        elif use_multi_crop:
            regions = get_fixed_crop_regions(image, crop_scale, positions)
            clean_tensors = torch.stack([clean_transform(r) for r in regions]).to(DEVICE)
            region_embs = extract_embedding(backbone, clean_tensors)   # (num_regions, dim)
            c_emb = region_embs.max(dim=0, keepdim=True).values         # (1, dim)
        else:
            clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
            c_emb = extract_embedding(backbone, clean_tensor)
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
    best_epoch = 0
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
                best_epoch = epoch + 1
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
            best_epoch = epoch + 1
            if print_epochs:
                epoch_time_ms = (time.time() - epoch_start) * 1000
                print(f"  [{fold_label}] Epoch {epoch+1:3d}/{epochs} | Train loss: {loss.item():.4f} "
                      f"| Train acc: {train_acc:.2%} | {epoch_time_ms:.1f}ms")

    head.load_state_dict(best_state)
    return head, best_val_acc, best_val_loss, best_val_probs, best_epoch


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
    run_start = time.time()
    print(f"Đang chạy trên: {DEVICE}")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Multi-crop: {'BẬT' if MULTI_CROP_CONFIG['enable'] else 'TẮT'}"
          + (f" (mode={MULTI_CROP_CONFIG['mode']}, "
             + (f"regions={list(CUSTOM_REGIONS.keys())})" if MULTI_CROP_CONFIG['mode'] == 'custom'
                else f"crop_scale={MULTI_CROP_CONFIG['crop_scale']}, positions={MULTI_CROP_CONFIG['positions']})")
             if MULTI_CROP_CONFIG['enable'] else ""))
    print(f"Cutout (phá shortcut, chỉ áp cho train): {'BẬT' if CUTOUT_CONFIG['enable'] else 'TẮT'}"
          + (f" (prob={CUTOUT_CONFIG['prob']}, target={CUTOUT_CONFIG['target']}, "
             f"region_scale={CUTOUT_CONFIG['region_scale']})" if CUTOUT_CONFIG['enable'] else ""))

    image_paths, labels, class_names = list_images(DATASET_DIRS)
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class: {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")

    skip_rrc = MULTI_CROP_CONFIG["enable"]  # multi-crop tự lo việc "zoom" rồi, khỏi crop chồng thêm lần nữa
    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG, skip_random_resized_crop=skip_rrc)

    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    print(f"\nĐang extract embedding cho toàn bộ {len(image_paths)} ảnh "
          f"({AUGMENT_COPIES} bản augment/ảnh, chỉ chạy 1 lần)...")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, image_paths, augment_transform, clean_transform, AUGMENT_COPIES,
        MULTI_CROP_CONFIG, CUSTOM_REGIONS
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # --------------------------------------------------------------------
    fold_assignment = stratified_k_fold_indices(labels, K_FOLDS, seed=RANDOM_SEED)
    labels_np = np.array(labels)

    print(f"\n=== K-Fold Cross Validation (K={K_FOLDS}) ===")
    fold_accs, fold_losses, fold_best_epochs = [], [], []

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
        _, val_acc, val_loss, val_probs, best_epoch = train_head(
            train_emb, train_lbl, val_emb, val_lbl, embedding_dim, num_classes,
            HEAD_EPOCHS, HEAD_LR, HEAD_DROPOUT, EARLY_STOP_PATIENCE,
            print_epochs=PRINT_EPOCH_DETAILS, fold_label=f"Fold {fold+1}/{K_FOLDS}"
        )

        fold_accs.append(val_acc)
        fold_losses.append(val_loss)
        fold_best_epochs.append(best_epoch)

        for local_i, global_i in enumerate(val_idx):
            probs_row = val_probs[local_i]
            oof_pred[global_i] = int(probs_row.argmax().item())
            oof_probs[global_i] = probs_row.tolist()
            oof_fold[global_i] = fold

        print(f"Fold {fold+1}/{K_FOLDS}: {len(train_idx)} ảnh train, {len(val_idx)} ảnh val "
              f"-> Val acc: {val_acc:.2%} | Val loss: {val_loss:.4f} | Best epoch: {best_epoch}")

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

    # Số epoch cho model cuối = trung bình epoch "best" thực tế đo được qua các fold
    # (chính xác hơn hẳn so với ước lượng cứng HEAD_EPOCHS*0.5 của bản gốc)
    final_epochs = max(50, int(np.mean(fold_best_epochs)))
    print(f"Số epoch dùng cho model cuối (trung bình best-epoch của {K_FOLDS} fold): {final_epochs}")

    final_head, _, _, _, _ = train_head(
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
        # Lưu lại config multi-crop để code inference/production tái tạo ĐÚNG
        # bước tiền xử lý (5-crop + max-pool) - bắt buộc nếu enable=True,
        # nếu không sẽ bị lệch phân phối giữa lúc train và lúc chạy thật.
        "multi_crop_config": MULTI_CROP_CONFIG,
        "custom_regions": CUSTOM_REGIONS,
        "input_size": INPUT_SIZE,
    }, MODEL_OUT)
    print(f"Đã lưu model cuối cùng tại: {MODEL_OUT}")

    # --------------------------------------------------------------------
    # Đo thông số edge
    # --------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = full_model(dummy_input)
    avg_latency_ms = (time.time() - start) / n_runs * 1000
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)

    print(f"\n--- Thông số edge ---")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Latency trung bình (1 ảnh, KHÔNG tính bước 5-crop tiền xử lý nếu multi-crop bật): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")
    print(f"K-Fold accuracy (độ tin cậy thật): {fold_accs.mean():.2%} ± {fold_accs.std():.2%}")

    total_elapsed = time.time() - run_start
    print(f"\n=== TỔNG THỜI GIAN CHẠY: {total_elapsed/60:.1f} phút ({total_elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()