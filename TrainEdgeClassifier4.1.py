"""
KIẾN TRÚC CHUNG: Pretrained-CNN + Few-shot learning + K-Fold Cross Validation.

Đây là pipeline TỔNG QUÁT, dùng lại được cho BẤT KỲ bài toán phân loại ảnh
ít dữ liệu nào (màu sắc, trái cây, OK/NG, vật thể...) - chỉ cần sửa phần
CONFIG bên dưới, không cần đụng vào logic code.

--------------------------------------------------------------------------
Ý TƯỞNG CỐT LÕI (không đổi giữa các bài toán) - VÌ SAO THIẾT KẾ NHƯ VẬY:
--------------------------------------------------------------------------
1. Backbone CNN (MobileNetV2/V3) ĐÓNG BĂNG HOÀN TOÀN, không fine-tune.
   -> Backbone đã được train sẵn trên ImageNet (1.4 triệu ảnh, 1000 class),
      nên nó đã "biết" cách trích đặc trưng thị giác tổng quát (cạnh, texture,
      hình khối...). Với dataset few-shot (vài chục ảnh), nếu train lại cả
      CNN (fine-tune) thì số tham số quá lớn so với số ảnh -> overfit gần
      như chắc chắn. Đóng băng backbone = chỉ dùng nó như 1 "máy trích đặc
      trưng" cố định, không học thêm gì từ backbone nữa.

2. Extract embedding (vector đặc trưng) cho mỗi ảnh 1 LẦN DUY NHẤT, cache
   lại -> việc "train" sau đó chỉ là train 1 classifier NHẸ trên embedding,
   nhanh gấp hàng chục lần so với train lại cả CNN.
   -> Vì backbone đông lạnh, kết quả feature-extraction của backbone cho 1
      ảnh KHÔNG đổi bất kể chạy bao nhiêu epoch. Nếu chạy lại backbone mỗi
      epoch sẽ rất lãng phí (backbone nặng hơn classifier hàng nghìn lần).
      Nên: chạy backbone đúng 1 lần cho mỗi ảnh (và mỗi bản augment của nó),
      lưu kết quả (embedding) vào RAM. Từ đó, "training" thực chất chỉ là
      train 1 lớp Linear nhỏ trên các vector đã có sẵn - cực nhanh.

3. Vì dataset ít, KHÔNG chia train/val cố định một lần (dễ bị đánh giá sai
    lệch, không ổn định) -> dùng K-FOLD CROSS VALIDATION: gộp hết dữ liệu,
    xoay vòng chia K phần, mỗi ảnh đều được làm "val" ít nhất 1 lần.
   -> Nếu chỉ chia 1 lần (vd 80% train / 20% val) trên dataset nhỏ, kết quả
      accuracy đo được phụ thuộc RẤT NHIỀU vào việc "may rủi" 20% val đó rơi
      trúng ảnh dễ hay ảnh khó. K-Fold giải quyết bằng cách để MỌI ảnh đều
      lần lượt được đánh giá (làm val) đúng 1 lần trong toàn bộ quá trình,
      rồi gộp kết quả lại -> con số accuracy phản ánh đúng thực tế hơn nhiều.
      Đây là cách giải quyết "làm giàu val" ĐÚNG NGUYÊN TẮC và TỔNG QUÁT,
      không phụ thuộc vào bài toán cụ thể là gì.

4. Sau khi đánh giá xong bằng K-Fold, train 1 model "sản xuất" cuối cùng
   trên TOÀN BỘ dữ liệu (không giữ lại val) để deploy.
   -> K-Fold ở bước 3 chỉ để ĐO ĐỘ TIN CẬY của cách làm (con số accuracy),
      không phải để chọn ra 1 model cụ thể để dùng. Sau khi đã tin tưởng
      con số đó, ta train LẠI 1 model hoàn toàn mới, lần này cho nó học từ
      100% dữ liệu có (không chừa lại phần nào để làm val nữa) - vì trong
      thực tế deploy, càng nhiều dữ liệu train càng tốt, không cần giữ val
      riêng nữa (đã đánh giá xong ở bước 3 rồi).

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
    python few_shot_pipeline_annotated.py
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
# lại rồi chia K-Fold, không cần bạn tự gộp tay. Lý do gộp: K-Fold cần TOÀN
# BỘ dữ liệu nằm chung 1 chỗ để tự chia lại theo fold, không dùng khái niệm
# "train/val cố định" của bạn nữa.
DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\HZT.Bottom\OK_drop\train",
    r"D:\TongHop\RTC Technologi\HZT.Bottom\OK_drop\val",
]

MODEL_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\OK_drop\model\model1"
BACKBONE_NAME = "mobilenet_v3_large"   # "mobilenet_v2" | "mobilenet_v3_large" | "mobilenet_v3_small"
# Càng "large" thì embedding càng chi tiết (960 chiều) nhưng cũng nặng hơn
# 1 chút so với "small" (576 chiều). Với ảnh công nghiệp cần phân biệt chi
# tiết nhỏ, "large" thường cho kết quả tốt hơn, đổi lại latency cao hơn
# đôi chút (vẫn rất nhanh vì chỉ chạy 1 lần lúc extract, xem ý (2) ở trên).
MODEL_OUT = os.path.join(MODEL_DIR, f"edge_classifier_fewshot_{BACKBONE_NAME}.pt")

# Cấu hình augmentation - BẬT/TẮT theo đặc thù bài toán, không hard-code cứng
AUGMENT_CONFIG = {
    "horizontal_flip": True,     # tắt nếu ảnh có chữ/hướng quan trọng
    "rotation_degrees": 20,      # 0 để tắt xoay
    "color_jitter": False,       # tắt nếu bài toán phụ thuộc MÀU SẮC (như hiện tại)
    "random_resized_crop": True, # tắt nếu bố cục/vị trí vật thể trong ảnh quan trọng
}
# Vì sao augmentation quan trọng với few-shot: dataset chỉ có vài chục ảnh,
# nếu model chỉ nhìn thấy đúng NHỮNG pixel đó, nó rất dễ "học thuộc lòng"
# (overfit) thay vì học đặc trưng tổng quát. Augmentation tạo ra nhiều biến
# thể (xoay, lật, crop khác góc...) từ CÙNG 1 ảnh gốc, giúp model thấy nhiều
# "phiên bản" hơn của cùng 1 lỗi/đối tượng, khó học thuộc từng pixel hơn.

AUGMENT_COPIES = 20   # số bản augmentation/ảnh khi extract embedding cho train
# Con số này = số "bản sao đã biến đổi" của MỖI ảnh train được tạo ra. Ảnh
# càng ít, số này nên càng lớn để bù lại (nhưng tăng quá cao cũng không giúp
# thêm nhiều vì các bản augment bắt đầu bị trùng lặp/tương tự nhau).

K_FOLDS = 5            # số fold cross-validation - tăng nếu dataset lớn hơn
# K=5 nghĩa là dữ liệu chia làm 5 phần, mỗi lần lấy 4 phần train + 1 phần
# val, xoay vòng 5 lần. K càng lớn thì mỗi fold train có nhiều dữ liệu hơn
# (tốt), nhưng val mỗi fold lại ít hơn (kết quả từng fold dễ dao động hơn).
# K=5 là lựa chọn cân bằng phổ biến. Nếu dataset RẤT nhỏ (dưới ~20 ảnh/class)
# có thể cân nhắc giảm K (vd 3) để mỗi fold train có đủ dữ liệu học.

HEAD_EPOCHS = 300      # số epoch TỐI ĐA để train classifier head (có early stop)
HEAD_LR = 1e-3         # learning rate cho Adam optimizer
HEAD_DROPOUT = 0.3     # tỉ lệ dropout trong classifier head, chống overfit
EARLY_STOP_PATIENCE = 50   # dừng sớm nếu val loss không cải thiện sau 30 epoch liên tiếp
PRINT_EPOCH_DETAILS = True   # True = in chi tiết từng epoch trong mỗi fold (như trước)

RANDOM_SEED = 42
# Cố định seed để mỗi lần chạy lại cho kết quả GIỐNG HỆT nhau (từ việc chia
# fold, tới augmentation ngẫu nhiên, tới khởi tạo trọng số classifier) -
# quan trọng để so sánh công bằng khi thử nghiệm thay đổi config.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# benchmark=True: để cuDNN tự thử vài thuật toán convolution lúc đầu và chọn
# cái nhanh nhất cho đúng kích thước input đang dùng - có lợi khi kích thước
# ảnh đầu vào KHÔNG đổi qua các lần forward (đúng trường hợp ở đây).

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa khi đổi bài toán
# ==========================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def build_transforms(augment_cfg):
    """Tạo 2 pipeline biến đổi ảnh khác nhau:
      - aug_ops (augment_transform): dùng cho TRAIN - có random augmentation
        (flip, xoay, crop, color jitter tuỳ bật/tắt trong CONFIG) để tăng đa
        dạng dữ liệu, chống overfit.
      - clean_ops (clean_transform): dùng cho VAL/INFERENCE - KHÔNG có gì
        ngẫu nhiên, chỉ resize + chuẩn hoá, để đánh giá công bằng, ổn định
        (không muốn kết quả val dao động do augmentation ngẫu nhiên).
    Cả 2 đều dùng Normalize với mean/std của ImageNet - BẮT BUỘC phải khớp
    với những gì backbone pretrained đã quen, nếu không embedding sẽ sai
    lệch nghiêm trọng (đây là "hợp đồng ngầm" giữa dữ liệu và pretrained
    weight, không phải chọn số tuỳ ý)."""
    aug_ops = [transforms.Resize((224, 224))]
    # 224x224: kích thước input chuẩn mà MobileNet được pretrain trên
    # ImageNet - đổi khác đi thì features vẫn tính được (vì đây là mạng tích
    # chập, không có lớp Fully-Connected cố định kích thước ở giữa) nhưng
    # càng lệch xa 224 thì đặc trưng học được lúc pretrain càng ít khớp với
    # tỉ lệ vật thể trong ảnh mới.
    if augment_cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(augment_cfg["rotation_degrees"]))
    if augment_cfg.get("random_resized_crop"):
        aug_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
        # scale=(0.8, 1.0): mỗi lần crop 1 vùng ngẫu nhiên chiếm 80%-100%
        # diện tích ảnh gốc rồi resize lại về 224x224 - mô phỏng việc vật
        # thể xuất hiện ở các vị trí/kích thước hơi khác nhau trong khung
        # hình, không "zoom" quá sâu (0.8 là mức nhẹ, giữ gần như cả ảnh).
    if augment_cfg.get("color_jitter"):
        aug_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        # Chỉ bật khi màu sắc KHÔNG phải đặc trưng quan trọng để phân loại -
        # nếu bài toán phụ thuộc màu (vd phân loại theo màu sắc sản phẩm),
        # bật cái này sẽ "phá" chính tín hiệu cần học, nên mặc định tắt.
    aug_ops += [
        transforms.ToTensor(),  # chuyển ảnh PIL (0-255) -> tensor PyTorch (0.0-1.0)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 3 số mean và 3 số std này là thống kê (trung bình, độ lệch chuẩn)
        # của tập ImageNet gốc mà MobileNet được train trên đó - chuẩn hoá
        # ảnh mới về đúng "thang đo" mà backbone quen, giúp các layer đầu
        # tiên của CNN hoạt động đúng như lúc pretrain.
    ]

    clean_ops = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(aug_ops), transforms.Compose(clean_ops)


def list_images(dataset_dirs):
    """Gộp ảnh từ nhiều thư mục (vd train/ + val/) thành 1 danh sách chung.
    Trả về: image_paths (list[str]), labels (list[int]), class_names (list[str]).

    Cách hoạt động:
      1. Quét TẤT CẢ thư mục trong dataset_dirs, thu thập tên các thư mục
         con (mỗi thư mục con = 1 class, vd "NG", "OK") - GỘP tên class từ
         mọi nơi lại (dùng set để tự loại trùng), rồi sort để thứ tự class
         luôn cố định giữa các lần chạy (quan trọng vì class_to_idx dựa vào
         thứ tự này - nếu thứ tự đổi, label index cũng đổi theo, gây lẫn lộn
         nếu so sánh giữa các lần chạy khác nhau).
      2. Với mỗi thư mục trong dataset_dirs, với mỗi class đã biết, liệt kê
         hết ảnh trong thư mục con tương ứng (nếu có), gán nhãn số theo
         class_to_idx.
    """
    class_names = set()
    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {d}")
        for entry in os.scandir(d):
            if entry.is_dir():
                class_names.add(entry.name)
    class_names = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    # sorted() đảm bảo thứ tự alphabet cố định, vd ["NG", "OK"] -> NG=0, OK=1
    # LUÔN NHƯ VẬY dù chạy máy nào, ngày nào - vì os.scandir() không đảm bảo
    # thứ tự trả về ổn định giữa các hệ điều hành/lần chạy.

    image_paths, labels = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for d in dataset_dirs:
        for cls in class_names:
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue   # thư mục d này không có class cls -> bỏ qua, không lỗi
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

    return image_paths, labels, class_names


def build_backbone(backbone_name):
    """Tạo backbone CNN đã pretrain trên ImageNet, ĐÓNG BĂNG toàn bộ tham số
    (requires_grad=False) - nghĩa là khi gọi .backward() sau này, KHÔNG có
    gradient nào chảy vào các trọng số của backbone, chúng giữ nguyên giá
    trị pretrained mãi mãi. backbone.eval() cũng tắt Dropout/BatchNorm-update
    bên trong backbone, đảm bảo output ổn định (không đổi ngẫu nhiên) mỗi
    lần forward cùng 1 input - cần thiết vì ta chỉ extract embedding 1 lần
    rồi cache lại, nếu output không ổn định thì cache sẽ sai."""
    if backbone_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        embedding_dim = backbone.last_channel
    elif backbone_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        embedding_dim = 960   # số kênh (channel) ở lớp feature cuối cùng của MobileNetV3-Large
    elif backbone_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        embedding_dim = 576
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")

    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False   # đóng băng - đây là dòng quan trọng nhất của cả file
    return backbone, embedding_dim


def build_full_model(backbone_name):
    """Tạo 1 model TOÀN VẸN (backbone + classifier gắn liền) để LƯU RA FILE
    cuối cùng cho việc deploy - khác với build_backbone() ở trên (chỉ dùng
    nội bộ để tính embedding). Model này khi load lại ở nơi khác chỉ cần gọi
    model(ảnh) là ra thẳng logits, không cần biết gì về bước tách backbone/
    classifier lúc train nữa - tiện cho việc triển khai thực tế."""
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
    # Ở cuối main(), model.classifier của model này sẽ bị THAY THẾ bằng cái
    # head vừa train xong (xem đoạn "full_model.classifier = final_head").


@torch.no_grad()
def extract_embedding(backbone, image_tensor):
    """Chạy ảnh qua phần "features" của backbone (các lớp convolution, KHÔNG
    gồm lớp phân loại 1000-class gốc của ImageNet - ta không cần nó), rồi
    Global Average Pooling (GAP) để nén feature map không gian (C, H, W)
    thành 1 vector (C,) duy nhất - "tóm tắt" toàn bộ ảnh thành 1 điểm trong
    không gian đặc trưng C chiều (vd 960 chiều với MobileNetV3-Large).

    LƯU Ý QUAN TRỌNG (đặc biệt nếu sau này áp dụng cho lỗi nhỏ/cục bộ): GAP
    lấy TRUNG BÌNH toàn bộ không gian ảnh, nên 1 lỗi rất nhỏ so với cả ảnh sẽ
    bị "loãng" đi nhiều trong phép trung bình này - đây là giới hạn cố hữu
    của cách làm "GAP trên cả ảnh", cần nhớ nếu sau này thấy model khó phát
    hiện lỗi nhỏ cục bộ."""
    x = backbone.features(image_tensor)          # (N, C, H, W) - feature map không gian
    x = F.adaptive_avg_pool2d(x, (1, 1))          # (N, C, 1, 1) - trung bình theo H, W
    return torch.flatten(x, 1)                    # (N, C) - vector embedding phẳng


@torch.no_grad()
def precompute_all_embeddings(backbone, image_paths, augment_transform, clean_transform, augment_copies):
    """
    Với MỖI ảnh gốc, tính sẵn:
      - augmented_emb[i]: tensor (augment_copies, dim) - dùng khi ảnh này rơi vào phần TRAIN của 1 fold
      - clean_emb[i]:      tensor (1, dim)               - dùng khi ảnh này rơi vào phần VAL của 1 fold
    Việc này chỉ làm 1 LẦN cho toàn bộ dataset, sau đó K-Fold chỉ việc
    "chọn lại" các embedding có sẵn theo từng fold - không phải extract lại.

    Vì sao TRAIN và VAL cần 2 loại embedding khác nhau: TRAIN cần nhiều bản
    augment (đa dạng) để classifier học tổng quát hơn; VAL cần đúng 1 bản
    SẠCH (không augment) để đo hiệu năng thực tế 1 cách ổn định, không lẫn
    ngẫu nhiên của augmentation vào kết quả đánh giá.
    """
    augmented_emb, clean_emb = [], []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        # .convert("RGB"): ép về 3 kênh màu dù ảnh gốc là grayscale (1 kênh)
        # hay có kênh alpha (4 kênh, vd PNG trong suốt) - backbone ImageNet
        # luôn cần input đúng 3 kênh.

        # Tạo augment_copies (vd 20) bản biến đổi ngẫu nhiên KHÁC NHAU từ
        # CÙNG 1 ảnh gốc, gộp thành 1 batch rồi forward qua backbone 1 lượt
        # (nhanh hơn forward từng ảnh 1 do tận dụng song song của GPU).
        aug_tensors = torch.stack([augment_transform(image) for _ in range(augment_copies)]).to(DEVICE)
        aug_emb = extract_embedding(backbone, aug_tensors)
        augmented_emb.append(aug_emb.cpu())
        # .cpu(): chuyển embedding về RAM thường (không giữ trên VRAM), vì
        # sau đây ta sẽ cache TOÀN BỘ embedding của TOÀN BỘ dataset trong
        # suốt quá trình K-Fold - nếu giữ trên GPU sẽ dễ tràn VRAM với
        # dataset lớn hơn. Lúc train_head() cần dùng, sẽ .to(DEVICE) lại.

        clean_tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
        # .unsqueeze(0): thêm 1 chiều batch giả (1, C, H, W) vì backbone
        # luôn kỳ vọng input có chiều batch, kể cả khi chỉ có 1 ảnh.
        c_emb = extract_embedding(backbone, clean_tensor)
        clean_emb.append(c_emb.cpu())

    return augmented_emb, clean_emb


def stratified_k_fold_indices(labels, k, seed=42):
    """Chia index thành k fold, đảm bảo tỉ lệ mỗi class đồng đều giữa các fold
    (stratified) - viết tay, không phụ thuộc thư viện ngoài.

    Vì sao cần STRATIFIED (chia đều theo class) thay vì chia ngẫu nhiên đơn
    thuần: nếu chia hoàn toàn ngẫu nhiên trên dataset nhỏ, có thể xảy ra
    trường hợp 1 fold nào đó "vô tình" chứa toàn ảnh của 1 class (vd fold 3
    chỉ có ảnh OK, không có ảnh NG nào) - lúc đó model train trên 4 fold còn
    lại sẽ không cân bằng, và đánh giá trên fold 3 (100% OK) sẽ cho accuracy
    giả tạo, không phản ánh đúng khả năng phân biệt NG/OK. Stratified đảm
    bảo MỌI fold đều có đại diện của MỌI class, theo đúng tỉ lệ tổng thể.
    """
    labels = np.array(labels)
    rng = np.random.RandomState(seed)
    fold_assignment = np.zeros(len(labels), dtype=int)

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]   # toàn bộ vị trí (index) của class c trong dataset
        rng.shuffle(idx_c)                  # xáo trộn ngẫu nhiên thứ tự các ảnh của class c
        # Chia đều idx_c thành k phần, phân bổ lần lượt fold 0,1,2,...,k-1
        for i, idx in enumerate(idx_c):
            fold_assignment[idx] = i % k
            # vd k=5: ảnh thứ 0,5,10,... của class c -> fold 0
            #         ảnh thứ 1,6,11,... của class c -> fold 1 ... v.v.
            # -> mỗi class được "rải đều" qua 5 fold theo kiểu vòng tròn.

    return fold_assignment


def make_head(embedding_dim, num_classes, dropout):
    """Classifier "head" - phần DUY NHẤT thực sự được TRAIN trong toàn bộ
    pipeline. Cấu trúc cực đơn giản: Dropout rồi 1 lớp Linear duy nhất
    (không có hidden layer, không có activation phi tuyến ở giữa) - đây gọi
    là "linear probing". Vì sao đơn giản như vậy lại đủ: embedding đầu vào
    đã được backbone ImageNet trích xuất RẤT tốt rồi (không gian đặc trưng
    đã gần như tuyến tính-phân-tách-được cho nhiều bài toán); và với dataset
    ít, 1 lớp Linear (số tham số = embedding_dim x num_classes, rất nhỏ) khó
    overfit hơn nhiều so với 1 mạng nhiều lớp."""
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes)).to(DEVICE)


def train_head(train_emb, train_labels, val_emb, val_labels, embedding_dim, num_classes,
                epochs, lr, dropout, patience, print_epochs=False, fold_label=""):
    """Train 1 classifier head trên embedding đã cache. Dùng chung cho cả
    K-Fold (train/val nội bộ mỗi fold) lẫn train model sản xuất cuối cùng.
    Trả về thêm val_probs (xác suất từng class cho từng ảnh val) để phục vụ
    báo cáo chi tiết. Nếu print_epochs=True, in ra tiến trình từng epoch."""
    head = make_head(embedding_dim, num_classes, dropout)

    # --- Class weighting: bù lại mất cân bằng số lượng ảnh giữa các class ---
    # Công thức: weight_c = tổng_số_mẫu / (số_class * số_mẫu_của_class_c)
    # -> class có ÍT mẫu hơn sẽ có weight LỚN hơn trong hàm loss, nghĩa là
    # mỗi lần model đoán sai 1 ảnh thuộc class hiếm sẽ bị "phạt" nặng hơn so
    # với đoán sai ảnh thuộc class phổ biến - chống việc model chỉ cần "luôn
    # đoán class đông hơn" để đạt accuracy cao giả tạo.
    class_weights = train_emb.shape[0] / (
        num_classes * np.bincount(train_labels.cpu().numpy(), minlength=num_classes)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    # ReduceLROnPlateau: theo dõi val_acc, nếu 3 epoch liên tiếp KHÔNG cải
    # thiện thì tự động giảm learning rate đi 1 nửa (factor=0.5) - giúp model
    # "tinh chỉnh" nhẹ hơn khi đã gần hội tụ, tránh dao động quanh điểm tối
    # ưu do bước nhảy (LR) quá lớn.

    has_val = val_emb is not None and val_emb.shape[0] > 0
    # has_val=False xảy ra khi train model SẢN XUẤT cuối cùng (không còn
    # tách val riêng nữa, dùng 100% dữ liệu để train) - lúc đó không có gì
    # để early-stop theo, phải chạy đủ số epoch được truyền vào.

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_probs = None
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    # Lưu "bản sao" trọng số tốt nhất (theo val_loss thấp nhất) - vì epoch
    # CUỐI CÙNG chưa chắc là epoch TỐT NHẤT (model có thể overfit dần về
    # cuối) - ta muốn khôi phục lại đúng lúc model tốt nhất, không phải lúc
    # dừng lại.
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        head.train()   # bật chế độ train (kích hoạt Dropout ngẫu nhiên)
        optimizer.zero_grad()
        outputs = head(train_emb)
        # LƯU Ý: đây là FULL-BATCH training - đưa TOÀN BỘ train_emb vào 1
        # lần duy nhất mỗi epoch (không chia mini-batch, không có DataLoader)
        # - hợp lý vì train_emb chỉ là vài nghìn vector (không phải ảnh thô),
        # gọn nhẹ, GPU xử lý trong 1 lần thoải mái với dataset nhỏ thế này.
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_acc = (outputs.argmax(dim=1) == train_labels).float().mean().item()

        if has_val:
            head.eval()   # tắt Dropout khi đánh giá, để kết quả val ổn định/tái lập được
            with torch.no_grad():
                val_outputs = head(val_emb)
                val_loss = criterion(val_outputs, val_labels).item()
                val_probs = F.softmax(val_outputs, dim=1)
                val_acc = (val_probs.argmax(dim=1) == val_labels).float().mean().item()
            scheduler.step(val_acc)

            status = ""
            if val_loss < best_val_loss:
                # Chọn "tốt nhất" theo VAL LOSS (không phải val accuracy) -
                # vì loss mịn hơn (liên tục), phản ánh độ TỰ TIN đúng/sai của
                # model chi tiết hơn accuracy (chỉ đúng/sai nhị phân), nên
                # nhạy hơn để phát hiện model đang cải thiện hay đang overfit.
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
                # EARLY STOPPING: nếu đã "patience" epoch liên tiếp mà val
                # loss không cải thiện thêm chút nào -> model gần như chắc
                # chắn đã hội tụ (hoặc đang bắt đầu overfit) -> dừng sớm để
                # tiết kiệm thời gian, không chạy hết epochs vô ích.
                if print_epochs:
                    print(f"  [{fold_label}] Val loss không cải thiện sau {patience} epoch -> dừng sớm.")
                break
        else:
            # Nhánh KHÔNG có val (train model sản xuất cuối) - không có gì
            # để so sánh "tốt nhất", nên đơn giản lưu trọng số MỚI NHẤT mỗi
            # epoch (sẽ chạy đủ hết epochs vì patience=epochs trong lần gọi
            # này, xem main() bên dưới).
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            if print_epochs:
                epoch_time_ms = (time.time() - epoch_start) * 1000
                print(f"  [{fold_label}] Epoch {epoch+1:3d}/{epochs} | Train loss: {loss.item():.4f} "
                      f"| Train acc: {train_acc:.2%} | {epoch_time_ms:.1f}ms")

    head.load_state_dict(best_state)   # khôi phục lại trọng số TỐT NHẤT đã lưu, không phải trọng số ở epoch cuối
    return head, best_val_acc, best_val_loss, best_val_probs


def print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names):
    """In báo cáo chi tiết dựa trên kết quả OUT-OF-FOLD (oof) - nghĩa là với
    MỖI ảnh trong dataset, dùng đúng lần nó được làm VAL (không phải lúc nó
    làm train) để lấy prediction. Vì mỗi ảnh chỉ làm val ĐÚNG 1 LẦN trong
    toàn bộ K-Fold, kết quả oof cho TOÀN BỘ dataset này hoàn toàn KHÔNG
    THIÊN VỊ (unbiased) - không có ảnh nào được đánh giá bằng chính model đã
    "nhìn thấy" nó lúc train."""
    import csv

    num_classes = len(class_names)
    n = len(image_paths)

    # --- Confusion matrix (hàng = thật, cột = dự đoán) ---
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(n):
        confusion[labels_np[i], oof_pred[i]] += 1
    # confusion[a][b] = số ảnh có NHÃN THẬT là class a nhưng bị ĐOÁN thành
    # class b. Đường chéo (confusion[a][a]) = số ảnh đoán đúng của class a.
    # Nhìn confusion matrix cho biết CHÍNH XÁC model hay nhầm lẫn theo chiều
    # nào (vd hay đoán NG thành OK, hay ngược lại) - thông tin mà chỉ 1 con
    # số accuracy tổng không thể hiện được.

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
        # "Conf TB khi sai" đặc biệt đáng chú ý: nếu con số này CAO (model
        # tự tin nhưng vẫn sai), đó là dấu hiệu model đang dựa vào 1 quy tắc
        # nào đó SAI một cách hệ thống (giống kiểu "shortcut learning"), chứ
        # không phải chỉ là các ca khó/mập mờ (mập mờ thường đi kèm conf THẤP).

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
        # encoding="utf-8-sig": thêm BOM đầu file để Excel tự nhận đúng
        # encoding UTF-8 khi mở (không bị lỗi hiển thị tiếng Việt có dấu).
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

    # BƯỚC 1: liệt kê + gộp toàn bộ ảnh từ mọi thư mục cấu hình
    image_paths, labels, class_names = list_images(DATASET_DIRS)
    num_classes = len(class_names)
    print(f"Tổng số ảnh (đã gộp mọi thư mục): {len(image_paths)}")
    print(f"Các class: {class_names}")
    counts = np.bincount(labels, minlength=num_classes)
    print(f"Số ảnh mỗi class: {dict(zip(class_names, counts))}")

    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)

    # BƯỚC 2: tạo backbone đông lạnh (chỉ để trích đặc trưng, không train)
    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    # BƯỚC 3: trích embedding cho TOÀN BỘ ảnh, CHỈ 1 LẦN - đây là bước tốn
    # thời gian nhất (phải chạy CNN thật), nhưng chỉ chạy đúng 1 lần cho cả
    # dataset, không lặp lại theo epoch hay theo fold.
    print(f"\nĐang extract embedding cho toàn bộ {len(image_paths)} ảnh "
          f"({AUGMENT_COPIES} bản augment/ảnh, chỉ chạy 1 lần)...")
    t0 = time.time()
    augmented_emb_list, clean_emb_list = precompute_all_embeddings(
        backbone, image_paths, augment_transform, clean_transform, AUGMENT_COPIES
    )
    print(f"  -> Xong sau {time.time()-t0:.1f}s")

    # --------------------------------------------------------------------
    # BƯỚC 4: K-FOLD CROSS VALIDATION - đánh giá độ ổn định thật sự của model
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
        val_idx = np.where(fold_assignment == fold)[0]     # index các ảnh làm VAL ở fold này
        train_idx = np.where(fold_assignment != fold)[0]   # phần còn lại làm TRAIN

        # Ghép TOÀN BỘ bản augment (augment_copies bản/ảnh) của các ảnh
        # TRAIN thành 1 tensor lớn - mỗi bản augment được coi là 1 "mẫu
        # train" độc lập, cùng chung nhãn với ảnh gốc của nó.
        train_emb = torch.cat([augmented_emb_list[i] for i in train_idx], dim=0).to(DEVICE)
        train_lbl = torch.tensor(
            [labels_np[i] for i in train_idx for _ in range(AUGMENT_COPIES)], dtype=torch.long
        ).to(DEVICE)

        # VAL thì chỉ dùng đúng 1 embedding SẠCH (không augment) mỗi ảnh
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

        # Ghi lại kết quả từng ảnh trong fold này vào mảng oof_* dùng chung
        # cho TOÀN BỘ dataset (mỗi ảnh chỉ được ghi đúng 1 lần, vì mỗi ảnh
        # chỉ thuộc val_idx của ĐÚNG 1 fold trong 5 fold).
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
    # Độ lệch chuẩn (std) LỚN giữa các fold = model KHÔNG ổn định (kết quả
    # phụ thuộc nhiều vào việc ảnh nào rơi vào fold nào) - thường xảy ra khi
    # dataset quá nhỏ. Đừng chỉ nhìn accuracy trung bình, phải nhìn cả std.

    # --------------------------------------------------------------------
    # BƯỚC 5: BÁO CÁO CHI TIẾT - tổng hợp từ kết quả out-of-fold của TOÀN BỘ ảnh
    # --------------------------------------------------------------------
    print_detailed_report(image_paths, labels_np, oof_pred, oof_probs, oof_fold, class_names)

    # --------------------------------------------------------------------
    # BƯỚC 6: Train model SẢN XUẤT cuối cùng trên TOÀN BỘ dữ liệu (không giữ val)
    # --------------------------------------------------------------------
    print(f"\n=== Train model cuối cùng trên toàn bộ {len(image_paths)} ảnh (để deploy) ===")
    all_train_emb = torch.cat(augmented_emb_list, dim=0).to(DEVICE)
    all_train_lbl = torch.tensor(
        [labels_np[i] for i in range(len(image_paths)) for _ in range(AUGMENT_COPIES)], dtype=torch.long
    ).to(DEVICE)

    # Số epoch cho model cuối = ước lượng từ 50% HEAD_EPOCHS cấu hình (vì
    # không còn tập val riêng để early-stop theo dõi nữa, phải "đoán" trước
    # 1 con số hợp lý). Đây là điểm YẾU của cách làm: con số này KHÔNG dựa
    # trên epoch hội tụ THỰC TẾ đo được ở bước K-Fold, chỉ là ước lượng thô.
    final_epochs = max(50, int(HEAD_EPOCHS * 0.5))

    final_head, _, _, _ = train_head(
        all_train_emb, all_train_lbl, None, None, embedding_dim, num_classes,
        final_epochs, HEAD_LR, HEAD_DROPOUT, patience=final_epochs,  # không early stop, chạy hết
        print_epochs=PRINT_EPOCH_DETAILS, fold_label="Model cuối cùng"
    )

    # Gắn classifier head vừa train vào 1 model TOÀN VẸN (backbone + head)
    # để lưu ra file - xem giải thích ở build_full_model() phía trên.
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
        # Lưu kèm cả kết quả K-Fold vào checkpoint - để sau này (vd 6 tháng
        # sau) mở lại file .pt vẫn biết model này đã được đánh giá ra sao,
        # không cần nhớ/tra lại log console cũ.
    }, MODEL_OUT)
    print(f"Đã lưu model cuối cùng tại: {MODEL_OUT}")

    # --------------------------------------------------------------------
    # BƯỚC 7: Đo thông số edge (để biết model có đủ nhanh/nhẹ cho triển khai không)
    # --------------------------------------------------------------------
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = full_model(dummy_input)
    avg_latency_ms = (time.time() - start) / n_runs * 1000
    # Chạy 50 lần rồi lấy trung bình (không tính lần đầu tiên riêng - có
    # thể muốn "warm-up" 1 lần trước nếu muốn đo chính xác hơn, vì lần
    # forward đầu tiên trên GPU thường chậm hơn do khởi tạo kernel CUDA).
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)

    print(f"\n--- Thông số edge ---")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Latency trung bình (1 ảnh): {avg_latency_ms:.2f} ms")
    print(f"Kích thước file model: {model_size_mb:.2f} MB")
    print(f"K-Fold accuracy (độ tin cậy thật): {fold_accs.mean():.2%} ± {fold_accs.std():.2%}")


if __name__ == "__main__":
    main()