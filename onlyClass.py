"""
ONE-CLASS ANOMALY DETECTION - chi hoc "OK trong nhu the nao", khong can
hoc rieng tung loai loi (A/B/E/F/G/H/I...).

--------------------------------------------------------------------------
KHAC BIET CAN HIEU RO SO VOI PIPELINE MULTI-CLASS (few_shot_pipeline.py):
--------------------------------------------------------------------------
Multi-class:  anh -> embedding -> Linear(dim, num_classes) -> Softmax ->
              argmax ra dung 1 trong N class.
One-class:    anh -> embedding -> do KHOANG CACH toi "memory bank" (tap
              hop embedding cua cac anh OK) -> khoang cach lon = bat
              thuong = NG.

KHONG co buoc train trong so nao ca (khong Linear, khong Softmax, khong
optimizer). "Train" o day chi la buoc TRICH VA LUU embedding cua anh OK
lam "memory bank" - giong nhu lam 1 cuon so tay ghi lai "OK trong ra sao"
roi sau nay so sanh.

--------------------------------------------------------------------------
CACH CHIA DU LIEU (quan trong, khac han multi-class):
--------------------------------------------------------------------------
  - OK: chia 3 phan - TRAIN (xay memory bank), VAL (hieu chinh nguong),
    TEST (danh gia cuoi).
  - NG (moi loai loi A/B/E/F/G/H/I... gop chung lai la "NG"): CHI chia 2
    phan - VAL va TEST. KHONG dua NG vao TRAIN, vi memory bank chi luu
    OK, dua NG vao khong co tac dung gi (xem giai thich o cuoi file).
  - VAL dung de DO xem NG thuc te roi vao khoang cach bao nhieu, tu do
    CHON nguong phan tach OK/NG cho hop ly - khong the doan mu nguong
    duoc, phai co it nhat vai anh NG that de soi.

Cach chay:
    python one_class_anomaly_pipeline.py
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==========================================================================
# CONFIG - SUA O DAY CHO DUNG DATASET CUA BAN
# ==========================================================================

# Danh sach thu muc goc, moi thu muc co cac thu muc con la TEN CLASS
# (vd "OK", "A", "B", "E", "F", "G", "H", "I" theo bang ma loi cua ban).
DATASET_DIRS = [
    r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\train",
    r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\val",
]

OK_CLASS_NAME = "Pin_OK_masked"   # ten thu muc con duoc coi la "binh thuong" - SUA cho dung
                        # ten thu muc OK thuc te cua ban. Bat ky thu muc con
                        # nao KHAC ten nay deu duoc gop chung thanh "NG".

MODEL_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\model\anomaly_model_2"
BACKBONE_NAME = "mobilenet_v3_large"
MODEL_OUT = os.path.join(MODEL_DIR, f"anomaly_bank_{BACKBONE_NAME}.pt")

# Ty le chia du lieu OK thanh 3 phan train/val/test (cong lai = 1.0)
OK_TRAIN_RATIO = 0.70
OK_VAL_RATIO = 0.15
OK_TEST_RATIO = 0.15

# Ty le chia du lieu NG thanh 2 phan val/test (cong lai = 1.0) - KHONG co
# phan train vi NG khong duoc dua vao memory bank.
NG_VAL_RATIO = 0.5
NG_TEST_RATIO = 0.5

# Cau hinh augmentation - dung de LAM GIAU memory bank (nhieu ban embedding
# hoi khac nhau cho 1 anh OK), giup memory bank bao phu rong hon, khong chi
# dung cho val/test (val/test luon dung anh SACH, khong augment, de danh
# gia on dinh).
AUGMENT_CONFIG = {
    "horizontal_flip": True,
    "rotation_degrees": 0,
    "brightness_range": (0.8, 1.2),   # None de tat
    "random_resized_crop": False,
}
BANK_AUGMENT_COPIES = 20   # so ban augment/anh OK khi xay memory bank

K_NEIGHBORS = 5        # so lang gieng gan nhat dung de tinh diem bat thuong
                        # (trung binh khoang cach toi K anh OK gan nhat trong bank)
NORMALIZE_EMBEDDING = True   # chuan hoa embedding ve do dai 1 (cosine-like)
                              # truoc khi tinh khoang cach - thuong on dinh hon
                              # khoang cach Euclid tho, it bi anh huong boi
                              # do lon embedding dao dong giua cac anh.

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==========================================================================
# BUOC 1: doc du lieu + chia train/val/test
# ==========================================================================

def list_images_with_class(dataset_dirs):
    """Doc toan bo anh, tra ve image_paths (list[str]) va class_raw
    (list[str] - ten thu muc con goc, vd 'OK', 'A', 'B'...) - CHUA gop
    NG lai, de con giu lai thong tin loai loi cho bao cao chi tiet sau nay."""
    image_paths, class_raw = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for d in dataset_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Khong tim thay thu muc: {d}")
        for entry in os.scandir(d):
            if not entry.is_dir():
                continue
            cls = entry.name
            for fname in os.listdir(entry.path):
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(entry.path, fname))
                    class_raw.append(cls)

    return image_paths, class_raw


def split_dataset(image_paths, class_raw, ok_class_name,
                   ok_ratios, ng_ratios, seed):
    """Chia du lieu thanh train/val/test THEO DUNG NGUYEN TAC one-class:
      - OK: chia 3 phan theo ok_ratios (train, val, test)
      - NG (moi thu khac OK, giu nguyen ma loi goc): chia 2 phan theo
        ng_ratios (val, test) - KHONG co train.
    Tra ve dict co 6 key: ok_train, ok_val, ok_test, ng_val, ng_test -
    moi key la list (path, class_raw)."""
    rng = random.Random(seed)

    ok_items = [(p, c) for p, c in zip(image_paths, class_raw) if c == ok_class_name]
    ng_items = [(p, c) for p, c in zip(image_paths, class_raw) if c != ok_class_name]

    if len(ok_items) == 0:
        raise RuntimeError(
            f"Khong tim thay anh nao thuoc class OK ('{ok_class_name}'). "
            f"Kiem tra lai OK_CLASS_NAME co khop ten thu muc that khong."
        )
    if len(ng_items) == 0:
        raise RuntimeError(
            "Khong tim thay anh NG nao - can it nhat vai anh NG de hieu "
            "chinh nguong (val) va danh gia (test)."
        )

    rng.shuffle(ok_items)
    rng.shuffle(ng_items)

    n_ok = len(ok_items)
    n_ok_train = int(n_ok * ok_ratios[0])
    n_ok_val = int(n_ok * ok_ratios[1])
    ok_train = ok_items[:n_ok_train]
    ok_val = ok_items[n_ok_train:n_ok_train + n_ok_val]
    ok_test = ok_items[n_ok_train + n_ok_val:]

    n_ng = len(ng_items)
    n_ng_val = int(n_ng * ng_ratios[0])
    ng_val = ng_items[:n_ng_val]
    ng_test = ng_items[n_ng_val:]

    return {
        "ok_train": ok_train, "ok_val": ok_val, "ok_test": ok_test,
        "ng_val": ng_val, "ng_test": ng_test,
    }


# ==========================================================================
# BUOC 2: backbone + embedding (giong het pipeline multi-class, tai dung)
# ==========================================================================

def build_transforms(augment_cfg):
    aug_ops = [transforms.Resize((224, 224))]
    if augment_cfg.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation_degrees", 0) > 0:
        aug_ops.append(transforms.RandomRotation(augment_cfg["rotation_degrees"]))
    if augment_cfg.get("random_resized_crop"):
        aug_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    brightness_range = augment_cfg.get("brightness_range")
    if brightness_range:
        aug_ops.append(transforms.ColorJitter(brightness=brightness_range))
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
        raise ValueError(f"Backbone khong duoc ho tro: {backbone_name}")

    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, embedding_dim


@torch.no_grad()
def extract_embedding(backbone, image_tensor):
    x = backbone.features(image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


@torch.no_grad()
def embed_images_clean(backbone, image_paths, clean_transform):
    """Trich 1 embedding SACH (khong augment) cho moi anh - dung cho
    val/test, de danh gia on dinh khong bi nhieu boi augmentation ngau nhien."""
    embs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
        embs.append(extract_embedding(backbone, tensor).cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0)


@torch.no_grad()
def build_memory_bank(backbone, image_paths, augment_transform, copies):
    """Trich NHIEU ban embedding augment cho moi anh OK train, gop het lai
    thanh 1 memory bank lon (N_anh * copies, dim). Cang nhieu ban augment,
    memory bank cang bao phu rong cac bien the anh sang/goc chup cua OK -
    giup giam nham OK bi bao dong gia (false positive) chi vi anh sang khac
    1 chut so voi luc chup mau."""
    all_embs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensors = torch.stack([augment_transform(image) for _ in range(copies)]).to(DEVICE)
        embs = extract_embedding(backbone, tensors)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0) if all_embs else torch.empty(0)


# ==========================================================================
# BUOC 3: tinh diem bat thuong (khoang cach k-NN toi memory bank)
# ==========================================================================

def maybe_normalize(x):
    if NORMALIZE_EMBEDDING:
        return F.normalize(x, dim=1)
    return x


def knn_anomaly_score(query_embeddings, bank_embeddings, k):
    """Voi moi embedding trong query_embeddings, tinh diem bat thuong =
    trung binh khoang cach Euclid toi K embedding GAN NHAT trong bank.
    Diem CANG CAO = anh CANG khac cum OK = cang co kha nang la NG."""
    query = maybe_normalize(query_embeddings)
    bank = maybe_normalize(bank_embeddings)

    dists = torch.cdist(query, bank)   # (n_query, n_bank)
    k_actual = min(k, bank.shape[0])
    nearest, _ = torch.topk(dists, k_actual, dim=1, largest=False)
    return nearest.mean(dim=1)   # (n_query,)


# ==========================================================================
# BUOC 4: hieu chinh nguong bang tap VAL (co ca OK va NG that)
# ==========================================================================

def calibrate_threshold(val_scores, val_is_ng):
    """Quet qua nhieu muc nguong, tinh TPR (ty le bat dung NG) va FPR (ty
    le bao nham OK thanh NG) tai moi nguong, in bang ra man hinh de nguoi
    dung tu chon, dong thoi TU DONG chon 1 nguong mac dinh theo tieu chi
    Youden's J (TPR - FPR lon nhat) - can bang giua bat du NG va it bao
    nham OK. Nguoi dung co the doi lai nguong nay thu cong sau khi xem
    bang neu muon uu tien khac (vd uu tien khong bo lot NG hon la it bao
    nham)."""
    val_scores = np.asarray(val_scores)
    val_is_ng = np.asarray(val_is_ng)

    n_ok = (~val_is_ng).sum()
    n_ng = val_is_ng.sum()
    if n_ok == 0 or n_ng == 0:
        raise RuntimeError(
            "Tap VAL can co CA anh OK lan anh NG de hieu chinh nguong - "
            "kiem tra lai OK_VAL_RATIO / NG_VAL_RATIO."
        )

    candidate_thresholds = np.unique(val_scores)
    best_j, best_thresh = -1.0, candidate_thresholds[0]
    print(f"\n{'Nguong':>10}{'TPR (bat NG)':>16}{'FPR (bao nham OK)':>20}{'Youden J':>12}")
    print("-" * 60)
    for thresh in candidate_thresholds:
        pred_ng = val_scores > thresh
        tpr = (pred_ng & val_is_ng).sum() / n_ng
        fpr = (pred_ng & ~val_is_ng).sum() / n_ok
        j = tpr - fpr
        print(f"{thresh:>10.4f}{tpr:>16.2%}{fpr:>20.2%}{j:>12.3f}")
        if j > best_j:
            best_j, best_thresh = j, thresh

    print(f"\n>>> Nguong tu dong chon (Youden's J lon nhat): {best_thresh:.4f}")
    print(">>> Neu muon UU TIEN khong bo lot NG (chap nhan bao nham OK nhieu hon),")
    print(">>> chon 1 nguong THAP hon trong bang tren (TPR cao hon). Neu muon")
    print(">>> UU TIEN it bao nham OK, chon nguong CAO hon (chap nhan bo lot NG nhieu hon).")
    return float(best_thresh)


# ==========================================================================
# BUOC 5: danh gia tren tap TEST (chua tung dung de hieu chinh nguong)
# ==========================================================================

def evaluate_test(test_scores, test_is_ng, test_class_raw, threshold):
    test_scores = np.asarray(test_scores)
    test_is_ng = np.asarray(test_is_ng)
    pred_ng = test_scores > threshold

    n_ok = (~test_is_ng).sum()
    n_ng = test_is_ng.sum()
    tpr = (pred_ng & test_is_ng).sum() / n_ng if n_ng else float("nan")
    fpr = (pred_ng & ~test_is_ng).sum() / n_ok if n_ok else float("nan")
    acc = (pred_ng == test_is_ng).mean()

    print("\n" + "=" * 70)
    print("KET QUA TREN TAP TEST (chua tung dung de hieu chinh nguong)")
    print("=" * 70)
    print(f"Accuracy tong (OK vs NG): {acc:.2%}")
    print(f"TPR - ty le bat dung NG (recall NG):  {tpr:.2%}")
    print(f"FPR - ty le bao nham OK thanh NG:      {fpr:.2%}")

    # Bao cao chi tiet theo TUNG LOAI LOI (A/B/E/F/G/H/I...) - quan trong
    # de biet loai loi nao dang bi bo lot nhieu nhat, khong chi nhin 1 con
    # so TPR tong gop chung tat ca loai loi lai.
    print("\nChi tiet theo tung loai loi (chi tinh tren cac anh NG trong test):")
    print(f"{'Ma loi':<10}{'So anh':>8}{'Bat dung':>10}{'Recall':>10}")
    print("-" * 40)
    ng_codes = sorted(set(c for c, is_ng in zip(test_class_raw, test_is_ng) if is_ng))
    for code in ng_codes:
        idx = [i for i, c in enumerate(test_class_raw) if c == code]
        total = len(idx)
        caught = sum(1 for i in idx if pred_ng[i])
        recall = caught / total if total else 0.0
        print(f"{code:<10}{total:>8}{caught:>10}{recall:>10.2%}")


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

def main():
    print(f"Dang chay tren: {DEVICE}")
    print(f"Backbone: {BACKBONE_NAME}")

    image_paths, class_raw = list_images_with_class(DATASET_DIRS)
    print(f"Tong so anh: {len(image_paths)}")
    print(f"Cac class tim thay: {sorted(set(class_raw))}")

    split = split_dataset(
        image_paths, class_raw, OK_CLASS_NAME,
        (OK_TRAIN_RATIO, OK_VAL_RATIO, OK_TEST_RATIO),
        (NG_VAL_RATIO, NG_TEST_RATIO),
        seed=RANDOM_SEED,
    )
    print(f"\nChia du lieu:")
    print(f"  OK  -> train: {len(split['ok_train'])}, val: {len(split['ok_val'])}, test: {len(split['ok_test'])}")
    print(f"  NG  -> val: {len(split['ng_val'])}, test: {len(split['ng_test'])}")

    augment_transform, clean_transform = build_transforms(AUGMENT_CONFIG)
    backbone, embedding_dim = build_backbone(BACKBONE_NAME)
    print(f"Embedding dimension: {embedding_dim}")

    # --- Xay memory bank tu OK train (co augment de lam giau) ---
    print(f"\nDang xay memory bank tu {len(split['ok_train'])} anh OK "
          f"({BANK_AUGMENT_COPIES} ban augment/anh)...")
    ok_train_paths = [p for p, c in split["ok_train"]]
    bank_embeddings = build_memory_bank(backbone, ok_train_paths, augment_transform, BANK_AUGMENT_COPIES)
    print(f"  -> Memory bank co {bank_embeddings.shape[0]} vector embedding.")

    # --- Trich embedding SACH cho val/test (OK + NG) ---
    def embed_split(items):
        paths = [p for p, c in items]
        classes = [c for p, c in items]
        embs = embed_images_clean(backbone, paths, clean_transform)
        return embs, classes

    ok_val_emb, ok_val_cls = embed_split(split["ok_val"])
    ng_val_emb, ng_val_cls = embed_split(split["ng_val"])
    ok_test_emb, ok_test_cls = embed_split(split["ok_test"])
    ng_test_emb, ng_test_cls = embed_split(split["ng_test"])

    # --- Tinh diem bat thuong cho val, hieu chinh nguong ---
    val_emb = torch.cat([ok_val_emb, ng_val_emb], dim=0)
    val_is_ng = np.array([False] * len(ok_val_cls) + [True] * len(ng_val_cls))
    val_scores = knn_anomaly_score(val_emb, bank_embeddings, K_NEIGHBORS).numpy()

    threshold = calibrate_threshold(val_scores, val_is_ng)

    # --- Danh gia tren test (chua dung de hieu chinh nguong) ---
    test_emb = torch.cat([ok_test_emb, ng_test_emb], dim=0)
    test_is_ng = np.array([False] * len(ok_test_cls) + [True] * len(ng_test_cls))
    test_class_raw = ok_test_cls + ng_test_cls
    test_scores = knn_anomaly_score(test_emb, bank_embeddings, K_NEIGHBORS).numpy()

    evaluate_test(test_scores, test_is_ng, test_class_raw, threshold)

    # --- Luu memory bank + nguong de deploy ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "bank_embeddings": bank_embeddings,
        "threshold": threshold,
        "k_neighbors": K_NEIGHBORS,
        "normalize_embedding": NORMALIZE_EMBEDDING,
        "backbone_name": BACKBONE_NAME,
        "ok_class_name": OK_CLASS_NAME,
    }, MODEL_OUT)
    print(f"\nDa luu memory bank + nguong tai: {MODEL_OUT}")
    print("Dung ham score_new_image() ben duoi (hoac load lai file .pt nay o "
          "noi khac) de kiem tra anh moi luc deploy.")


@torch.no_grad()
def score_new_image(image_path, checkpoint_path):
    """VI DU ham dung luc DEPLOY: load lai memory bank + nguong da luu,
    cham diem 1 anh moi, tra ve (is_ng: bool, score: float, threshold: float)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    backbone, _ = build_backbone(ckpt["backbone_name"])
    _, clean_transform = build_transforms(AUGMENT_CONFIG)

    image = Image.open(image_path).convert("RGB")
    tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
    emb = extract_embedding(backbone, tensor).cpu()

    global NORMALIZE_EMBEDDING
    NORMALIZE_EMBEDDING = ckpt["normalize_embedding"]
    score = knn_anomaly_score(emb, ckpt["bank_embeddings"], ckpt["k_neighbors"]).item()
    threshold = ckpt["threshold"]
    return score > threshold, score, threshold


if __name__ == "__main__":
    main()

    # VI DU dung score_new_image() sau khi da train xong (BO COMMENT va sua
    # duong dan de dung thu):
    #
    # is_ng, score, threshold = score_new_image(
    #     r"D:\duong_dan\toi\anh_can_kiem_tra.bmp", MODEL_OUT
    # )
    # print(f"NG: {is_ng} | score={score:.4f} | threshold={threshold:.4f}")