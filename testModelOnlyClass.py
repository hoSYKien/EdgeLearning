"""
TEST + HEATMAP cho one-class anomaly detection (one_class_anomaly_pipeline.py).

--------------------------------------------------------------------------
Y TUONG HEATMAP:
--------------------------------------------------------------------------
Memory bank trong pipeline chinh dung EMBEDDING TOAN ANH (da qua Global
Average Pooling - GAP) de quyet dinh OK/NG - phu hop de RA QUYET DINH
cuoi cung, nhung khong cho biet "vung nao trong anh trong bat thuong" vi
GAP da tron het khong gian lai roi.

De ve duoc heatmap, file nay xay THEM 1 "PATCH BANK" khac - thay vi 1
vector/anh (sau GAP), no luu lai NHIEU vector nho hon, MOI VECTOR UNG
VOI 1 VI TRI KHONG GIAN tren feature map (truoc GAP). Voi anh can kiem
tra: cung tach feature map, roi voi MOI vi tri, do khoang cach toi patch
GAN NHAT trong bank - vi tri nao xa bank (khac moi mau OK da thay) thi
diem cao = to mau do trong heatmap.

QUYET DINH OK/NG (nhan + % hien thi tren anh) VAN dung memory bank +
nguong da luu san trong checkpoint .pt cua pipeline chinh - heatmap chi
la lop TRUC QUAN THEM, khong thay doi quyet dinh cuoi cung.

Hien thi ket qua bang cv2.imshow (giong file Grad-CAM): bam PHIM BAT KY
de qua anh tiep theo, bam ESC de dung han.

Cach chay:
    python test_with_heatmap.py
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==========================================================================
# CONFIG - SUA O DAY
# ==========================================================================

# File .pt da luu tu one_class_anomaly_pipeline.py (chua bank GAP + nguong)
CHECKPOINT_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\model\anomaly_model_2\anomaly_bank_mobilenet_v3_large.pt"

# Thu muc chua anh OK dung de xay PATCH BANK rieng cho heatmap - NEN trung
# voi cac thu muc OK da dung luc train pipeline chinh.
OK_BANK_DIRS = [
    r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\train",
    r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\val",
]
OK_CLASS_NAME = "Pin_OK_masked"

# Gioi han so anh OK dung xay patch bank (heatmap khong can nhieu bang GAP
# bank - vai chuc anh la du, cang nhieu cang lau vi phai luu nhieu patch)
MAX_BANK_IMAGES_FOR_HEATMAP = 40

# Anh (hoac thu muc anh) can kiem tra + ve heatmap
TEST_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\test\Pin_NG_masked"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\train\output_pins\tonghop\test\heatmap_output"

K_NEIGHBORS_PATCH = 3   # so patch gan nhat dung tinh diem bat thuong cho 1 vi tri
HEATMAP_ALPHA = 0.45    # do dam cua heatmap khi chong len anh goc (0-1)
SHOW_ON_SCREEN = True   # True = hien cua so xem ket qua (phim bat ky = qua anh, ESC = dung)
PREVIEW_DISPLAY_MAX_WIDTH = 700   # thu nho cua so hien thi neu anh goc qua lon

# Tang/giam do sang - THUC (khong chi hien thi) - ap dung TRUOC KHI dua anh
# vao model, nen anh model "nhin thay" cung sang/toi giong het anh hien thi.
# BRIGHTNESS_FACTOR: he so NHAN do sang (1.0 = giu nguyen, >1 = sang hon,
#   vd 1.3 = sang hon 30%, <1 = toi hon, vd 0.8 = toi bot 20%).
# BRIGHTNESS_OFFSET: cong THEM 1 luong do sang co dinh (0 = giu nguyen,
#   vd 20 = sang hon deu 20 muc tren thang 0-255).
# Cong thuc: anh_moi = anh_cu * BRIGHTNESS_FACTOR + BRIGHTNESS_OFFSET
BRIGHTNESS_FACTOR = 1.3
BRIGHTNESS_OFFSET = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# Backbone + transform (giong pipeline chinh, tai dung nguyen)
# ==========================================================================

def build_backbone(backbone_name):
    if backbone_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif backbone_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif backbone_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Backbone khong duoc ho tro: {backbone_name}")
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def build_clean_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_gap_embedding(backbone, image_tensor):
    """Embedding TOAN ANH (sau GAP) - dung cho quyet dinh OK/NG, giong
    het pipeline chinh."""
    x = backbone.features(image_tensor)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


@torch.no_grad()
def extract_feature_map(backbone, image_tensor):
    """Feature map KHONG qua GAP - tra ve (1, C, Hf, Wf), dung de xay
    patch bank va tinh heatmap."""
    return backbone.features(image_tensor)


# ==========================================================================
# Xay PATCH BANK tu anh OK (dung rieng cho heatmap)
# ==========================================================================

def list_ok_images(dataset_dirs, ok_class_name):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for d in dataset_dirs:
        cls_dir = os.path.join(d, ok_class_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(exts):
                paths.append(os.path.join(cls_dir, fname))
    return paths


@torch.no_grad()
def build_patch_bank(backbone, ok_paths, clean_transform, max_images):
    """Voi moi anh OK, trich feature map (C, Hf, Wf), coi MOI VI TRI
    KHONG GIAN la 1 vector C-chieu doc lap, gop tat ca lai thanh 1 bang
    lon (N_patch_total, C)."""
    if len(ok_paths) > max_images:
        ok_paths = ok_paths[:max_images]

    all_patches = []
    feat_hw = None
    for path in ok_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_bgr = adjust_brightness(img_bgr, BRIGHTNESS_FACTOR, BRIGHTNESS_OFFSET)
        image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor = clean_transform(image).unsqueeze(0).to(DEVICE)
        feat = extract_feature_map(backbone, tensor)   # (1, C, Hf, Wf)
        feat_hw = feat.shape[-2:]
        c = feat.shape[1]
        patches = feat.permute(0, 2, 3, 1).reshape(-1, c)  # (Hf*Wf, C)
        all_patches.append(patches.cpu())

    bank = torch.cat(all_patches, dim=0)
    return bank, feat_hw


def patch_anomaly_map(query_feat, patch_bank, k):
    """query_feat: (1, C, Hf, Wf) cua 1 anh can kiem tra.
    Voi MOI vi tri (Hf, Wf), tinh trung binh khoang cach toi K patch gan
    nhat trong bank -> tra ve ma tran diem so (Hf, Wf) - CANG CAO cang
    bat thuong."""
    c = query_feat.shape[1]
    hf, wf = query_feat.shape[-2:]
    query_patches = query_feat.permute(0, 2, 3, 1).reshape(-1, c).cpu()

    query_n = F.normalize(query_patches, dim=1)
    bank_n = F.normalize(patch_bank, dim=1)

    dists = torch.cdist(query_n, bank_n)
    k_actual = min(k, bank_n.shape[0])
    nearest, _ = torch.topk(dists, k_actual, dim=1, largest=False)
    scores = nearest.mean(dim=1)
    return scores.reshape(hf, wf).numpy()


# ==========================================================================
# Ve heatmap chong len anh goc - dung cv2 (giong code Grad-CAM tham khao,
# khong dung matplotlib de tranh xung dot GUI backend tren Windows)
# ==========================================================================

def overlay_heatmap_cv2(img_bgr, score_map, alpha):
    """img_bgr: anh goc dang numpy BGR (doc bang cv2.imread).
    score_map: numpy array (Hf, Wf), CANG CAO cang bat thuong.
    Tra ve anh BGR da chong mau jet len."""
    smin, smax = score_map.min(), score_map.max()
    norm = (score_map - smin) / (smax - smin + 1e-8)   # chuan hoa ve [0,1]

    H, W = img_bgr.shape[:2]
    norm_resized = cv2.resize(norm.astype(np.float32), (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * norm_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return overlay


def score_to_display(score, threshold):
    """Doi diem khoang cach thanh nhan OK/NG + % hien thi tren anh - CHI
    de TRUC QUAN, khong dung so nay lam quyet dinh (quyet dinh that la
    score > threshold)."""
    ratio = score / threshold if threshold > 0 else 1.0
    if ratio <= 1.0:
        pct = 50.0 + (1.0 - ratio) * 50.0
        label = "OK"
    else:
        pct = 50.0 + min(ratio - 1.0, 1.0) * 50.0
        label = "NG"
    return label, min(pct, 99.9)


def adjust_brightness(img_bgr, factor, offset):
    """Tang/giam do sang THUC su cua anh (khong phai chi hien thi) - dung
    cv2.convertScaleAbs de nhan + cong roi tu dong cat ve dung khoang
    0-255 (khong bi tran/am gia tri)."""
    if factor == 1.0 and offset == 0:
        return img_bgr
    return cv2.convertScaleAbs(img_bgr, alpha=factor, beta=offset)


def _resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# ==========================================================================
# MAIN
# ==========================================================================

def process_one_image(image_path, backbone, clean_transform, gap_bank, threshold,
                       patch_bank, out_dir):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  Loi doc anh: {image_path}")
        return None, None
    img_bgr = adjust_brightness(img_bgr, BRIGHTNESS_FACTOR, BRIGHTNESS_OFFSET)

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = clean_transform(pil_img).unsqueeze(0).to(DEVICE)

    # --- Quyet dinh OK/NG (dung dung logic + bank cua pipeline chinh) ---
    gap_emb = extract_gap_embedding(backbone, tensor).cpu()
    gap_n = F.normalize(gap_emb, dim=1)
    bank_n = F.normalize(gap_bank, dim=1)
    dists = torch.cdist(gap_n, bank_n)
    k = min(5, bank_n.shape[0])
    nearest, _ = torch.topk(dists, k, dim=1, largest=False)
    image_score = nearest.mean(dim=1).item()

    label, pct = score_to_display(image_score, threshold)

    # --- Heatmap tu patch bank - ve tren anh da resize 224x224 (dung anh
    # model thuc su nhin thay) ---
    feat = extract_feature_map(backbone, tensor)
    score_map = patch_anomaly_map(feat, patch_bank, K_NEIGHBORS_PATCH)
    img_224_bgr = cv2.resize(img_bgr, (224, 224))
    overlay = overlay_heatmap_cv2(img_224_bgr, score_map, HEATMAP_ALPHA)

    text = f"{label} ({pct:.1f}%)"
    cv2.putText(overlay, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(overlay, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
    cv2.imwrite(out_path, overlay)

    print(f"  {os.path.basename(image_path)}: {label} ({pct:.1f}%) | "
          f"diem khoang cach thuc = {image_score:.4f} (nguong = {threshold:.4f}) "
          f"-> {out_path}")

    key = None
    if SHOW_ON_SCREEN:
        # Ghep anh goc (trai) + heatmap (phai) canh nhau - can cung chieu
        # cao de hstack, resize anh goc ve dung 224x224 cho khop overlay.
        combined = np.hstack([img_224_bgr, overlay])
        disp = _resize_for_display(combined, PREVIEW_DISPLAY_MAX_WIDTH * 2)
        window_title = (f"{os.path.basename(image_path)}  -  Trai = anh goc | "
                         f"Phai = heatmap  (phim bat ky = qua anh, ESC = dung)")
        cv2.imshow(window_title, disp)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_path, key


def main():
    print(f"Dang chay tren: {DEVICE}")

    if BRIGHTNESS_FACTOR != 1.0 or BRIGHTNESS_OFFSET != 0:
        print(f"[LUU Y] Dang chinh sang anh test (factor={BRIGHTNESS_FACTOR}, "
              f"offset={BRIGHTNESS_OFFSET}). Patch bank (heatmap) cung duoc xay lai "
              f"voi cung muc sang nay nen NHAT QUAN. Nhung GAP bank dung de quyet "
              f"dinh OK/NG (load tu checkpoint) da duoc xay tu TRUOC, KHONG co chinh "
              f"sang nay - neu anh goc cua ban von di toi, chinh sang o day co the lam "
              f"lech ket qua OK/NG so voi luc train. Neu muon dong bo hoan toan, nen "
              f"chinh sang ngay tu buoc chuan bi du lieu train, khong phai o day.")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    threshold = ckpt["threshold"]
    gap_bank = ckpt["bank_embeddings"].cpu()   # ep ve CPU cho dong nhat voi gap_emb
    backbone_name = ckpt["backbone_name"]
    print(f"Da load checkpoint: {CHECKPOINT_PATH}")
    print(f"  Backbone: {backbone_name} | Nguong: {threshold:.4f} | "
          f"So vector trong GAP bank: {gap_bank.shape[0]}")

    backbone = build_backbone(backbone_name)
    clean_transform = build_clean_transform()

    print(f"\nDang xay patch bank cho heatmap tu anh OK trong: {OK_BANK_DIRS}")
    ok_paths = list_ok_images(OK_BANK_DIRS, OK_CLASS_NAME)
    if not ok_paths:
        raise RuntimeError(
            f"Khong tim thay anh OK nao trong OK_BANK_DIRS voi ten class "
            f"'{OK_CLASS_NAME}'. Kiem tra lai duong dan / OK_CLASS_NAME."
        )
    patch_bank, feat_hw = build_patch_bank(backbone, ok_paths, clean_transform, MAX_BANK_IMAGES_FOR_HEATMAP)
    print(f"  -> Patch bank: {patch_bank.shape[0]} vector "
          f"(tu {min(len(ok_paths), MAX_BANK_IMAGES_FOR_HEATMAP)} anh OK, "
          f"feature map {feat_hw[0]}x{feat_hw[1]}/anh)")

    print(f"\nDang xu ly anh test: {TEST_PATH}")
    if os.path.isdir(TEST_PATH):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in sorted(os.listdir(TEST_PATH)) if f.lower().endswith(exts)]
        print(f"Tim thay {len(files)} anh trong: {TEST_PATH}\n")
        for fname in files:
            _, key = process_one_image(os.path.join(TEST_PATH, fname), backbone, clean_transform,
                                        gap_bank, threshold, patch_bank, OUTPUT_DIR)
            if key == 27:   # ESC
                print("Da nhan ESC - dung.")
                break
    else:
        process_one_image(TEST_PATH, backbone, clean_transform,
                           gap_bank, threshold, patch_bank, OUTPUT_DIR)

    print(f"\nXong. Anh heatmap da luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()