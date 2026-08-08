"""
Script inference cho model được train bằng few_shot_pipeline_multicrop.py

LÝ DO CẦN FILE NÀY (đừng dùng script test cũ): model mới được train với 1
chuỗi tiền xử lý cụ thể - 4-crop (theo CUSTOM_REGIONS, hoặc grid) -> resize
INPUT_SIZE -> extract embedding từng vùng -> MAX-POOL các embedding thành 1 ->
đưa qua classifier. Nếu lúc test chỉ resize thẳng cả ảnh như kiểu cũ, embedding
sẽ lệch hẳn phân phối so với lúc train -> dự đoán sai lệch toàn bộ (đây chính
là nguyên nhân ảnh OK bị đoán nhầm gần như 100% ở lần test trước).

File này đọc lại ĐÚNG các config đã lưu trong checkpoint (multi_crop_config,
custom_regions, input_size) - không cần tự tay khớp lại tay, tránh lệch.

Chỉ cần sửa phần CONFIG bên dưới rồi chạy:
    python infer_multicrop.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from PIL import Image, ImageDraw

# ==========================================================================
# CONFIG - SỬA Ở ĐÂY
# ==========================================================================

MODEL_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\model\model2\edge_classifier_fewshot_mobilenet_v3_large_multicrop.pt"

# Trỏ tới 1 FILE ảnh HOẶC 1 THƯ MỤC (script tự nhận diện dựa vào đường dẫn).
INPUT_PATH = r"D:\TongHop\RTC Technologi\HZT.Bottom\test\NG"

# Chỉ có tác dụng khi INPUT_PATH là thư mục. True nếu thư mục đó có các thư
# mục con đặt tên đúng theo class (vd NG/, OK/) và bạn muốn đối chiếu nhãn
# thật + in bảng accuracy (giống format báo cáo cũ). False nếu chỉ muốn liệt
# kê dự đoán cho từng ảnh, không cần biết nhãn thật.
USE_REPORT_MODE = False

# Xuất thêm ảnh Grad-CAM (khớp đúng cơ chế multi-crop + max-pool) cho mỗi ảnh.
ENABLE_HEATMAP = True
HEATMAP_OUT_DIR = r"D:\TongHop\RTC Technologi\HZT.Bottom\heatmap_out"
# True: mở ảnh heatmap trực tiếp bằng trình xem ảnh mặc định của Windows ngay
# sau khi xử lý xong mỗi ảnh, thay vì chỉ lưu lặng lẽ vào HEATMAP_OUT_DIR.
# Tắt nếu xử lý hàng loạt nhiều ảnh (--report) để tránh mở quá nhiều cửa sổ.
SHOW_ON_SCREEN = True

# ==========================================================================
# Từ đây trở xuống là LOGIC CHUNG - không cần sửa
# ==========================================================================


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================================================================
# Các hàm crop - COPY Y HỆT logic từ few_shot_pipeline_multicrop.py để đảm
# bảo khớp tuyệt đối với lúc train. Nếu sau này sửa crop trong file train,
# nhớ sửa lại y hệt ở đây.
# ==========================================================================
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


def get_custom_region_boxes(image_size, custom_regions):
    """Trả về list (ten_vung, (x1,y1,x2,y2)) theo pixel - dịch từ CUSTOM_REGIONS
    (fraction 0-1, lưu trong checkpoint) sang kích thước ảnh thực tế. Copy y hệt
    logic từ few_shot_pipeline_multicrop.py để đảm bảo khớp lúc train."""
    w, h = image_size
    boxes = []
    for name, (fx0, fy0, fx1, fy1) in custom_regions.items():
        x1, y1 = int(fx0 * w), int(fy0 * h)
        x2, y2 = int(fx1 * w), int(fy1 * h)
        boxes.append((name, (x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))))
    return boxes


def get_fixed_crop_regions(image, crop_scale, positions):
    w, h = image.size
    cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
    return [image.crop(_crop_box(w, h, cw, ch, pos)) for pos in positions]


def build_clean_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


EMBEDDING_DIMS = {"mobilenet_v2": 1280, "mobilenet_v3_large": 960, "mobilenet_v3_small": 576}


def build_bare_backbone(backbone_name):
    """weights=None: không tải pretrained từ mạng - không cần, vì toàn bộ
    trọng số (kể cả phần backbone đông lạnh) đã nằm sẵn trong checkpoint."""
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    backbone_name = ckpt["backbone_name"]
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    embedding_dim = EMBEDDING_DIMS[backbone_name]

    full_model = build_bare_backbone(backbone_name)
    # Gắn lại classifier ĐÚNG shape đã lưu (Dropout không có tham số nên giá
    # trị dropout ở đây không cần khớp - model.eval() tắt dropout khi infer).
    full_model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    full_model.load_state_dict(ckpt["model_state"])
    full_model = full_model.to(device)
    full_model.eval()

    mc_cfg = ckpt.get("multi_crop_config", {"enable": False})
    custom_regions = ckpt.get("custom_regions", {})
    input_size = ckpt.get("input_size", 224)

    if "multi_crop_config" not in ckpt:
        print("[CẢNH BÁO] Checkpoint này không có multi_crop_config "
              "(model cũ, train trước khi thêm tính năng này) - sẽ chạy chế độ full-image thường.")
    if mc_cfg.get("mode") == "custom" and not custom_regions:
        print("[CẢNH BÁO] mode='custom' nhưng checkpoint không có custom_regions - "
              "model cũ train trước khi thêm custom regions, kết quả sẽ sai lệch.")

    return full_model, class_names, mc_cfg, custom_regions, input_size


@torch.no_grad()
def predict_image(image_path, model, class_names, mc_cfg, custom_regions, input_size, device):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    clean_transform = build_clean_transform(input_size)
    use_multi_crop = mc_cfg.get("enable") and min(w, h) >= mc_cfg.get("min_source_size", 0)

    if use_multi_crop and mc_cfg.get("mode") == "custom":
        boxes = get_custom_region_boxes((w, h), custom_regions)
        regions = [image.crop(box) for _, box in boxes]
    elif use_multi_crop:
        regions = get_fixed_crop_regions(image, mc_cfg["crop_scale"], mc_cfg["positions"])
    else:
        regions = [image]

    tensors = torch.stack([clean_transform(r) for r in regions]).to(device)

    features = model.features(tensors)
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    embeddings = torch.flatten(pooled, 1)                      # (num_regions, dim)
    final_embedding = embeddings.max(dim=0, keepdim=True).values  # (1, dim) - khớp lúc train

    logits = model.classifier(final_embedding)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())

    return class_names[pred_idx], {c: float(probs[i]) for i, c in enumerate(class_names)}


def apply_jet_colormap(gray):
    """gray: mảng float (H,W) trong [0,1] -> ảnh RGB uint8 kiểu 'jet' (xanh->đỏ).
    Viết tay bằng numpy để không cần cài thêm matplotlib."""
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compute_multicrop_gradcam(image_path, model, class_names, mc_cfg, custom_regions, input_size, device):
    """Grad-CAM ĐÚNG với cơ chế multi-crop + max-pool đang dùng: forward các
    vùng (custom hoặc grid, hoặc 1 vùng nếu multi-crop tắt) trong CÙNG 1 đồ
    thị tính toán, backward từ logit của class dự đoán -> gradient tự động
    chỉ chảy về (các) vùng THỰC SỰ đóng góp vào giá trị max ở từng chiều
    embedding. Vùng nào "thua" phép max ở toàn bộ chiều sẽ gần như không
    nhận được gradient -> heatmap mờ đúng nghĩa.

    Trả về: pred_class, probs (dict), roi_image (PIL, ảnh gốc),
    combined_heatmap (numpy (H,W) trong [0,1], cùng kích thước roi_image),
    region_boxes (list các (ten_vung, (x1,y1,x2,y2)) trong tọa độ roi_image).
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    clean_transform = build_clean_transform(input_size)
    use_multi_crop = mc_cfg.get("enable") and min(w, h) >= mc_cfg.get("min_source_size", 0)

    if use_multi_crop and mc_cfg.get("mode") == "custom":
        boxes = get_custom_region_boxes((w, h), custom_regions)
    elif use_multi_crop:
        positions = mc_cfg["positions"]
        crop_scale = mc_cfg["crop_scale"]
        cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
        boxes = [(pos, _crop_box(w, h, cw, ch, pos)) for pos in positions]
    else:
        boxes = [("full", (0, 0, w, h))]

    regions = [image.crop(box) for _, box in boxes]
    tensors = torch.stack([clean_transform(r) for r in regions]).to(device)

    # Tách khỏi đồ thị backbone (đông lạnh, requires_grad=False) rồi bật lại
    # grad cho chính tensor feature map - đây là cách chuẩn để làm Grad-CAM
    # trên backbone đông lạnh.
    features = model.features(tensors)                      # (N_vung, C, H', W')
    features = features.detach().requires_grad_(True)
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    embeddings = torch.flatten(pooled, 1)                    # (N_vung, C)
    final_embedding = embeddings.max(dim=0, keepdim=True).values  # (1, C) - khớp lúc train/infer

    logits = model.classifier(final_embedding)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())

    model.zero_grad()
    logits[0, pred_idx].backward()
    grads = features.grad                                    # (N_vung, C, H', W')

    canvas = np.zeros((h, w), dtype=np.float32)
    for i, (_, (x1, y1, x2, y2)) in enumerate(boxes):
        weights = grads[i].mean(dim=(1, 2))                  # (C,) - GAP của gradient, đúng chuẩn Grad-CAM
        cam = F.relu((weights.view(-1, 1, 1) * features[i]).sum(dim=0))  # (H', W')
        cam = cam.detach()
        region_w, region_h = x2 - x1, y2 - y1
        cam_resized = F.interpolate(cam.view(1, 1, *cam.shape), size=(region_h, region_w),
                                     mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        # Ghép vào canvas chung bằng MAX - nhất quán với việc model cũng lấy
        # max giữa các vùng để ra quyết định cuối cùng.
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], cam_resized)

    if canvas.max() > 1e-8:
        canvas = canvas / canvas.max()

    return (class_names[pred_idx], {c: float(probs[j]) for j, c in enumerate(class_names)},
            image, canvas, boxes)


def save_gradcam_visualization(roi_image, heatmap, pred_class, confidence, boxes, out_path, alpha=0.45, show=False):
    """Tạo ảnh 2 tấm ghép ngang giống layout công cụ cũ:
    Trái = ảnh gốc + khung các vùng | Phải = heatmap overlay.
    Nếu show=True, mở ảnh bằng trình xem ảnh mặc định của hệ điều hành."""
    w, h = roi_image.size

    left = roi_image.copy()
    draw = ImageDraw.Draw(left)
    for pos, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 0, 0), width=2)

    heat_rgb = apply_jet_colormap(heatmap)
    base_rgb = np.array(roi_image).astype(np.float32)
    overlay_arr = (alpha * heat_rgb.astype(np.float32) + (1 - alpha) * base_rgb).astype(np.uint8)
    right = Image.fromarray(overlay_arr)
    draw_r = ImageDraw.Draw(right)
    draw_r.text((10, 10), f"{pred_class} ({confidence:.1%})", fill=(255, 255, 255))

    combined = Image.new("RGB", (w * 2 + 8, h), (30, 30, 30))
    combined.paste(left, (0, 0))
    combined.paste(right, (w + 8, 0))
    combined.save(out_path)
    if show:
        combined.show()



def run_flat_listing(paths, model, class_names, mc_cfg, custom_regions, input_size, device, heatmap_dir=None):
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    for p in paths:
        if heatmap_dir:
            pred, probs, roi_image, heatmap, boxes = compute_multicrop_gradcam(
                p, model, class_names, mc_cfg, custom_regions, input_size, device)
            out_path = os.path.join(heatmap_dir, os.path.splitext(os.path.basename(p))[0] + "_cam.png")
            save_gradcam_visualization(roi_image, heatmap, pred, probs[pred], boxes, out_path, show=SHOW_ON_SCREEN)
        else:
            pred, probs = predict_image(p, model, class_names, mc_cfg, custom_regions, input_size, device)
        probs_str = " | ".join(f"{c}: {v:.1%}" for c, v in probs.items())
        print(f"{os.path.basename(p):<30} -> {pred:<6} ({probs_str})")


def run_report(root_dir, model, class_names, mc_cfg, custom_regions, input_size, device, heatmap_dir=None):
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    total, correct = 0, 0
    for true_cls in class_names:
        cls_dir = os.path.join(root_dir, true_cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS))
        if not files:
            continue

        print(f"\n--- Thư mục: {true_cls} ---")
        header = f"{'File':<28}{'Đúng/Sai':<10}" + "".join(f"{c:>10}" for c in class_names)
        print(header)
        print("-" * len(header))

        for fname in files:
            path = os.path.join(cls_dir, fname)
            if heatmap_dir:
                pred, probs, roi_image, heatmap, boxes = compute_multicrop_gradcam(
                    path, model, class_names, mc_cfg, custom_regions, input_size, device)
                out_name = f"{true_cls}_{os.path.splitext(fname)[0]}_cam.png"
                save_gradcam_visualization(roi_image, heatmap, pred, probs[pred], boxes,
                                            os.path.join(heatmap_dir, out_name), show=SHOW_ON_SCREEN)
            else:
                pred, probs = predict_image(path, model, class_names, mc_cfg, custom_regions, input_size, device)
            status = "Đúng" if pred == true_cls else "SAI"
            total += 1
            correct += int(pred == true_cls)
            row = f"{fname:<28}{status:<10}"
            for c in class_names:
                row += f"{probs[c]:>9.1%} "
            print(row)

    if total:
        print(f"\nTổng: {correct}/{total} đúng ({correct/total:.1%})")
    else:
        print("Không tìm thấy thư mục con nào khớp tên class để đối chiếu nhãn thật "
              f"(class_names={class_names}). Dùng --dir không kèm --report để chỉ liệt kê dự đoán.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, mc_cfg, custom_regions, input_size = load_model(MODEL_PATH, device)

    print(f"Đang chạy trên: {device}")
    print(f"Classes: {class_names} | Input size: {input_size}")
    print(f"Multi-crop: {mc_cfg}")
    if mc_cfg.get("mode") == "custom":
        print(f"Custom regions: {list(custom_regions.keys())}")
    if ENABLE_HEATMAP:
        print(f"Heatmap: BẬT -> lưu vào {HEATMAP_OUT_DIR}/")

    heatmap_dir = HEATMAP_OUT_DIR if ENABLE_HEATMAP else None

    if os.path.isfile(INPUT_PATH):
        run_flat_listing([INPUT_PATH], model, class_names, mc_cfg, custom_regions,
                          input_size, device, heatmap_dir)
    elif os.path.isdir(INPUT_PATH):
        if USE_REPORT_MODE:
            run_report(INPUT_PATH, model, class_names, mc_cfg, custom_regions,
                        input_size, device, heatmap_dir)
        else:
            paths = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.lower().endswith(IMAGE_EXTS)]
            run_flat_listing(paths, model, class_names, mc_cfg, custom_regions,
                              input_size, device, heatmap_dir)
    else:
        raise FileNotFoundError(f"Không tìm thấy INPUT_PATH: {INPUT_PATH}")


if __name__ == "__main__":
    main()