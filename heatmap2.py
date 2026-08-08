"""
Grad-CAM: trực quan hoá xem model classifier đang "nhìn" vào đâu trên ảnh để
đưa ra dự đoán - giúp kiểm tra model đã học đúng đặc trưng của sản phẩm (chữ,
logo, hình dạng...) hay đang học nhầm đặc trưng của NỀN xung quanh.

Ý TƯỞNG:
- Lấy activation (đầu ra) của lớp conv CUỐI CÙNG trong backbone.
- Lấy gradient của lớp đó đối với điểm số (logit) của class được dự đoán.
- Trung bình gradient theo từng kênh -> trọng số quan trọng của từng kênh.
- Tổng có trọng số của activation theo các kênh đó -> bản đồ nhiệt (CAM).
- Resize CAM về kích thước ảnh gốc, tô màu, chồng lên ảnh gốc.

Vùng ĐỎ/VÀNG = ảnh hưởng nhiều đến quyết định của model.
Vùng XANH DƯƠNG/tối = ít ảnh hưởng.

Cách chạy:
    python gradcam_heatmap.py
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ----------------------------
# 1. Cấu hình - SỬA Ở ĐÂY
# ----------------------------
MODEL_PATH = r"D:\TongHop\RTC Technologi\PCB\model\model1\edge_classifier_fewshot_mobilenet_v3_large.pt"

# Có thể là 1 FILE ảnh, hoặc 1 THƯ MỤC chứa nhiều ảnh (duyệt lần lượt)
IMAGE_PATH = r"D:\TongHop\RTC Technologi\9\test\NG"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Class mục tiêu để tính Grad-CAM. None = dùng class model DỰ ĐOÁN (mặc định,
# hợp lý nhất khi muốn xem "tại sao model đoán ra kết quả này").
# Đặt tên class cụ thể (vd: "type_1") nếu muốn xem model nhìn gì cho 1 class cố định.
TARGET_CLASS = "NG"

# --- Detect + crop trước khi đưa vào model (threshold cổ điển, không model AI) ---
# QUAN TRỌNG: model lúc train/chạy thật đều nhận ẢNH ĐÃ CROP, nên Grad-CAM
# cũng phải chạy trên đúng ảnh đã crop mới phản ánh đúng model thực tế.
USE_DETECT_CROP = False
METHOD = "threshold"     # "threshold" | "background_subtraction"
BACKGROUND_REF_PATH = ""  # chỉ dùng khi METHOD="background_subtraction"
DIFF_THRESHOLD = 10
INVERT_THRESHOLD = "auto"   # "auto" | True | False
MORPH_KERNEL_SIZE = 7
MIN_CONTOUR_AREA_RATIO = 0.01
MAX_CONTOUR_AREA_RATIO = 0.98
BORDER_TOUCH_MAX_RATIO = 0.3
CROP_PADDING = 15

PREVIEW_DISPLAY_MAX_WIDTH = 800
HEATMAP_ALPHA = 0.45   # độ đậm của heatmap khi chồng lên ảnh gốc (0-1)

# Transform PHẢI giống hệt lúc train/test (không augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ----------------------------
# 2. Detect + crop (threshold cổ điển, giống detect_and_crop_edge_lightweight.py)
# ----------------------------
_bg_gray = None


def _get_bg_gray():
    global _bg_gray
    if _bg_gray is None and BACKGROUND_REF_PATH and os.path.isfile(BACKGROUND_REF_PATH):
        bg = cv2.imread(BACKGROUND_REF_PATH)
        if bg is not None:
            _bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return _bg_gray


def _clean(mask):
    k = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def _border_touch_ratio(contour, shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    border = np.zeros((H, W), bool)
    border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True
    return np.count_nonzero((m > 0) & border) / border.sum()


def _best_contour(mask, img_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if MIN_CONTOUR_AREA_RATIO * img_area <= cv2.contourArea(c) <= MAX_CONTOUR_AREA_RATIO * img_area]
    if not valid:
        return None
    shape = mask.shape[:2]
    clean = [c for c in valid if _border_touch_ratio(c, shape) <= BORDER_TOUCH_MAX_RATIO]
    if not clean:
        clean = sorted(valid, key=lambda c: _border_touch_ratio(c, shape))[:1]
    return max(clean, key=cv2.contourArea)


def detect_bbox(img_bgr):
    """Trả về (x, y, w, h) hoặc None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    bg_gray = _get_bg_gray() if METHOD == "background_subtraction" else None

    if bg_gray is not None:
        if bg_gray.shape != gray.shape:
            bg_gray = cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]))
        diff = cv2.absdiff(gray, bg_gray)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        contour = _best_contour(_clean(mask), img_area)
    else:
        directions = [False, True] if INVERT_THRESHOLD == "auto" else [bool(INVERT_THRESHOLD)]
        candidates = []
        for inv in directions:
            t = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray, 0, 255, t + cv2.THRESH_OTSU)
            c = _best_contour(_clean(mask), img_area)
            if c is not None:
                candidates.append(c)
        contour = max(candidates, key=cv2.contourArea) if candidates else None

    return cv2.boundingRect(contour) if contour is not None else None


def crop_img(img_bgr, bbox, padding=CROP_PADDING):
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    return img_bgr[y1:y2, x1:x2]


# ----------------------------
# 3. Load model (giống test_model_report.py)
# ----------------------------
def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        embedding_dim = model.last_channel
        model.classifier[1] = nn.Linear(embedding_dim, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        embedding_dim = 960
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        embedding_dim = 576
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes))
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model, backbone_name


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    backbone_name = checkpoint.get("backbone_name", "mobilenet_v2")

    model, backbone_name = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Đã load model: {model_path}")
    print(f"Backbone: {backbone_name} | Class: {class_names}\n")
    return model, class_names, backbone_name


def get_target_layer(model, backbone_name):
    """Lớp conv cuối cùng trong backbone - nơi lấy activation để tính Grad-CAM.
    Với cả 3 kiến trúc mobilenet đang hỗ trợ, đó là conv cuối trong model.features."""
    return model.features[-1]


# ----------------------------
# 3. Grad-CAM
# ----------------------------
class GradCAM:
    """Grad-CAM dùng forward/backward hook - không cần thư viện ngoài."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Trọng số mỗi kênh = trung bình gradient theo chiều không gian (Global Average Pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)   # chỉ giữ phần ảnh hưởng DƯƠNG tới class đó

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()   # chuẩn hoá về 0-1
        return cam, output


# ----------------------------
# 4. Vẽ heatmap chồng lên ảnh gốc
# ----------------------------
def overlay_heatmap(img_bgr, cam, alpha=HEATMAP_ALPHA):
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return overlay


def _resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# ----------------------------
# 5. Xử lý 1 ảnh
# ----------------------------
def process_image(image_path, model, class_names, gradcam):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  Lỗi đọc ảnh: {image_path}")
        return

    bbox = None
    crop_bgr = img_bgr
    if USE_DETECT_CROP:
        bbox = detect_bbox(img_bgr)
        if bbox is not None:
            crop_bgr = crop_img(img_bgr, bbox)
        else:
            print(f"  CẢNH BÁO: không detect được sản phẩm trong {os.path.basename(image_path)} "
                  f"- dùng nguyên ảnh gốc.")

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(False)

    if TARGET_CLASS is not None:
        class_idx = class_names.index(TARGET_CLASS)
    else:
        with torch.no_grad():
            probs_tmp = F.softmax(model(input_tensor), dim=1)
        class_idx = probs_tmp.argmax(dim=1).item()

    cam, output = gradcam.compute(input_tensor, class_idx)
    probs = F.softmax(output, dim=1)[0]
    pred_class = class_names[class_idx]
    confidence = probs[class_idx].item()

    # Heatmap được tính trên ẢNH ĐÃ CROP (đúng thứ model thực sự nhìn thấy).
    # Vẽ khung xanh lên ảnh GỐC để biết vùng nào đã được crop đưa vào model.
    orig_with_bbox = img_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(orig_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 4)

    overlay = overlay_heatmap(crop_bgr, cam)
    label = f"{pred_class} ({confidence*100:.1f}%)"
    cv2.putText(overlay, label, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (255, 255, 255), 2)
    #cv2.putText(overlay, label, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Ghép 2 ảnh cạnh nhau: cần cùng chiều cao để hstack, resize theo chiều cao ảnh gốc
    target_h = orig_with_bbox.shape[0]
    overlay_resized = cv2.resize(overlay, (int(overlay.shape[1] * target_h / overlay.shape[0]), target_h))
    combined = np.hstack([orig_with_bbox, overlay_resized])
    # Resize xuong con 1 nua kich thuoc that (giu nguyen moi thu khac, chi
    # them dung 1 dong nay thay cho _resize_for_display truoc do):
    disp = cv2.resize(combined, None, fx=0.5, fy=0.5)

    print(f"  {os.path.basename(image_path)}: đoán '{pred_class}' ({confidence*100:.1f}%)"
          + ("" if bbox is not None else "  [không detect được]"))
    cv2.imshow("Trai = anh goc + bbox da crop | Phai = Grad-CAM tren anh crop (phim=qua anh, ESC=dung)", disp)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


# ----------------------------
# 6. Main
# ----------------------------
def main():
    model, class_names, backbone_name = load_model(MODEL_PATH)
    target_layer = get_target_layer(model, backbone_name)
    gradcam = GradCAM(model, target_layer)

    if os.path.isdir(IMAGE_PATH):
        files = sorted([f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(IMG_EXTENSIONS)])
        print(f"Tìm thấy {len(files)} ảnh trong: {IMAGE_PATH}\n")
        for fname in files:
            key = process_image(os.path.join(IMAGE_PATH, fname), model, class_names, gradcam)
            if key == 27:
                print("Đã nhấn ESC - dừng.")
                break
    elif os.path.isfile(IMAGE_PATH):
        process_image(IMAGE_PATH, model, class_names, gradcam)
    else:
        print(f"Không tìm thấy file/thư mục: {IMAGE_PATH}")


if __name__ == "__main__":
    main()