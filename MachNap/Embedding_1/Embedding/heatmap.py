"""
LIVE crop + Grad-CAM:
  - Camera Hikrobot (MVS) chạy live.
  - Học nền lúc khởi động (giữ khung hình TRỐNG).
  - Bấm 'c': chụp frame -> background subtraction -> crop bbox quanh vật
             -> đưa ảnh crop vào classifier -> Grad-CAM xem model nhìn đâu.
  - CHỈ crop box (giữ nguyên nền trong box), KHÔNG xoá nền.

PHÍM:
  c     : chụp + detect + crop + Grad-CAM
  r     : học lại nền (khi đổi setup / nền trôi)
  s     : (màn hình kết quả) lưu ảnh kết quả
  q/ESC : thoát

Cách chạy:
    python live_crop_gradcam.py
"""

import sys
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# ============================================================
#  KHỞI TẠO SDK HIKROBOT (MVS)
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(dll_dir)
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)
from MvCameraControl_class import *


# ============================================================
# 1. CẤU HÌNH - SỬA Ở ĐÂY
# ============================================================
MODEL_PATH = r"D:\TongHop\RTC Technologi\G8\model\model2\edge_classifier_fewshot_mobilenet_v3_large.pt"
SAVE_DIR = r"D:\TongHop\RTC Technologi\G8\gradcam_captures"

CAMERA_MODEL_HINT = "MV-CS200"   # để "" nếu lấy camera đầu tiên
RESIZE_DIV = 4                   # thu nhỏ frame trước khi detect cho nhẹ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CLASS = None              # None = dùng class model dự đoán

# --- Ngưỡng background subtraction (mượn từ pipeline băng tải) ---
THRESH_STRONG = 35
THRESH_WEAK = 20
CHROMA_TOL = 0.04
SHADOW_DIM = (0.3, 0.95)
SHADOW_MAX_DIFF = 70

MIN_AREA_RATIO = 0.002           # blob nhỏ hơn tỉ lệ này của frame -> bỏ (nhiễu)
MAX_AREA_RATIO = 0.95
MORPH_OPEN_KSIZE = (5, 5)
MORPH_CLOSE_KSIZE = (7, 7)
FILL_HOLES = True
CROP_PADDING = 15                # đệm quanh bbox (theo pixel ảnh nhỏ)

# --- Học nền ---
WARMUP_FRAMES = 30
BG_FRAMES = 40

# --- Hiển thị ---
HEATMAP_ALPHA = 0.45
RESULT_DISPLAY_SCALE = 0.5
DISPLAY_MAX_W, DISPLAY_MAX_H = 1280, 720

# Transform PHẢI giống hệt lúc train/test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 2. WRAPPER CAMERA HIKROBOT
# ============================================================
class HikrobotCamera:
    def __init__(self, model_hint=CAMERA_MODEL_HINT):
        self.model_hint = model_hint
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = 0
        self.stFrameInfo = None
        self._convert_buf = None

    def open(self):
        print("SDK Version:", hex(MvCamera.MV_CC_GetSDKVersion()))
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print("Không tìm thấy camera nào! Mã lỗi:", ret)
            return False
        print(f"Tìm thấy {deviceList.nDeviceNum} camera")

        target_index = 0
        for i in range(deviceList.nDeviceNum):
            info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType == MV_GIGE_DEVICE:
                name = "".join(chr(c) for c in info.SpecialInfo.stGigEInfo.chModelName if c != 0)
                print(f"[{i}] GigE Camera: {name}")
            elif info.nTLayerType == MV_USB_DEVICE:
                name = "".join(chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
                print(f"[{i}] USB Camera: {name}")
            else:
                name = ""
            if self.model_hint and self.model_hint in name:
                target_index = i

        stDeviceInfo = cast(deviceList.pDeviceInfo[target_index],
                            POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        if self.cam.MV_CC_CreateHandle(stDeviceInfo) != 0:
            print("Tạo handle lỗi.")
            return False
        if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            print("Mở camera lỗi.")
            self.cam.MV_CC_DestroyHandle()
            return False
        if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if nPacketSize > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        return True

    def start(self):
        if self.cam.MV_CC_StartGrabbing() != 0:
            print("Start grabbing lỗi.")
            return False
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self.nPayloadSize = stParam.nCurValue
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        return True

    def read(self):
        """Lấy 1 frame -> BGR full-res. Trả None nếu lỗi frame."""
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
        if ret != 0:
            return None

        w = self.stFrameInfo.nWidth
        h = self.stFrameInfo.nHeight
        n = self.stFrameInfo.nFrameLen
        pt = self.stFrameInfo.enPixelType

        if pt == PixelType_Gvsp_Mono8:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n).reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        if pt == PixelType_Gvsp_BGR8_Packed:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n)
            return raw.reshape(h, w, 3).copy()
        if pt == PixelType_Gvsp_RGB8_Packed:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n)
            return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        # Bayer... -> nhờ SDK demosaic sang BGR8 để màu chuẩn
        nDstSize = w * h * 3
        if self._convert_buf is None or len(self._convert_buf) != nDstSize:
            self._convert_buf = (c_ubyte * nDstSize)()
        cvt = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(cvt), 0, sizeof(cvt))
        cvt.nWidth = w
        cvt.nHeight = h
        cvt.pSrcData = self.data_buf
        cvt.nSrcDataLen = n
        cvt.enSrcPixelType = pt
        cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        cvt.pDstBuffer = self._convert_buf
        cvt.nDstBufferSize = nDstSize
        if self.cam.MV_CC_ConvertPixelType(cvt) != 0:
            print("ConvertPixelType lỗi, pixel type:", pt)
            return None
        img = np.frombuffer(self._convert_buf, dtype=np.uint8, count=nDstSize)
        return img.reshape(h, w, 3).copy()

    def close(self):
        if self.cam is not None:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()


def show_fit(winname, img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imshow(winname, img)


def get_small_frame(camera):
    """Đọc frame full-res + bản thu nhỏ 1/RESIZE_DIV. Trả (full, small) hoặc (None,None)."""
    full = camera.read()
    if full is None:
        return None, None
    h, w = full.shape[:2]
    small = cv2.resize(full, (w // RESIZE_DIV, h // RESIZE_DIV))
    return full, small


# ============================================================
# 3. HỌC NỀN (khung hình trống)
# ============================================================
def learn_background(camera):
    print(f"Chờ camera ổn định ({WARMUP_FRAMES} frames)...")
    skipped = 0
    while skipped < WARMUP_FRAMES:
        _, small = get_small_frame(camera)
        if small is None:
            continue
        skipped += 1
        disp = small.copy()
        cv2.putText(disp, f"Warming up... {skipped}/{WARMUP_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        show_fit("LIVE", disp)
        cv2.waitKey(1)

    print("Học nền... giữ khung hình TRỐNG")
    frames = []
    while len(frames) < BG_FRAMES:
        _, small = get_small_frame(camera)
        if small is None:
            continue
        frames.append(cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32))
        disp = small.copy()
        cv2.putText(disp, f"Learning BG... {len(frames)}/{BG_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        show_fit("LIVE", disp)
        cv2.waitKey(1)
    print("Xong!\n")
    return np.mean(frames, axis=0)


class BackgroundModel:
    def __init__(self, bg_bgr_float):
        self.bg = bg_bgr_float.astype(np.float32)
        b, g, r = cv2.split(self.bg)
        self.bg_sum = b + g + r + 3.0
        self.bg_sum_3d = cv2.merge([self.bg_sum, self.bg_sum, self.bg_sum])
        self.bg_plus_1 = self.bg + 1.0
        self.shadow_dim_low = SHADOW_DIM[0] * self.bg_sum
        self.shadow_dim_high = SHADOW_DIM[1] * self.bg_sum


# ============================================================
# 4. TẠO MASK + BBOX (single-frame, không tracking)
# ============================================================
def mask_from_background(img_bgr, bgm):
    blur = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    img_f = blur.astype(np.float32)

    diff_c = cv2.absdiff(img_f, bgm.bg)
    b, g, r = cv2.split(diff_c)
    diff = cv2.max(cv2.max(b, g), r)

    ib, ig, ir = cv2.split(img_f)
    S_img = ib + ig + ir + 3.0
    S_img_3d = cv2.merge([S_img, S_img, S_img])

    term_img = (img_f + 1.0) * bgm.bg_sum_3d
    term_bg = bgm.bg_plus_1 * S_img_3d
    chroma_diff = cv2.absdiff(term_img, term_bg)
    cb, cg, cr = cv2.split(chroma_diff)
    max_chroma = cv2.max(cv2.max(cb, cg), cr)

    same_color = max_chroma < (CHROMA_TOL * bgm.bg_sum) * S_img
    dim_moderate = (S_img > bgm.shadow_dim_low) & (S_img < bgm.shadow_dim_high)
    diff[same_color & dim_moderate & (diff < SHADOW_MAX_DIFF)] = 0

    strong = (diff > THRESH_STRONG).astype(np.uint8)
    weak = (diff > THRESH_WEAK).astype(np.uint8)
    if cv2.countNonZero(strong) == 0:
        return np.zeros(img_bgr.shape[:2], np.uint8)

    n_lbl, lbl = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(lbl[strong > 0])
    strong_labels = strong_labels[strong_labels != 0]
    keep = np.zeros(n_lbl, dtype=bool)
    keep[strong_labels] = True
    return keep[lbl].astype(np.uint8) * 255


def fill_holes(mask):
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    ff = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    filled = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
    return filled[1:-1, 1:-1]


def refine_mask(mask):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN_KSIZE)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KSIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    if FILL_HOLES:
        mask = fill_holes(mask)
    return mask


def detect_bbox(small_bgr, bgm):
    """Trả bbox (x, y, w, h) trên ảnh NHỎ, hoặc None."""
    mask = refine_mask(mask_from_background(small_bgr, bgm))
    area_img = small_bgr.shape[0] * small_bgr.shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if MIN_AREA_RATIO * area_img <= cv2.contourArea(c) <= MAX_AREA_RATIO * area_img]
    if not valid:
        return None
    return cv2.boundingRect(max(valid, key=cv2.contourArea))


# ============================================================
# 5. LOAD MODEL + GRAD-CAM
# ============================================================
def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(960, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(576, num_classes))
    else:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone_name}")
    return model


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    backbone_name = checkpoint.get("backbone_name", "mobilenet_v2")
    model = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE).eval()
    print(f"Đã load model: {model_path}")
    print(f"Backbone: {backbone_name} | {len(class_names)} class: {class_names}\n")
    return model, class_names, backbone_name


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def compute(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output


def overlay_heatmap(img_bgr, cam, alpha=HEATMAP_ALPHA):
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)


# ============================================================
# 6. XỬ LÝ 1 LẦN CHỤP
# ============================================================
def process_capture(full_bgr, small_bgr, bgm, model, class_names, gradcam):
    """Detect trên ảnh nhỏ -> map bbox lên full-res -> crop -> classifier + Grad-CAM."""
    bbox_small = detect_bbox(small_bgr, bgm)

    Hf, Wf = full_bgr.shape[:2]
    if bbox_small is not None:
        # map bbox từ ảnh nhỏ lên full-res
        x, y, w, h = [v * RESIZE_DIV for v in bbox_small]
        pad = CROP_PADDING * RESIZE_DIV
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(Wf, x + w + pad); y2 = min(Hf, y + h + pad)
        crop_bgr = full_bgr[y1:y2, x1:x2].copy()
        bbox_full = (x1, y1, x2, y2)
    else:
        print("  CẢNH BÁO: không detect được vật -> dùng nguyên frame.")
        crop_bgr = full_bgr.copy()
        bbox_full = None

    # --- Classifier + Grad-CAM trên ảnh crop ---
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)

    if TARGET_CLASS is not None:
        class_idx = class_names.index(TARGET_CLASS)
    else:
        with torch.no_grad():
            class_idx = F.softmax(model(input_tensor), dim=1).argmax(1).item()

    cam, output = gradcam.compute(input_tensor, class_idx)
    probs = F.softmax(output, dim=1)[0]
    pred_class = class_names[class_idx]
    confidence = probs[class_idx].item()

    overlay = overlay_heatmap(crop_bgr, cam)
    label = f"{pred_class} ({confidence*100:.1f}%)"
    cv2.putText(overlay, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2)

    # Ảnh gốc (nhỏ) + khung bbox để biết vùng đã crop
    disp_orig = small_bgr.copy()
    if bbox_full is not None:
        bx1, by1, bx2, by2 = [v // RESIZE_DIV for v in bbox_full]
        cv2.rectangle(disp_orig, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    # Ghép: trái = ảnh gốc + bbox, phải = heatmap (cùng chiều cao)
    th = disp_orig.shape[0]
    overlay_r = cv2.resize(overlay, (int(overlay.shape[1] * th / overlay.shape[0]), th))
    combined = np.hstack([disp_orig, overlay_r])

    print(f"  Đoán '{pred_class}' ({confidence*100:.1f}%)"
          + ("" if bbox_full is not None else "  [không detect được]"))
    return combined, pred_class


def save_result(combined, pred_class):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{pred_class}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(path, combined)
    print(f"  Đã lưu: {path}")


# ============================================================
# 7. MAIN
# ============================================================
LIVE_WIN = "LIVE"
RESULT_WIN = "KET QUA - Trai: goc+bbox | Phai: Grad-CAM  ('s'=luu, phim khac=quay lai)"


def main():
    model, class_names, backbone_name = load_model(MODEL_PATH)
    gradcam = GradCAM(model, model.features[-1])

    camera = HikrobotCamera()
    if not camera.open() or not camera.start():
        print("Không mở được camera Hikrobot.")
        camera.close()
        sys.exit(1)

    bg = learn_background(camera)
    bgm = BackgroundModel(bg)

    print("Bấm 'c' để chụp + Grad-CAM | 'r' học lại nền | 'q'/ESC thoát.\n")

    try:
        while True:
            full, small = get_small_frame(camera)
            if small is None:
                continue

            disp = small.copy()
            cv2.putText(disp, "c=chup+GradCAM  r=hoc lai nen  q=thoat",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            show_fit(LIVE_WIN, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            if key == ord('r'):
                bg = learn_background(camera)
                bgm = BackgroundModel(bg)
                continue

            if key == ord('c'):
                print("Đã chụp - đang xử lý...")
                combined, pred_class = process_capture(
                    full, small, bgm, model, class_names, gradcam)
                disp_res = cv2.resize(combined, None,
                                      fx=RESULT_DISPLAY_SCALE, fy=RESULT_DISPLAY_SCALE)
                show_fit(RESULT_WIN, disp_res)
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('s'):
                    save_result(combined, pred_class)
                cv2.destroyWindow(RESULT_WIN)
                if k2 in (ord('q'), 27):
                    break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()