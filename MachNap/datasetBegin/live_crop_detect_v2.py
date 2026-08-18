"""
LIVE DETECT V2: Camera Hikrobot + Crop + OOD Gate + Grad-CAM + TOP-5
===================================================================
Tính năng nổi bật V2:
  - Tự động chạy cả GPU (CUDA) và CPU (không lo thiếu GPU).
  - Tự động tìm và nạp model Classifier (.pt) + Bộ lọc OOD (.joblib) thông minh.
  - Tích hợp trừ nền, tách bóng, crop bbox, tính điểm OOD và vẽ Grad-CAM + Top-5.
  - Hỗ trợ 2 chế độ: "camera" (Realtime) hoặc "images" (Thư mục ảnh).
  - Camera hỗ trợ 2 chế độ thu hình: "continuous" (liên tục) và "singleframe" (từng khung hình).

PHÍM ĐIỀU KHIỂN (CAMERA - CONTINUOUS):
  r         : học lại nền
  s         : lưu frame kết quả hiện tại
  q/ESC     : thoát

PHÍM ĐIỀU KHIỂN (CAMERA - SINGLEFRAME):
  SPACE / c : chụp và xử lý frame tiếp theo
  r         : học lại nền
  s         : lưu frame kết quả hiện tại
  q/ESC     : thoát

PHÍM ĐIỀU KHIỂN (CHẾ ĐỘ ẢNH):
  SPACE / d : ảnh sau
  a         : ảnh trước
  s         : lưu ảnh kết quả
  q/ESC     : thoát

DÒNG LỆNH:
  python live_crop_detect_v2.py                     -> chạy theo cấu hình RUN_MODE & CAMERA_ACQUISITION_MODE
  python live_crop_detect_v2.py camera              -> chạy chế độ camera
  python live_crop_detect_v2.py camera singleframe  -> chạy camera chế độ single frame
  python live_crop_detect_v2.py camera continuous   -> chạy camera chế độ continuous
  python live_crop_detect_v2.py images              -> chạy chế độ thư mục ảnh
  python live_crop_detect_v2.py images <thu_muc>    -> chạy trên thư mục ảnh chỉ định
  python live_crop_detect_v2.py <thu_muc>           -> tự động chạy chế độ ảnh
"""

import sys
import os
import time
import glob
import joblib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


# ============================================================
# KHỞI TẠO SDK HIKROBOT (MVS)
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)

mvimport_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MvImport"
)
sys.path.append(mvimport_path)

try:
    from MvCameraControl_class import *
    HIKROBOT_SDK_OK = True
except Exception as _e:
    HIKROBOT_SDK_OK = False
    _HIKROBOT_IMPORT_ERROR = _e


# ============================================================
# 1. CẤU HÌNH HỆ THỐNG
# ============================================================

# CHẾ ĐỘ CHẠY: "camera" hoặc "images"
RUN_MODE = "camera"

# CHẾ ĐỘ THU HÌNH CAMERA: "continuous" (liên tục) hoặc "singleframe" (từng khung hình)
CAMERA_ACQUISITION_MODE = "continuous"  # "continuous" | "singleframe"

# Thư mục chứa nhiều ảnh (dùng khi RUN_MODE = "images")
IMAGE_FOLDER = r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1\Part6"

# Thư mục chứa Model & OOD Detector (Có thể trỏ vào runs cụ thể hoặc thư mục model tổng)
OOD_RUNS_DIR = r"D:\TongHop\RTC Technologi\G8\modelClassifierOOD\runs\20260817_233900"
OOD_PAYLOAD_PATH = ""  # Để trống sẽ tự động tìm .joblib trong OOD_RUNS_DIR

SAVE_DIR = r"D:\TongHop\RTC Technologi\G8\gradcam_captures"
CAMERA_MODEL_HINT = "MV-CS200"

RESIZE_DIV = 4

# TỰ ĐỘNG NHẬN DIỆN GPU HOẶC CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_COSINE_GATE = False


# ============================================================
# BACKGROUND SUBTRACTION
# ============================================================
THRESH_STRONG = 35
THRESH_WEAK = 20
CHROMA_TOL = 0.04
SHADOW_DIM = (0.3, 0.95)
SHADOW_MAX_DIFF = 70
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.8
MORPH_OPEN_KSIZE = (5, 5)
MORPH_CLOSE_KSIZE = (7, 7)
FILL_HOLES = True
CROP_PADDING = 15


# ============================================================
# HỌC NỀN
# ============================================================
WARMUP_FRAMES = 30
BG_FRAMES = 40


# ============================================================
# HIỂN THỊ
# ============================================================
HEATMAP_ALPHA = 0.45
RESULT_DISPLAY_SCALE = 0.5
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

# Xử lý mỗi N frame (classifier + OOD + Grad-CAM)
PROCESS_EVERY_N_FRAMES = 1

# Bật Grad-CAM cho realtime (Tắt đi nếu muốn tối đa FPS trên máy yếu)
REALTIME_GRADCAM = True


# ============================================================
# TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp"
)


def imread_unicode(path):
    """Đọc ảnh, hỗ trợ đường dẫn Unicode/tiếng Việt trên Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """Ghi ảnh, hỗ trợ đường dẫn Unicode."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def list_images(folder):
    """Liệt kê toàn bộ file ảnh trong thư mục (đã sort theo tên)."""
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(folder, name))
    return files


# ============================================================
# 2. CAMERA HIKROBOT
# ============================================================
class HikrobotCamera:
    def __init__(self, model_hint=CAMERA_MODEL_HINT, acquisition_mode=CAMERA_ACQUISITION_MODE):
        self.model_hint = model_hint
        self.acquisition_mode = acquisition_mode.lower() if acquisition_mode else "continuous"
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
            print(f"Không tìm thấy camera nào! Mã lỗi: {ret}")
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

        stDeviceInfo = cast(deviceList.pDeviceInfo[target_index], POINTER(MV_CC_DEVICE_INFO)).contents
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

        # Cài đặt AcquisitionMode: Continuous hoặc SingleFrame
        if self.acquisition_mode in ("singleframe", "single_frame", "single"):
            ret_mode = self.cam.MV_CC_SetEnumValueByString("AcquisitionMode", "SingleFrame")
            if ret_mode != 0:
                ret_mode = self.cam.MV_CC_SetEnumValue("AcquisitionMode", 1)
            print(f"📷 Chế độ AcquisitionMode: SingleFrame (Chụp từng frame) [mã: {ret_mode}]")
        else:
            ret_mode = self.cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
            if ret_mode != 0:
                ret_mode = self.cam.MV_CC_SetEnumValue("AcquisitionMode", 0)
            print(f"📷 Chế độ AcquisitionMode: Continuous (Liên tục) [mã: {ret_mode}]")

        return True

    def start(self):
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self.nPayloadSize = stParam.nCurValue
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        self.data_buf = (c_ubyte * self.nPayloadSize)()

        # Nếu ở chế độ Continuous, bắt đầu grabbing liên tục
        if self.acquisition_mode not in ("singleframe", "single_frame", "single"):
            if self.cam.MV_CC_StartGrabbing() != 0:
                print("Start grabbing lỗi.")
                return False
        return True

    def read(self):
        is_single = self.acquisition_mode in ("singleframe", "single_frame", "single")
        if is_single:
            if self.cam.MV_CC_StartGrabbing() != 0:
                return None

        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf,
            self.nPayloadSize,
            self.stFrameInfo,
            1000
        )

        if is_single:
            self.cam.MV_CC_StopGrabbing()
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
            print(f"ConvertPixelType lỗi, pixel type: {pt}")
            return None

        img = np.frombuffer(self._convert_buf, dtype=np.uint8, count=nDstSize)
        return img.reshape(h, w, 3).copy()

    def close(self):
        if self.cam is not None:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()


# ============================================================
# HIỂN THỊ
# ============================================================
def show_fit(winname, img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imshow(winname, img)


def get_small_frame(camera):
    full = camera.read()
    if full is None:
        return None, None
    h, w = full.shape[:2]
    small = cv2.resize(full, (w // RESIZE_DIV, h // RESIZE_DIV))
    return full, small


# ============================================================
# 3. HỌC NỀN
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
        cv2.putText(disp, f"Warming up... {skipped}/{WARMUP_FRAMES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
        cv2.putText(disp, f"Learning BG... {len(frames)}/{BG_FRAMES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        show_fit("LIVE", disp)
        cv2.waitKey(1)

    print("Học nền xong!\n")
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
# 4. MASK + BBOX
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
    return (keep[lbl].astype(np.uint8) * 255)


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
    mask = refine_mask(mask_from_background(small_bgr, bgm))
    area_img = small_bgr.shape[0] * small_bgr.shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [
        c for c in contours
        if (MIN_AREA_RATIO * area_img) <= cv2.contourArea(c) <= (MAX_AREA_RATIO * area_img)
    ]
    if not valid:
        return None
    return cv2.boundingRect(max(valid, key=cv2.contourArea))


# ============================================================
# 5. LOAD MODEL + OOD DETECTOR
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
        raise ValueError(f"Backbone không hỗ trợ: {backbone_name}")
    return model


def resolve_payload_path():
    """Tự động tìm kiếm file ood_detector.joblib"""
    if OOD_PAYLOAD_PATH and os.path.exists(OOD_PAYLOAD_PATH):
        return OOD_PAYLOAD_PATH

    # 1. Tìm trực tiếp file ood_detector.joblib trong OOD_RUNS_DIR
    direct_joblib = os.path.join(OOD_RUNS_DIR, "ood_detector.joblib")
    if os.path.isfile(direct_joblib):
        return direct_joblib

    # 2. Tìm theo con trỏ ood_detector_latest.txt hoặc LATEST_RUN.txt
    for ptr_name in ["ood_detector_latest.txt", "LATEST_RUN.txt"]:
        pointer = os.path.join(OOD_RUNS_DIR, ptr_name)
        if os.path.exists(pointer):
            with open(pointer, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
                for line in lines:
                    line = line.strip()
                    if line.endswith(".joblib") and os.path.exists(line):
                        return line
                    if "Thư mục gốc của run:" in line:
                        run_path = line.split(":")[-1].strip()
                        jb = os.path.join(run_path, "ood_detector.joblib")
                        if os.path.exists(jb):
                            return jb

    # 3. Tìm đệ quy bất kỳ file .joblib nào mới nhất
    joblibs = glob.glob(os.path.join(OOD_RUNS_DIR, "**", "*.joblib"), recursive=True)
    if joblibs:
        joblibs.sort(key=os.path.getmtime, reverse=True)
        return joblibs[0]

    raise FileNotFoundError(
        f"Không tìm thấy file OOD .joblib trong: {OOD_RUNS_DIR}\n"
        "Hãy kiểm tra lại đường dẫn OOD_RUNS_DIR hoặc gán OOD_PAYLOAD_PATH."
    )


def load_everything():
    payload_path = resolve_payload_path()
    payload = joblib.load(payload_path)

    print("=" * 65)
    print(f"📦 Đã load OOD Payload  : {payload_path}")
    print(f"   -> Method            : {payload['ood_method']}")
    print(f"   -> Ngưỡng Threshold  : {payload['threshold']:.4f}")
    if "target_tpr" in payload:
        print(f"   -> Target TPR        : {payload['target_tpr']}")

    # Tìm file checkpoint .pt
    ckpt_path = payload.get("classifier_ckpt", "")
    if not os.path.exists(ckpt_path):
        # Fallback tìm .pt ngay cùng thư mục với file .joblib
        parent_dir = os.path.dirname(payload_path)
        pt_files = glob.glob(os.path.join(parent_dir, "*.pt"))
        if pt_files:
            ckpt_path = pt_files[0]
        else:
            raise FileNotFoundError(f"Không tìm thấy file model .pt cho payload tại: {payload_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    backbone_name = ckpt.get("backbone_name", payload.get("backbone_name", "mobilenet_v3_large"))

    model = build_model_architecture(backbone_name, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()

    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"🤖 Đã load PyTorch Model: {ckpt_path}")
    print(f"   -> Backbone          : {backbone_name} ({len(class_names)} classes)")
    print(f"   -> Thiết bị chạy     : {DEVICE.type.upper()} ({device_name})")
    print("=" * 65 + "\n")

    return model, class_names, payload


# ============================================================
# GRAD-CAM
# ============================================================
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
# EMBEDDING + OOD SCORE
# ============================================================
@torch.no_grad()
def extract_embedding(model, input_tensor):
    feat = model.features(input_tensor)
    pooled = F.adaptive_avg_pool2d(feat, (1, 1))
    emb = torch.flatten(pooled, 1)
    return emb.cpu().numpy().astype(np.float32)


def l2norm(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def score_cosine(X, centroids):
    return (l2norm(X) @ centroids.T).max(1)


def score_mahalanobis(X, means, precision):
    N, C = X.shape[0], means.shape[0]
    dists = np.empty((N, C), np.float64)
    for c in range(C):
        d = X - means[c]
        dists[:, c] = np.einsum("nd,de,ne->n", d, precision, d)
    return -dists.min(1)


def ood_score(emb, payload):
    method = payload["ood_method"]
    best_cos = float(score_cosine(emb, payload["centroids"])[0])
    if method == "mahalanobis":
        m = payload["mahalanobis"]
        s = float(score_mahalanobis(emb, m["means"], m["precision"])[0])
    else:
        s = best_cos
    return s, best_cos


# ============================================================
# TOP-5 PANEL
# ============================================================
def get_top5_classes(probs, class_names):
    k = min(5, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)
    return [
        (class_names[int(idx)], float(prob))
        for prob, idx in zip(top_probs, top_indices)
    ]


def draw_top5_panel(img, top5):
    panel = img.copy()
    x0, y0 = 25, 100
    title_scale, text_scale = 1.60, 1.45
    title_thickness, text_thickness = 4, 4
    line_h = 78
    panel_w = 900
    top_padding = 50
    title_to_first_line = 70
    bottom_padding = 30
    panel_h = top_padding + title_to_first_line + line_h * len(top5) + bottom_padding

    h, w = panel.shape[:2]
    x1 = min(w, x0 + panel_w)
    y1 = min(h, y0 + panel_h)

    roi = panel[y0:y1, x0:x1].copy()
    if roi.size > 0:
        cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (40, 40, 40), -1)
        panel[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.75, panel[y0:y1, x0:x1], 0.25, 0)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (180, 180, 180), 2)

    cv2.putText(panel, "TOP-5 DUDOAN:", (x0 + 25, y0 + top_padding),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 255), title_thickness, cv2.LINE_AA)

    first_line_y = y0 + top_padding + title_to_first_line
    for i, (name, prob) in enumerate(top5):
        yy = first_line_y + i * line_h
        text_line = f"{i + 1}. {name}: {prob * 100:.2f}%"
        cv2.putText(panel, text_line, (x0 + 25, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return panel


# ============================================================
# 6. XỬ LÝ 1 FRAME / 1 ẢNH
# ============================================================
def process_capture(
    full_bgr,
    small_bgr,
    bgm,
    model,
    class_names,
    gradcam,
    payload,
    verbose=True,
    do_gradcam=True
):
    bbox_small = detect_bbox(small_bgr, bgm) if bgm is not None else None
    Hf, Wf = full_bgr.shape[:2]

    # CROP
    if bbox_small is not None:
        x, y, w, h = [v * RESIZE_DIV for v in bbox_small]
        pad = CROP_PADDING * RESIZE_DIV
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(Wf, x + w + pad)
        y2 = min(Hf, y + h + pad)
        crop_bgr = full_bgr[y1:y2, x1:x2].copy()
        bbox_full = (x1, y1, x2, y2)
    else:
        if bgm is not None and verbose:
            print("  CẢNH BÁO: không detect được vật -> dùng nguyên frame.")
        crop_bgr = full_bgr.copy()
        bbox_full = None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)

    # 1. CLASSIFIER
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        confidence, pred_idx = probs.max(0)
        confidence = float(confidence)
        class_idx = int(pred_idx)
        pred_class = class_names[class_idx]
        top5 = get_top5_classes(probs, class_names)

    if verbose:
        print(f"Top-1: {pred_class} ({confidence * 100:.2f}%)")
        print("Top-5:")
        for rank, (name, prob) in enumerate(top5, 1):
            print(f"    {rank}. {name}: {prob * 100:.2f}%")

    # 2. OOD
    emb = extract_embedding(model, input_tensor)
    score, best_cos = ood_score(emb, payload)
    threshold = float(payload["threshold"])
    is_ood = score < threshold

    if USE_COSINE_GATE and best_cos < float(payload.get("cosine_threshold", -1)):
        is_ood = True

    # 3. STATUS
    if is_ood:
        status = "UNKNOWN_OOD"
        label = f"UNKNOWN (vat la) sim={best_cos:.2f}"
        color = (0, 0, 255)
        if verbose:
            print(f"  [X] TU CHOI (OOD) | score={score:.2f} < tau={threshold:.2f} | best_cos={best_cos:.2f}")
    else:
        status = "OK"
        label = f"{pred_class} ({confidence * 100:.1f}%)"
        color = (0, 255, 0)
        if verbose:
            print(f"  [OK] {pred_class} ({confidence * 100:.1f}%) | score={score:.2f} >= tau={threshold:.2f}")

    # 4. GRAD-CAM
    if (not is_ood) and do_gradcam:
        cam, _ = gradcam.compute(input_tensor, class_idx)
        overlay = overlay_heatmap(crop_bgr, cam)
    else:
        overlay = crop_bgr.copy()

    cv2.putText(overlay, label, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
    overlay = draw_top5_panel(overlay, top5)

    disp_orig = small_bgr.copy()
    if bbox_full is not None:
        bx1, by1, bx2, by2 = [v // RESIZE_DIV for v in bbox_full]
        cv2.rectangle(disp_orig, (bx1, by1), (bx2, by2), color, 2)

    th = disp_orig.shape[0]
    overlay_r = cv2.resize(overlay, (int(overlay.shape[1] * th / overlay.shape[0]), th))
    combined = np.hstack([disp_orig, overlay_r])
    tag = pred_class if not is_ood else "UNKNOWN"

    return combined, tag, status


def save_result(combined, tag, status):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{status}_{tag}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    imwrite_unicode(path, combined)
    print(f"  Đã lưu: {path}")


# ============================================================
# 7. MAIN RUN LOOPS
# ============================================================
LIVE_WIN = "LIVE"
RESULT_WIN = "REALTIME - Trai: goc+bbox | Phai: Grad-CAM + TOP-5 ('s'=luu | 'r'=hoc lai nen | 'q'=thoat)"


def run_camera(model, class_names, gradcam, payload, acquisition_mode=CAMERA_ACQUISITION_MODE):
    if not HIKROBOT_SDK_OK:
        print("Không import được SDK Hikrobot (MvImport). Không thể chạy chế độ camera.\n"
              f"Chi tiết: {_HIKROBOT_IMPORT_ERROR}")
        sys.exit(1)

    is_single = acquisition_mode.lower() in ("singleframe", "single_frame", "single")
    camera = HikrobotCamera(acquisition_mode=acquisition_mode)
    if not camera.open() or not camera.start():
        print("Không mở được camera Hikrobot.")
        camera.close()
        sys.exit(1)

    bg = learn_background(camera)
    bgm = BackgroundModel(bg)

    if is_single:
        print("📷 CAMERA SINGLEFRAME: Bấm SPACE / 'c' = chụp frame mới | 'r' = học lại nền | 's' = lưu | 'q'/ESC = thoát.\n")
    else:
        print("📷 CAMERA CONTINUOUS: Đang chạy Realtime | 'r' = học lại nền | 's' = lưu frame | 'q'/ESC = thoát.\n")

    frame_id = 0
    last_combined = None
    last_tag = "UNKNOWN"
    last_status = "OK"
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            full, small = get_small_frame(camera)
            if small is None:
                continue

            frame_id += 1
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                combined, tag, status = process_capture(
                    full, small, bgm, model, class_names, gradcam, payload,
                    verbose=is_single, do_gradcam=REALTIME_GRADCAM
                )
                last_combined = combined
                last_tag = tag
                last_status = status

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if last_combined is not None:
                disp_res = cv2.resize(last_combined, None, fx=RESULT_DISPLAY_SCALE, fy=RESULT_DISPLAY_SCALE)
                if not is_single:
                    cv2.putText(disp_res, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(disp_res, "SINGLEFRAME [SPACE/c: next]", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                show_fit(RESULT_WIN, disp_res)
            else:
                show_fit(LIVE_WIN, small)

            if is_single:
                # Chờ lệnh bấm phím từ người dùng cho từng khung hình
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("q"), 27):
                        return
                    elif key == ord("r"):
                        bg = learn_background(camera)
                        bgm = BackgroundModel(bg)
                        break
                    elif key == ord("s") and last_combined is not None:
                        save_result(last_combined, last_tag, last_status)
                        continue
                    elif key in (ord(" "), ord("c"), 13, 10):  # SPACE, 'c', Enter
                        break
            else:
                # Chế độ Continuous liên tục
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    bg = learn_background(camera)
                    bgm = BackgroundModel(bg)
                elif key == ord("s") and last_combined is not None:
                    save_result(last_combined, last_tag, last_status)
    finally:
        camera.close()
        cv2.destroyAllWindows()


def run_image_folder(folder, model, class_names, gradcam, payload, bgm=None):
    if not os.path.isdir(folder):
        print("Thư mục ảnh không tồn tại:", folder)
        return

    files = list_images(folder)
    if not files:
        print("Không tìm thấy ảnh trong:", folder)
        return

    print(f"Tìm thấy {len(files)} ảnh trong: {folder}")
    print("Phím: SPACE/d = ảnh sau | a = ảnh trước | s = lưu | q/ESC = thoát\n")

    idx = 0
    while 0 <= idx < len(files):
        path = files[idx]
        img = imread_unicode(path)
        if img is None:
            print(f"[{idx + 1}/{len(files)}] LỖI đọc ảnh: {path}")
            idx += 1
            continue

        print(f"[{idx + 1}/{len(files)}] {os.path.basename(path)}")
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // RESIZE_DIV), max(1, h // RESIZE_DIV)))

        combined, tag, status = process_capture(
            img, small, bgm, model, class_names, gradcam, payload
        )

        disp_res = cv2.resize(combined, None, fx=RESULT_DISPLAY_SCALE, fy=RESULT_DISPLAY_SCALE)
        show_fit(RESULT_WIN, disp_res)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                save_result(combined, tag, status)
                continue
            break

        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyWindow(RESULT_WIN)
    cv2.destroyAllWindows()


def main():
    mode = RUN_MODE
    acq_mode = CAMERA_ACQUISITION_MODE
    folder = IMAGE_FOLDER

    args = sys.argv[1:]
    if args:
        a0 = args[0].lower()
        if a0 in ("camera", "cam"):
            mode = "camera"
            if len(args) >= 2:
                a1 = args[1].lower()
                if a1 in ("singleframe", "single", "single_frame", "1"):
                    acq_mode = "singleframe"
                elif a1 in ("continuous", "cont", "0"):
                    acq_mode = "continuous"
        elif a0 in ("singleframe", "single", "single_frame"):
            mode = "camera"
            acq_mode = "singleframe"
        elif a0 in ("continuous", "cont"):
            mode = "camera"
            acq_mode = "continuous"
        elif a0 in ("images", "image", "img", "folder"):
            mode = "images"
            if len(args) >= 2:
                folder = args[1]
        elif os.path.isdir(args[0]):
            mode = "images"
            folder = args[0]

    # Nạp toàn bộ Model + OOD
    model, class_names, payload = load_everything()

    # Khởi tạo Grad-CAM
    gradcam = GradCAM(model, model.features[-1])

    # Thực thi chế độ
    if mode == "images":
        print(f"=== CHẾ ĐỘ ẢNH: {folder} ===\n")
        run_image_folder(folder, model, class_names, gradcam, payload)
    else:
        print(f"=== CHẾ ĐỘ CAMERA ({acq_mode.upper()}) ===\n")
        run_camera(model, class_names, gradcam, payload, acquisition_mode=acq_mode)


if __name__ == "__main__":
    main()
