"""
LIVE DETECT: Camera Hikrobot + Crop + OOD gate + Grad-CAM + TOP-5
=================================================================
Kết hợp:
  - Camera Hikrobot (MVS) + học nền + background subtraction + crop bbox.
  - GATE OOD đọc từ payload train (ood_detector_*.joblib):
    Mahalanobis/cosine + threshold đã CALIBRATE.
  - Grad-CAM trên classifier -> chỉ vẽ khi ACCEPTED.
  - Hiển thị 5 class có confidence cao nhất.

CHẾ ĐỘ CAMERA GIỜ CHẠY REALTIME:
  - Không cần bấm 'c' nữa. Mọi frame được xử lý liên tục.

HAI CHẾ ĐỘ CHẠY:
  - RUN_MODE = "camera"  : chạy camera Hikrobot REALTIME.
  - RUN_MODE = "images"  : chạy trên 1 thư mục chứa nhiều ảnh
                           (bỏ trừ nền, đưa nguyên ảnh vào classifier).

PHÍM (CAMERA - REALTIME):
  r     : học lại nền
  s     : lưu frame kết quả hiện tại
  q/ESC : thoát

PHÍM (ẢNH):
  SPACE / d : ảnh sau
  a         : ảnh trước
  s         : lưu ảnh kết quả
  q/ESC     : thoát

DÒNG LỆNH:
  python live_detect.py                       -> theo RUN_MODE
  python live_detect.py camera
  python live_detect.py images
  python live_detect.py images <thu_muc_anh>
  python live_detect.py <thu_muc_anh>         -> tự nhận là chế độ ảnh
"""

import sys
import os
import time
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

# Import SDK camera. Ở chế độ ảnh không bắt buộc có SDK,
# nên bọc trong try để vẫn chạy được khi máy không cài MVS.
try:
    from MvCameraControl_class import *
    HIKROBOT_SDK_OK = True
except Exception as _e:
    HIKROBOT_SDK_OK = False
    _HIKROBOT_IMPORT_ERROR = _e


# ============================================================
# 1. CẤU HÌNH
# ============================================================

# CHẾ ĐỘ CHẠY: "camera" hoặc "images"
RUN_MODE = "camera"

# Thư mục chứa nhiều ảnh (dùng khi RUN_MODE = "images")
IMAGE_FOLDER = (
    r"D:\TongHop\RTC Technologi\G8\dataset\Parts3\Parts1\Part6"
)

OOD_RUNS_DIR = (
    r"D:\TongHop\RTC Technologi\G8\modelClassifierOOD\runs\20260817_233900"
)

OOD_PAYLOAD_PATH = ""

SAVE_DIR = (
    r"D:\TongHop\RTC Technologi\G8\gradcam_captures"
)

CAMERA_MODEL_HINT = "MV-CS200"

RESIZE_DIV = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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
MAX_AREA_RATIO = 0.95

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


# ============================================================
# REALTIME
# ============================================================

# Xử lý mỗi N frame (classifier + OOD + Grad-CAM).
# Tăng lên (2, 3, 4...) nếu bị lag/giật.
PROCESS_EVERY_N_FRAMES = 1

# Bật Grad-CAM cho realtime. Grad-CAM có bước backward() nên khá nặng.
# Đặt False sẽ nhanh hơn nhiều mà vẫn giữ nhãn + Top-5 + OOD.
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


# ============================================================
# TIỆN ÍCH ĐỌC/GHI/LIỆT KÊ ẢNH (HỖ TRỢ PATH UNICODE)
# ============================================================

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

    def __init__(
        self,
        model_hint=CAMERA_MODEL_HINT
    ):

        self.model_hint = model_hint

        self.cam = None

        self.data_buf = None

        self.nPayloadSize = 0

        self.stFrameInfo = None

        self._convert_buf = None


    def open(self):

        print(
            "SDK Version:",
            hex(
                MvCamera.MV_CC_GetSDKVersion()
            )
        )

        deviceList = (
            MV_CC_DEVICE_INFO_LIST()
        )

        tlayerType = (
            MV_GIGE_DEVICE |
            MV_USB_DEVICE
        )

        ret = MvCamera.MV_CC_EnumDevices(
            tlayerType,
            deviceList
        )

        if (
            ret != 0
            or deviceList.nDeviceNum == 0
        ):

            print(
                "Không tìm thấy camera nào! "
                f"Mã lỗi: {ret}"
            )

            return False

        print(
            f"Tìm thấy "
            f"{deviceList.nDeviceNum} camera"
        )

        target_index = 0

        for i in range(
            deviceList.nDeviceNum
        ):

            info = cast(
                deviceList.pDeviceInfo[i],
                POINTER(MV_CC_DEVICE_INFO)
            ).contents

            if info.nTLayerType == MV_GIGE_DEVICE:

                name = "".join(
                    chr(c)
                    for c in
                    info.SpecialInfo
                    .stGigEInfo
                    .chModelName
                    if c != 0
                )

                print(
                    f"[{i}] GigE Camera: "
                    f"{name}"
                )

            elif info.nTLayerType == MV_USB_DEVICE:

                name = "".join(
                    chr(c)
                    for c in
                    info.SpecialInfo
                    .stUsb3VInfo
                    .chModelName
                    if c != 0
                )

                print(
                    f"[{i}] USB Camera: "
                    f"{name}"
                )

            else:

                name = ""

            if (
                self.model_hint
                and self.model_hint in name
            ):
                target_index = i

        stDeviceInfo = cast(
            deviceList.pDeviceInfo[
                target_index
            ],
            POINTER(MV_CC_DEVICE_INFO)
        ).contents

        self.cam = MvCamera()

        if (
            self.cam.MV_CC_CreateHandle(
                stDeviceInfo
            ) != 0
        ):

            print(
                "Tạo handle lỗi."
            )

            return False

        if (
            self.cam.MV_CC_OpenDevice(
                MV_ACCESS_Exclusive,
                0
            ) != 0
        ):

            print(
                "Mở camera lỗi."
            )

            self.cam.MV_CC_DestroyHandle()

            return False

        if (
            stDeviceInfo.nTLayerType
            == MV_GIGE_DEVICE
        ):

            nPacketSize = (
                self.cam
                .MV_CC_GetOptimalPacketSize()
            )

            if nPacketSize > 0:

                self.cam.MV_CC_SetIntValue(
                    "GevSCPSPacketSize",
                    nPacketSize
                )

        self.cam.MV_CC_SetEnumValue(
            "TriggerMode",
            MV_TRIGGER_MODE_OFF
        )

        return True


    def start(self):

        if (
            self.cam.MV_CC_StartGrabbing()
            != 0
        ):

            print(
                "Start grabbing lỗi."
            )

            return False

        stParam = MVCC_INTVALUE()

        memset(
            byref(stParam),
            0,
            sizeof(MVCC_INTVALUE)
        )

        self.cam.MV_CC_GetIntValue(
            "PayloadSize",
            stParam
        )

        self.nPayloadSize = (
            stParam.nCurValue
        )

        self.stFrameInfo = (
            MV_FRAME_OUT_INFO_EX()
        )

        memset(
            byref(self.stFrameInfo),
            0,
            sizeof(self.stFrameInfo)
        )

        self.data_buf = (
            c_ubyte *
            self.nPayloadSize
        )()

        return True


    def read(self):

        ret = (
            self.cam
            .MV_CC_GetOneFrameTimeout(
                self.data_buf,
                self.nPayloadSize,
                self.stFrameInfo,
                1000
            )
        )

        if ret != 0:
            return None

        w = self.stFrameInfo.nWidth
        h = self.stFrameInfo.nHeight
        n = self.stFrameInfo.nFrameLen
        pt = self.stFrameInfo.enPixelType


        # ----------------------------------------------------
        # Mono8
        # ----------------------------------------------------

        if pt == PixelType_Gvsp_Mono8:

            raw = np.frombuffer(
                self.data_buf,
                dtype=np.uint8,
                count=n
            ).reshape(
                h,
                w
            )

            return cv2.cvtColor(
                raw,
                cv2.COLOR_GRAY2BGR
            )


        # ----------------------------------------------------
        # BGR8
        # ----------------------------------------------------

        if (
            pt
            == PixelType_Gvsp_BGR8_Packed
        ):

            raw = np.frombuffer(
                self.data_buf,
                dtype=np.uint8,
                count=n
            )

            return raw.reshape(
                h,
                w,
                3
            ).copy()


        # ----------------------------------------------------
        # RGB8
        # ----------------------------------------------------

        if (
            pt
            == PixelType_Gvsp_RGB8_Packed
        ):

            raw = np.frombuffer(
                self.data_buf,
                dtype=np.uint8,
                count=n
            )

            return cv2.cvtColor(
                raw.reshape(
                    h,
                    w,
                    3
                ),
                cv2.COLOR_RGB2BGR
            )


        # ----------------------------------------------------
        # Convert pixel format
        # ----------------------------------------------------

        nDstSize = w * h * 3

        if (
            self._convert_buf is None
            or
            len(self._convert_buf)
            != nDstSize
        ):

            self._convert_buf = (
                c_ubyte *
                nDstSize
            )()

        cvt = MV_CC_PIXEL_CONVERT_PARAM()

        memset(
            byref(cvt),
            0,
            sizeof(cvt)
        )

        cvt.nWidth = w
        cvt.nHeight = h

        cvt.pSrcData = self.data_buf
        cvt.nSrcDataLen = n

        cvt.enSrcPixelType = pt
        cvt.enDstPixelType = (
            PixelType_Gvsp_BGR8_Packed
        )

        cvt.pDstBuffer = (
            self._convert_buf
        )

        cvt.nDstBufferSize = (
            nDstSize
        )

        if (
            self.cam
            .MV_CC_ConvertPixelType(
                cvt
            ) != 0
        ):

            print(
                "ConvertPixelType lỗi, "
                f"pixel type: {pt}"
            )

            return None

        img = np.frombuffer(
            self._convert_buf,
            dtype=np.uint8,
            count=nDstSize
        )

        return img.reshape(
            h,
            w,
            3
        ).copy()


    def close(self):

        if self.cam is not None:

            self.cam.MV_CC_StopGrabbing()

            self.cam.MV_CC_CloseDevice()

            self.cam.MV_CC_DestroyHandle()


# ============================================================
# HIỂN THỊ
# ============================================================

def show_fit(
    winname,
    img,
    max_w=DISPLAY_MAX_W,
    max_h=DISPLAY_MAX_H
):

    h, w = img.shape[:2]

    s = min(
        1.0,
        max_w / w,
        max_h / h
    )

    if s < 1.0:

        img = cv2.resize(
            img,
            (
                int(w * s),
                int(h * s)
            )
        )

    cv2.imshow(
        winname,
        img
    )


def get_small_frame(camera):

    full = camera.read()

    if full is None:

        return None, None

    h, w = full.shape[:2]

    small = cv2.resize(
        full,
        (
            w // RESIZE_DIV,
            h // RESIZE_DIV
        )
    )

    return full, small


# ============================================================
# 3. HỌC NỀN
# ============================================================

def learn_background(camera):

    print(
        f"Chờ camera ổn định "
        f"({WARMUP_FRAMES} frames)..."
    )

    skipped = 0

    while (
        skipped
        < WARMUP_FRAMES
    ):

        _, small = (
            get_small_frame(camera)
        )

        if small is None:
            continue

        skipped += 1

        disp = small.copy()

        cv2.putText(
            disp,
            (
                f"Warming up... "
                f"{skipped}/"
                f"{WARMUP_FRAMES}"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

        show_fit(
            "LIVE",
            disp
        )

        cv2.waitKey(1)


    print(
        "Học nền... giữ khung hình TRỐNG"
    )

    frames = []

    while (
        len(frames)
        < BG_FRAMES
    ):

        _, small = (
            get_small_frame(camera)
        )

        if small is None:
            continue

        frames.append(
            cv2.GaussianBlur(
                small,
                (7, 7),
                0
            ).astype(
                np.float32
            )
        )

        disp = small.copy()

        cv2.putText(
            disp,
            (
                f"Learning BG... "
                f"{len(frames)}/"
                f"{BG_FRAMES}"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        show_fit(
            "LIVE",
            disp
        )

        cv2.waitKey(1)


    print("Xong!\n")

    return np.mean(
        frames,
        axis=0
    )


class BackgroundModel:

    def __init__(
        self,
        bg_bgr_float
    ):

        self.bg = (
            bg_bgr_float
            .astype(np.float32)
        )

        b, g, r = cv2.split(
            self.bg
        )

        self.bg_sum = (
            b + g + r + 3.0
        )

        self.bg_sum_3d = cv2.merge([
            self.bg_sum,
            self.bg_sum,
            self.bg_sum
        ])

        self.bg_plus_1 = (
            self.bg + 1.0
        )

        self.shadow_dim_low = (
            SHADOW_DIM[0]
            * self.bg_sum
        )

        self.shadow_dim_high = (
            SHADOW_DIM[1]
            * self.bg_sum
        )


# ============================================================
# 4. MASK + BBOX
# ============================================================

def mask_from_background(
    img_bgr,
    bgm
):

    blur = cv2.GaussianBlur(
        img_bgr,
        (7, 7),
        0
    )

    img_f = blur.astype(
        np.float32
    )

    diff_c = cv2.absdiff(
        img_f,
        bgm.bg
    )

    b, g, r = cv2.split(
        diff_c
    )

    diff = cv2.max(
        cv2.max(b, g),
        r
    )


    ib, ig, ir = cv2.split(
        img_f
    )

    S_img = (
        ib + ig + ir + 3.0
    )

    S_img_3d = cv2.merge([
        S_img,
        S_img,
        S_img
    ])


    term_img = (
        (img_f + 1.0)
        * bgm.bg_sum_3d
    )

    term_bg = (
        bgm.bg_plus_1
        * S_img_3d
    )

    chroma_diff = cv2.absdiff(
        term_img,
        term_bg
    )

    cb, cg, cr = cv2.split(
        chroma_diff
    )

    max_chroma = cv2.max(
        cv2.max(cb, cg),
        cr
    )


    same_color = (
        max_chroma
        <
        (
            CHROMA_TOL
            * bgm.bg_sum
        )
        * S_img
    )

    dim_moderate = (
        (S_img > bgm.shadow_dim_low)
        &
        (S_img < bgm.shadow_dim_high)
    )

    diff[
        same_color
        &
        dim_moderate
        &
        (diff < SHADOW_MAX_DIFF)
    ] = 0


    strong = (
        diff > THRESH_STRONG
    ).astype(
        np.uint8
    )

    weak = (
        diff > THRESH_WEAK
    ).astype(
        np.uint8
    )


    if (
        cv2.countNonZero(strong)
        == 0
    ):

        return np.zeros(
            img_bgr.shape[:2],
            np.uint8
        )


    n_lbl, lbl = (
        cv2.connectedComponents(
            weak,
            connectivity=8
        )
    )

    strong_labels = np.unique(
        lbl[
            strong > 0
        ]
    )

    strong_labels = (
        strong_labels[
            strong_labels != 0
        ]
    )

    keep = np.zeros(
        n_lbl,
        dtype=bool
    )

    keep[
        strong_labels
    ] = True

    return (
        keep[lbl]
        .astype(np.uint8)
        * 255
    )


def fill_holes(mask):

    h, w = mask.shape

    padded = np.zeros(
        (
            h + 2,
            w + 2
        ),
        np.uint8
    )

    padded[
        1:-1,
        1:-1
    ] = mask

    ff = padded.copy()

    ff_mask = np.zeros(
        (
            h + 4,
            w + 4
        ),
        np.uint8
    )

    cv2.floodFill(
        ff,
        ff_mask,
        (0, 0),
        255
    )

    filled = cv2.bitwise_or(
        padded,
        cv2.bitwise_not(ff)
    )

    return filled[
        1:-1,
        1:-1
    ]


def refine_mask(mask):

    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MORPH_OPEN_KSIZE
    )

    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MORPH_CLOSE_KSIZE
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        k_open
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        k_close
    )

    if FILL_HOLES:

        mask = fill_holes(
            mask
        )

    return mask


def detect_bbox(
    small_bgr,
    bgm
):

    mask = refine_mask(
        mask_from_background(
            small_bgr,
            bgm
        )
    )

    area_img = (
        small_bgr.shape[0]
        *
        small_bgr.shape[1]
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = [
        c
        for c in contours
        if
        (
            MIN_AREA_RATIO
            * area_img
        )
        <= cv2.contourArea(c)
        <=
        (
            MAX_AREA_RATIO
            * area_img
        )
    ]

    if not valid:

        return None

    return cv2.boundingRect(
        max(
            valid,
            key=cv2.contourArea
        )
    )


# ============================================================
# 5. LOAD MODEL + OOD
# ============================================================

def build_model_architecture(
    backbone_name,
    num_classes
):

    if (
        backbone_name
        == "mobilenet_v2"
    ):

        model = models.mobilenet_v2(
            weights=None
        )

        model.classifier[1] = (
            nn.Linear(
                model.last_channel,
                num_classes
            )
        )

    elif (
        backbone_name
        == "mobilenet_v3_large"
    ):

        model = (
            models.mobilenet_v3_large(
                weights=None
            )
        )

        model.classifier = (
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(
                    960,
                    num_classes
                )
            )
        )

    elif (
        backbone_name
        == "mobilenet_v3_small"
    ):

        model = (
            models.mobilenet_v3_small(
                weights=None
            )
        )

        model.classifier = (
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(
                    576,
                    num_classes
                )
            )
        )

    else:

        raise ValueError(
            f"Backbone không hỗ trợ: "
            f"{backbone_name}"
        )

    return model


def resolve_payload_path():

    if OOD_PAYLOAD_PATH:

        return OOD_PAYLOAD_PATH

    pointer = os.path.join(
        OOD_RUNS_DIR,
        "ood_detector_latest.txt"
    )

    if not os.path.exists(
        pointer
    ):

        raise FileNotFoundError(
            "Không thấy con trỏ latest: "
            f"{pointer}\n"
            "Hãy đặt OOD_PAYLOAD_PATH "
            "trỏ tay tới file .joblib."
        )

    with open(
        pointer,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read().strip()


def load_everything():

    payload_path = (
        resolve_payload_path()
    )

    if not os.path.exists(
        payload_path
    ):

        raise FileNotFoundError(
            "Không thấy payload OOD: "
            f"{payload_path}"
        )

    payload = joblib.load(
        payload_path
    )

    print(
        "Đã load payload OOD:",
        payload_path
    )

    print(
        f"  method="
        f"{payload['ood_method']}"
        f" | threshold="
        f"{payload['threshold']:.4f}"
        f" | target_tpr="
        f"{payload.get('target_tpr')}"
    )


    ckpt_path = (
        payload[
            "classifier_ckpt"
        ]
    )

    if not os.path.exists(
        ckpt_path
    ):

        raise FileNotFoundError(
            "Không thấy classifier "
            "mà payload trỏ tới: "
            f"{ckpt_path}"
        )


    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE
    )

    class_names = (
        ckpt["class_names"]
    )

    backbone_name = ckpt.get(
        "backbone_name",
        payload.get(
            "backbone_name",
            "mobilenet_v3_large"
        )
    )

    model = (
        build_model_architecture(
            backbone_name,
            len(class_names)
        )
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    model = model.to(
        DEVICE
    ).eval()

    print(
        f"Đã load classifier: "
        f"{backbone_name} | "
        f"{len(class_names)} class\n"
    )

    return (
        model,
        class_names,
        payload
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None

        self.gradients = None

        target_layer.register_forward_hook(
            self._save_activation
        )

        target_layer.register_full_backward_hook(
            self._save_gradient
        )


    def _save_activation(
        self,
        module,
        inp,
        out
    ):

        self.activations = (
            out.detach()
        )


    def _save_gradient(
        self,
        module,
        gin,
        gout
    ):

        self.gradients = (
            gout[0].detach()
        )


    def compute(
        self,
        input_tensor,
        class_idx
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        output[
            0,
            class_idx
        ].backward()

        weights = (
            self.gradients
            .mean(
                dim=(2, 3),
                keepdim=True
            )
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = (
            F.relu(cam)
            .squeeze()
            .cpu()
            .numpy()
        )

        if cam.max() > 0:

            cam = (
                cam
                / cam.max()
            )

        return (
            cam,
            output
        )


def overlay_heatmap(
    img_bgr,
    cam,
    alpha=HEATMAP_ALPHA
):

    H, W = img_bgr.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (W, H)
    )

    heatmap = cv2.applyColorMap(
        np.uint8(
            255 * cam_resized
        ),
        cv2.COLORMAP_JET
    )

    return cv2.addWeighted(
        heatmap,
        alpha,
        img_bgr,
        1 - alpha,
        0
    )


# ============================================================
# EMBEDDING + OOD SCORE
# ============================================================

@torch.no_grad()
def extract_embedding(
    model,
    input_tensor
):

    feat = model.features(
        input_tensor
    )

    pooled = (
        F.adaptive_avg_pool2d(
            feat,
            (1, 1)
        )
    )

    emb = torch.flatten(
        pooled,
        1
    )

    return (
        emb.cpu()
        .numpy()
        .astype(np.float32)
    )


def l2norm(
    X,
    eps=1e-8
):

    return (
        X /
        (
            np.linalg.norm(
                X,
                axis=1,
                keepdims=True
            )
            + eps
        )
    )


def score_cosine(
    X,
    centroids
):

    return (
        l2norm(X)
        @ centroids.T
    ).max(1)


def score_mahalanobis(
    X,
    means,
    precision
):

    N, C = (
        X.shape[0],
        means.shape[0]
    )

    dists = np.empty(
        (N, C),
        np.float64
    )

    for c in range(C):

        d = (
            X -
            means[c]
        )

        dists[:, c] = (
            np.einsum(
                "nd,de,ne->n",
                d,
                precision,
                d
            )
        )

    return (
        -dists.min(1)
    )


def ood_score(
    emb,
    payload
):

    method = (
        payload[
            "ood_method"
        ]
    )

    best_cos = float(
        score_cosine(
            emb,
            payload[
                "centroids"
            ]
        )[0]
    )

    if method == "mahalanobis":

        m = payload[
            "mahalanobis"
        ]

        s = float(
            score_mahalanobis(
                emb,
                m["means"],
                m["precision"]
            )[0]
        )

    else:

        s = best_cos

    return (
        s,
        best_cos
    )


# ============================================================
# TOP-5
# ============================================================

def get_top5_classes(
    probs,
    class_names
):

    """
    Lấy tối đa 5 class có xác suất cao nhất.
    """

    k = min(
        5,
        len(class_names)
    )

    top_probs, top_indices = (
        torch.topk(
            probs,
            k=k
        )
    )

    return [
        (
            class_names[
                int(idx)
            ],
            float(prob)
        )
        for prob, idx in zip(
            top_probs,
            top_indices
        )
    ]


def draw_top5_panel(
    img,
    top5
):

    """
    Vẽ bảng Top-5.

    Font:
        lớn khoảng 2.5 lần bản cũ.

    Khoảng cách dòng:
        tăng đáng kể để chữ không đè nhau.
    """

    panel = img.copy()

    # --------------------------------------------------------
    # Vị trí bảng
    # --------------------------------------------------------

    x0 = 25
    y0 = 100


    # --------------------------------------------------------
    # FONT
    # --------------------------------------------------------

    # Bản cũ:
    # title = 0.65
    # text  = 0.58
    #
    # Bản mới:
    # ~2.5 lần
    # --------------------------------------------------------

    title_scale = 1.60
    text_scale = 1.45

    title_thickness = 4
    text_thickness = 4


    # --------------------------------------------------------
    # KHOẢNG CÁCH DÒNG
    # --------------------------------------------------------

    line_h = 78


    # --------------------------------------------------------
    # KÍCH THƯỚC BẢNG
    # --------------------------------------------------------

    panel_w = 900

    top_padding = 50

    title_to_first_line = 70

    bottom_padding = 30

    panel_h = (
        top_padding
        +
        title_to_first_line
        +
        line_h * len(top5)
        +
        bottom_padding
    )


    h, w = panel.shape[:2]

    x1 = min(
        w,
        x0 + panel_w
    )

    y1 = min(
        h,
        y0 + panel_h
    )


    # ========================================================
    # BACKGROUND BÁN TRONG SUỐT
    # ========================================================

    roi = panel[
        y0:y1,
        x0:x1
    ].copy()

    if roi.size > 0:

        cv2.rectangle(
            roi,
            (0, 0),
            (
                roi.shape[1] - 1,
                roi.shape[0] - 1
            ),
            (40, 40, 40),
            -1
        )

        panel[
            y0:y1,
            x0:x1
        ] = cv2.addWeighted(
            roi,
            0.75,
            panel[
                y0:y1,
                x0:x1
            ],
            0.25,
            0
        )


    # ========================================================
    # TIÊU ĐỀ
    # ========================================================

    title_y = (
        y0 +
        top_padding
    )

    cv2.putText(
        panel,
        "Top-5 classes",
        (
            x0 + 25,
            title_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA
    )


    # ========================================================
    # CÁC CLASS
    # ========================================================

    first_line_y = (
        title_y
        +
        title_to_first_line
    )

    for i, (
        name,
        prob
    ) in enumerate(
        top5
    ):

        yy = (
            first_line_y
            +
            i * line_h
        )

        text_line = (
            f"{i + 1}. "
            f"{name}: "
            f"{prob * 100:.2f}%"
        )

        cv2.putText(
            panel,
            text_line,
            (
                x0 + 25,
                yy
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

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

    # Chỉ detect bbox khi có model nền (chế độ camera).
    # Chế độ ảnh: bgm = None -> bỏ qua trừ nền, dùng nguyên ảnh.
    bbox_small = (
        detect_bbox(small_bgr, bgm)
        if bgm is not None
        else None
    )


    Hf, Wf = (
        full_bgr.shape[:2]
    )


    # ========================================================
    # CROP
    # ========================================================

    if bbox_small is not None:

        x, y, w, h = [
            v * RESIZE_DIV
            for v in bbox_small
        ]

        pad = (
            CROP_PADDING
            *
            RESIZE_DIV
        )

        x1 = max(
            0,
            x - pad
        )

        y1 = max(
            0,
            y - pad
        )

        x2 = min(
            Wf,
            x + w + pad
        )

        y2 = min(
            Hf,
            y + h + pad
        )

        crop_bgr = full_bgr[
            y1:y2,
            x1:x2
        ].copy()

        bbox_full = (
            x1,
            y1,
            x2,
            y2
        )

    else:

        # Ở chế độ camera mới cảnh báo (bgm != None).
        # Chế độ ảnh cố tình dùng nguyên frame nên không in.
        if bgm is not None and verbose:

            print(
                "  CẢNH BÁO: "
                "không detect được vật "
                "-> dùng nguyên frame."
            )

        crop_bgr = (
            full_bgr.copy()
        )

        bbox_full = None


    # ========================================================
    # TRANSFORM
    # ========================================================

    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = transform(
        Image.fromarray(
            crop_rgb
        )
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )


    # ========================================================
    # 1. CLASSIFIER
    # ========================================================

    with torch.no_grad():

        probs = F.softmax(
            model(input_tensor),
            dim=1
        )[0]


    # ========================================================
    # TOP-5
    # ========================================================

    top5 = get_top5_classes(
        probs,
        class_names
    )


    # ========================================================
    # PREDICTION CHÍNH
    # ========================================================

    class_idx = int(
        probs.argmax().item()
    )

    pred_class = (
        class_names[
            class_idx
        ]
    )

    confidence = float(
        probs[
            class_idx
        ].item()
    )


    # ========================================================
    # IN TOP-5 RA TERMINAL
    # ========================================================

    if verbose:

        print(
            "  Top-5 classes:"
        )

        for rank, (
            name,
            prob
        ) in enumerate(
            top5,
            1
        ):

            print(
                f"    {rank}. "
                f"{name}: "
                f"{prob * 100:.2f}%"
            )


    # ========================================================
    # 2. OOD
    # ========================================================

    emb = extract_embedding(
        model,
        input_tensor
    )

    score, best_cos = (
        ood_score(
            emb,
            payload
        )
    )

    threshold = float(
        payload[
            "threshold"
        ]
    )

    is_ood = (
        score < threshold
    )


    if (
        USE_COSINE_GATE
        and
        best_cos <
        float(
            payload.get(
                "cosine_threshold",
                -1
            )
        )
    ):

        is_ood = True


    # ========================================================
    # 3. STATUS
    # ========================================================

    if is_ood:

        status = (
            "UNKNOWN_OOD"
        )

        label = (
            f"UNKNOWN (vat la) "
            f"sim={best_cos:.2f}"
        )

        color = (
            0,
            0,
            255
        )

        if verbose:

            print(
                f"  [X] TU CHOI (OOD) "
                f"| score={score:.2f} "
                f"< tau={threshold:.2f} "
                f"| best_cos={best_cos:.2f}"
            )

    else:

        status = "OK"

        label = (
            f"{pred_class} "
            f"({confidence * 100:.1f}%)"
        )

        color = (
            0,
            255,
            0
        )

        if verbose:

            print(
                f"  [OK] {pred_class} "
                f"({confidence * 100:.1f}%) "
                f"| score={score:.2f} "
                f">= tau={threshold:.2f}"
            )


    # ========================================================
    # 4. GRAD-CAM
    # ========================================================

    if (not is_ood) and do_gradcam:

        cam, _ = (
            gradcam.compute(
                input_tensor,
                class_idx
            )
        )

        overlay = (
            overlay_heatmap(
                crop_bgr,
                cam
            )
        )

    else:

        overlay = (
            crop_bgr.copy()
        )


    # ========================================================
    # LABEL
    # ========================================================

    cv2.putText(
        overlay,
        label,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        color,
        3,
        cv2.LINE_AA
    )


    # ========================================================
    # TOP-5 PANEL
    # ========================================================

    overlay = draw_top5_panel(
        overlay,
        top5
    )


    # ========================================================
    # ẢNH GỐC + BBOX
    # ========================================================

    disp_orig = (
        small_bgr.copy()
    )

    if bbox_full is not None:

        bx1, by1, bx2, by2 = [
            v // RESIZE_DIV
            for v in bbox_full
        ]

        cv2.rectangle(
            disp_orig,
            (
                bx1,
                by1
            ),
            (
                bx2,
                by2
            ),
            color,
            2
        )


    # ========================================================
    # GHÉP ẢNH
    # ========================================================

    th = (
        disp_orig.shape[0]
    )

    overlay_r = cv2.resize(
        overlay,
        (
            int(
                overlay.shape[1]
                *
                th
                /
                overlay.shape[0]
            ),
            th
        )
    )

    combined = np.hstack([
        disp_orig,
        overlay_r
    ])


    tag = (
        pred_class
        if not is_ood
        else "UNKNOWN"
    )


    return (
        combined,
        tag,
        status
    )


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    combined,
    tag,
    status
):

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    fname = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{status}_"
        f"{tag}.jpg"
    )

    path = os.path.join(
        SAVE_DIR,
        fname
    )

    imwrite_unicode(
        path,
        combined
    )

    print(
        f"  Đã lưu: {path}"
    )


# ============================================================
# 7. MAIN
# ============================================================

LIVE_WIN = "LIVE"

RESULT_WIN = (
    "REALTIME - Trai: goc+bbox | "
    "Phai: Grad-CAM + TOP-5 "
    "('s'=luu | 'r'=hoc lai nen | 'q'=thoat)"
)


# ------------------------------------------------------------
# CHẾ ĐỘ CAMERA (REALTIME)
# ------------------------------------------------------------

def run_camera(model, class_names, gradcam, payload):

    if not HIKROBOT_SDK_OK:
        print(
            "Không import được SDK Hikrobot (MvImport). "
            "Không thể chạy chế độ camera.\n"
            f"Chi tiết: {_HIKROBOT_IMPORT_ERROR}"
        )
        sys.exit(1)

    camera = HikrobotCamera()

    if (
        not camera.open()
        or
        not camera.start()
    ):

        print(
            "Không mở được camera Hikrobot."
        )

        camera.close()

        sys.exit(1)


    # ========================================================
    # HỌC NỀN
    # ========================================================

    bg = learn_background(
        camera
    )

    bgm = BackgroundModel(
        bg
    )


    print(
        "REALTIME đang chạy | "
        "'r' = học lại nền | "
        "'s' = lưu frame hiện tại | "
        "'q'/ESC = thoát.\n"
    )


    frame_id = 0

    last_combined = None
    last_tag = "UNKNOWN"
    last_status = "OK"

    # FPS (làm mượt)
    t_prev = time.time()
    fps = 0.0


    try:

        while True:

            full, small = (
                get_small_frame(
                    camera
                )
            )

            if small is None:

                continue


            frame_id += 1


            # =================================================
            # XỬ LÝ MỖI N FRAME
            # =================================================

            if (
                frame_id
                % PROCESS_EVERY_N_FRAMES
                == 0
            ):

                combined, tag, status = (
                    process_capture(
                        full,
                        small,
                        bgm,
                        model,
                        class_names,
                        gradcam,
                        payload,
                        verbose=False,   # tắt log để không ngập terminal
                        do_gradcam=REALTIME_GRADCAM
                    )
                )

                last_combined = combined
                last_tag = tag
                last_status = status


            # =================================================
            # FPS
            # =================================================

            now = time.time()

            dt = now - t_prev

            t_prev = now

            if dt > 0:

                fps = (
                    0.9 * fps
                    +
                    0.1 * (1.0 / dt)
                )


            # =================================================
            # HIỂN THỊ
            # =================================================

            if last_combined is not None:

                disp_res = cv2.resize(
                    last_combined,
                    None,
                    fx=RESULT_DISPLAY_SCALE,
                    fy=RESULT_DISPLAY_SCALE
                )

                cv2.putText(
                    disp_res,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                show_fit(
                    RESULT_WIN,
                    disp_res
                )

            else:

                show_fit(
                    LIVE_WIN,
                    small
                )


            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            # =================================================
            # QUIT
            # =================================================

            if key in (
                ord("q"),
                27
            ):

                break


            # =================================================
            # HỌC LẠI NỀN
            # =================================================

            elif key == ord("r"):

                bg = (
                    learn_background(
                        camera
                    )
                )

                bgm = (
                    BackgroundModel(
                        bg
                    )
                )


            # =================================================
            # LƯU FRAME HIỆN TẠI
            # =================================================

            elif (
                key == ord("s")
                and last_combined is not None
            ):

                save_result(
                    last_combined,
                    last_tag,
                    last_status
                )


    finally:

        camera.close()

        cv2.destroyAllWindows()


# ------------------------------------------------------------
# CHẾ ĐỘ ẢNH (THƯ MỤC NHIỀU ẢNH)
# ------------------------------------------------------------

def run_image_folder(
    folder,
    model,
    class_names,
    gradcam,
    payload,
    bgm=None
):

    if not os.path.isdir(folder):
        print("Thư mục ảnh không tồn tại:", folder)
        return

    files = list_images(folder)

    if not files:
        print("Không tìm thấy ảnh trong:", folder)
        return

    print(f"Tìm thấy {len(files)} ảnh trong: {folder}")
    print(
        "Phím: SPACE/d = ảnh sau | "
        "a = ảnh trước | "
        "s = lưu | "
        "q/ESC = thoát\n"
    )

    idx = 0

    while 0 <= idx < len(files):

        path = files[idx]
        img = imread_unicode(path)

        if img is None:
            print(
                f"[{idx + 1}/{len(files)}] "
                f"LỖI đọc ảnh: {path}"
            )
            idx += 1
            continue

        print(
            f"[{idx + 1}/{len(files)}] "
            f"{os.path.basename(path)}"
        )

        h, w = img.shape[:2]

        small = cv2.resize(
            img,
            (
                max(1, w // RESIZE_DIV),
                max(1, h // RESIZE_DIV)
            )
        )

        combined, tag, status = process_capture(
            img,
            small,
            bgm,       # None -> bỏ trừ nền, dùng nguyên ảnh
            model,
            class_names,
            gradcam,
            payload
        )

        disp_res = cv2.resize(
            combined,
            None,
            fx=RESULT_DISPLAY_SCALE,
            fy=RESULT_DISPLAY_SCALE
        )

        show_fit(
            RESULT_WIN,
            disp_res
        )

        # Cho phép bấm 's' nhiều lần trước khi chuyển ảnh.
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
            # SPACE, 'd', hoặc phím khác -> ảnh sau
            idx += 1

    cv2.destroyWindow(RESULT_WIN)
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    mode = RUN_MODE
    folder = IMAGE_FOLDER

    # Cho phép chọn chế độ qua dòng lệnh:
    #   python live_detect.py                    -> theo RUN_MODE
    #   python live_detect.py camera
    #   python live_detect.py images
    #   python live_detect.py images <thu_muc>
    #   python live_detect.py <thu_muc>          -> tự nhận là chế độ ảnh
    args = sys.argv[1:]

    if args:
        a0 = args[0].lower()

        if a0 in ("camera", "cam"):
            mode = "camera"

        elif a0 in ("images", "image", "img", "folder"):
            mode = "images"
            if len(args) >= 2:
                folder = args[1]

        elif os.path.isdir(args[0]):
            mode = "images"
            folder = args[0]


    # ========================================================
    # LOAD MODEL + OOD (dùng chung cho cả 2 chế độ)
    # ========================================================

    model, class_names, payload = load_everything()


    # ========================================================
    # GRAD-CAM
    # ========================================================

    gradcam = GradCAM(
        model,
        model.features[-1]
    )


    # ========================================================
    # CHỌN CHẾ ĐỘ
    # ========================================================

    if mode == "images":
        print(f"=== CHẾ ĐỘ ẢNH: {folder} ===\n")
        run_image_folder(
            folder,
            model,
            class_names,
            gradcam,
            payload
        )
    else:
        print("=== CHẾ ĐỘ CAMERA (REALTIME) ===\n")
        run_camera(
            model,
            class_names,
            gradcam,
            payload
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()