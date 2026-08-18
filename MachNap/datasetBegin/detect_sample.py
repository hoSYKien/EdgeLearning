"""
LIVE Camera Hikrobot + OOD Isolation Forest + Nearest Centroid + Prototype Heatmap:
  - Tự động crop linh kiện qua Background Subtraction.
  - Lọc vật thể lạ qua 2 lớp (Isolation Forest + Cosine Centroid Threshold).
  - Trực quan hóa vùng nhìn qua PROTOTYPE SIMILARITY MAP (thay cho Grad-CAM).
    => Tại mỗi ô 7x7 của feature map, đo cosine với centroid của class dự đoán.
       Không cần backward, không dính noise từ L2-normalize / hardswish.

PHÍM ĐIỀU KHIỂN:
  c     : Chụp + Phân tích OOD/Centroid + Heatmap
  r     : Học lại nền (Background)
  s     : Lưu ảnh kết quả (tại màn hình kết quả)
  q/ESC : Thoát
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
#  KHỞI TẠO SDK HIKROBOT (MVS)
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(dll_dir):
    os.add_dll_directory(dll_dir)
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)

try:
    from MvCameraControl_class import *
except ImportError:
    print("Cảnh báo: Không tìm thấy thư viện MvCameraControl_class. Hãy kiểm tra lại đường dẫn SDK Hikrobot.")

# ============================================================
# 1. CẤU HÌNH HỆ THỐNG
# ============================================================
MODEL_DIR = r"D:\TongHop\RTC Technologi\G8\model\model3"
BACKBONE_NAME = "mobilenet_v3_large"

PKL_PATH = os.path.join(MODEL_DIR, f"centroid_ood_detector_{BACKBONE_NAME}.joblib")
BACKBONE_WEIGHTS = os.path.join(MODEL_DIR, f"backbone_{BACKBONE_NAME}.pt")
SAVE_DIR = r"D:\TongHop\RTC Technologi\G8\gradcam_captures"

CAMERA_MODEL_HINT = "MV-CS200"   # Để "" nếu lấy camera đầu tiên
RESIZE_DIV = 4                   # Thu nhỏ frame trước khi detect cho nhẹ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COSINE_THRESHOLD = 0.65          # Ngưỡng tương đồng cosine tối thiểu (NÊN calibrate từ tập ID/OOD)

# --- Ngưỡng Background Subtraction ---
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

# --- Học nền ---
WARMUP_FRAMES = 30
BG_FRAMES = 40

# --- Hiển thị ---
HEATMAP_ALPHA = 0.45
RESULT_DISPLAY_SCALE = 0.5
DISPLAY_MAX_W, DISPLAY_MAX_H = 1280, 720

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 2. CAMERA SDK WRAPPER
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
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print("Không tìm thấy camera nào! Mã lỗi:", ret)
            return False

        target_index = 0
        for i in range(deviceList.nDeviceNum):
            info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            name = ""
            if info.nTLayerType == MV_GIGE_DEVICE:
                name = "".join(chr(c) for c in info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            elif info.nTLayerType == MV_USB_DEVICE:
                name = "".join(chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            if self.model_hint and self.model_hint in name:
                target_index = i

        stDeviceInfo = cast(deviceList.pDeviceInfo[target_index], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        if self.cam.MV_CC_CreateHandle(stDeviceInfo) != 0:
            return False
        if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
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
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
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
    full = camera.read()
    if full is None:
        return None, None
    h, w = full.shape[:2]
    small = cv2.resize(full, (w // RESIZE_DIV, h // RESIZE_DIV))
    return full, small


# ============================================================
# 3. BACKGROUND SUBTRACTION
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
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 2)
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
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        show_fit("LIVE", disp)
        cv2.waitKey(1)
    print("Học nền hoàn tất!\n")
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


def refine_mask(mask):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN_KSIZE)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KSIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    if FILL_HOLES:
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:-1, 1:-1] = mask
        ff = padded.copy()
        ff_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        filled = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
        mask = filled[1:-1, 1:-1]
    return mask


def detect_bbox(small_bgr, bgm):
    mask = refine_mask(mask_from_background(small_bgr, bgm))
    area_img = small_bgr.shape[0] * small_bgr.shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if MIN_AREA_RATIO * area_img <= cv2.contourArea(c) <= MAX_AREA_RATIO * area_img]
    if not valid:
        return None
    return cv2.boundingRect(max(valid, key=cv2.contourArea))


# ============================================================
# 4. LOAD CENTROID MODEL + PROTOTYPE SIMILARITY MAP
# ============================================================
class CentroidModel(nn.Module):
    """Forward-only: trả về sims (cosine với centroid), embedding chuẩn hóa và feature map thô."""
    def __init__(self, backbone, centroids):
        super().__init__()
        self.backbone = backbone
        self.centroids = centroids  # [num_classes, dim], đã là unit vector từ lúc train

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone.features(x)              # [B, C, 7, 7]
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        emb = torch.flatten(pooled, 1)
        emb_norm = F.normalize(emb, p=2, dim=1)
        sims = torch.mm(emb_norm, self.centroids.t())  # [B, num_classes]
        return sims, emb_norm, feat


@torch.no_grad()
def prototype_cam(feat, centroid_unit):
    """
    Prototype similarity map: cosine giữa vector đặc trưng tại mỗi ô 7x7 và centroid.
    - feat: [1, C, 7, 7] (feature map thô từ backbone.features)
    - centroid_unit: [C] vector centroid đã chuẩn hóa của class dự đoán
    Không backward => không dính noise từ L2-normalize / hardswish như Grad-CAM.
    """
    feat_n = F.normalize(feat, p=2, dim=1)                  # chuẩn hóa từng ô theo kênh
    cam = torch.einsum('bchw,c->bhw', feat_n, centroid_unit)  # cosine tại mỗi ô
    cam = F.relu(cam)[0]                                    # bỏ phần "ngược hướng" centroid
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def overlay_heatmap(img_bgr, cam, alpha=HEATMAP_ALPHA):
    H, W = img_bgr.shape[:2]
    # INTER_CUBIC cho mượt hơn vì cam gốc chỉ 7x7
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)


# ============================================================
# 5. QUY TRÌNH XỬ LÝ 1 LẦN CHỤP
# ============================================================
def process_capture(full_bgr, small_bgr, bgm, model, ood_detector, class_names):
    bbox_small = detect_bbox(small_bgr, bgm)
    Hf, Wf = full_bgr.shape[:2]

    if bbox_small is not None:
        x, y, w, h = [v * RESIZE_DIV for v in bbox_small]
        pad = CROP_PADDING * RESIZE_DIV
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(Wf, x + w + pad), min(Hf, y + h + pad)
        crop_bgr = full_bgr[y1:y2, x1:x2].copy()
        bbox_full = (x1, y1, x2, y2)
    else:
        print("  [!] Không tách được vật thể nền -> Dùng toàn bộ frame.")
        crop_bgr = full_bgr.copy()
        bbox_full = None

    # Forward qua Model (không cần gradient)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)

    sims, emb_norm, feat = model(input_tensor)

    # 1. Tìm class có Cosine Similarity cao nhất
    best_class_idx = int(sims.argmax(dim=1).item())
    best_sim = float(sims[0, best_class_idx].item())
    pred_class_name = class_names[best_class_idx]

    # 2. Prototype heatmap theo centroid của class dự đoán
    cam = prototype_cam(feat, model.centroids[best_class_idx])
    emb_np = emb_norm.cpu().numpy()

    # 3. LỚP BẢO VỆ 1: ISOLATION FOREST OOD CHECK
    is_inlier = ood_detector.predict(emb_np)[0]
    anomaly_score = float(ood_detector.decision_function(emb_np)[0])

    # Xác định trạng thái & nhãn
    if is_inlier == -1:
        status = "UNKNOWN_OOD"
        label_str = "UNKNOWN (Vat The La!)"
        color = (0, 0, 255)  # Đỏ
        print(f"  [X] Từ chối bởi Isolation Forest (Anomaly score: {anomaly_score:.3f})")
    elif best_sim < COSINE_THRESHOLD:
        status = "UNKNOWN_LOW_SIM"
        label_str = f"UNKNOWN (Sim={best_sim:.2f} < {COSINE_THRESHOLD})"
        color = (0, 215, 255)  # Vàng
        print(f"  [?] Độ tương đồng thấp với '{pred_class_name}' ({best_sim:.2%})")
    else:
        status = "ACCEPTED"
        label_str = f"{pred_class_name} ({best_sim*100:.1f}%)"
        color = (0, 255, 0)  # Xanh lá
        print(f"  [OK] Nhận diện: {pred_class_name} ({best_sim*100:.1f}%)")

    # Render kết quả
    # Chú ý: heatmap chỉ có ý nghĩa với mẫu ACCEPTED. Với mẫu bị từ chối,
    # class dự đoán là ngẫu nhiên nên heatmap không đáng tin -> làm mờ đi.
    if status == "ACCEPTED":
        overlay = overlay_heatmap(crop_bgr, cam)
    else:
        overlay = crop_bgr.copy()
    cv2.putText(overlay, label_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    disp_orig = small_bgr.copy()
    if bbox_full is not None:
        bx1, by1, bx2, by2 = [v // RESIZE_DIV for v in bbox_full]
        cv2.rectangle(disp_orig, (bx1, by1), (bx2, by2), color, 2)

    th = disp_orig.shape[0]
    overlay_r = cv2.resize(overlay, (int(overlay.shape[1] * th / overlay.shape[0]), th))
    combined = np.hstack([disp_orig, overlay_r])

    return combined, pred_class_name, status


def save_result(combined, pred_class, status):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{status}_{pred_class}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(path, combined)
    print(f"  Đã lưu ảnh kết quả: {path}")


# ============================================================
# 6. HÀM MAIN
# ============================================================
LIVE_WIN = "LIVE CAMERA - Nhan 'c' de test | 'r' hoc lai nen | 'q' thoat"
RESULT_WIN = "KET QUA - Trai: Bbox | Phai: Heatmap ('s'=luu, phim khac=tiep tuc)"


def main():
    if not os.path.exists(PKL_PATH) or not os.path.exists(BACKBONE_WEIGHTS):
        print(f"Lỗi: Không tìm thấy file model tại {MODEL_DIR}")
        print("Hãy chạy file train 'train_centroid_ood.py' trước!")
        sys.exit(1)

    # 1. Load Detector & Weights
    payload = joblib.load(PKL_PATH)
    class_names = payload["class_names"]
    centroids = torch.from_numpy(payload["centroids"]).float().to(DEVICE)
    ood_detector = payload["ood_detector"]

    if BACKBONE_NAME == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large()
    elif BACKBONE_NAME == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small()
    else:
        backbone = models.mobilenet_v2()

    backbone.load_state_dict(torch.load(BACKBONE_WEIGHTS, map_location=DEVICE))
    backbone = backbone.to(DEVICE).eval()

    model = CentroidModel(backbone, centroids)

    # 2. Khởi tạo Camera
    camera = HikrobotCamera()
    if not camera.open() or not camera.start():
        print("Không thể kết nối Camera Hikrobot.")
        camera.close()
        sys.exit(1)

    # 3. Học nền ban đầu
    bg = learn_background(camera)
    bgm = BackgroundModel(bg)

    print("\n" + "=" * 50)
    print(" HỆ THỐNG ĐÃ SẴN SÀNG")
    print(" Phím 'c': Chụp & Kiểm tra vật thể lạ")
    print(" Phím 'r': Học lại nền khi đổi góc/ánh sáng")
    print(" Phím 'q': Thoát chương trình")
    print("=" * 50 + "\n")

    try:
        while True:
            full, small = get_small_frame(camera)
            if small is None:
                continue

            disp = small.copy()
            cv2.putText(disp, "c: Chup + Kiem tra | r: Hoc nen | q: Thoat",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
            show_fit(LIVE_WIN, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            if key == ord('r'):
                bg = learn_background(camera)
                bgm = BackgroundModel(bg)
                continue

            if key == ord('c'):
                print("\n-> Đang chụp & xử lý frame...")
                combined, pred_class, status = process_capture(
                    full, small, bgm, model, ood_detector, class_names
                )
                disp_res = cv2.resize(combined, None, fx=RESULT_DISPLAY_SCALE, fy=RESULT_DISPLAY_SCALE)
                show_fit(RESULT_WIN, disp_res)

                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('s'):
                    save_result(combined, pred_class, status)
                cv2.destroyWindow(RESULT_WIN)
                if k2 in (ord('q'), 27):
                    break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()