# """
# Pipeline: Robust Background Subtraction → MobileSAM (vật chuyển động trên băng tải)
#   [1]   Color diff per-channel max
#   [2]   Loại bóng (3 điều kiện)
#   [2.5] Hysteresis 2 ngưỡng (chống loang)
#   [2.7] Lấp thủng: flood-fill lỗ kín + convex hull khuyết góc (có kiểm soát)
#   [3]   Xác nhận mức BLOB bằng tracking → tạo STABLE MASK HOÀN CHỈNH
#   [4]   (CUỐI CÙNG) Watershed trên stable mask → bbox
#   [5]   Bbox → MobileSAM segment (nguyên bản)
#
# ĐÃ CHỈNH: input camera Irayple -> camera Hikrobot (MVS SDK).
# """

import sys
import os
import time
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ============================================================
#  KHỞI TẠO SDK HIKROBOT (MVS)  — theo mẫu script camera của bạn
# ============================================================
# 1. Trỏ tới thư mục chứa DLL runtime của MVS
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(dll_dir)

# 2. Trỏ THẲNG vào thư mục MvImport (nằm cùng cấp với file này)
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)

# 3. Import kiểu module thường (không có tiền tố MvImport.)
from MvCameraControl_class import *


# ============================================================
#  CẤU HÌNH ROI & PIPELINE (GIỮ NGUYÊN BẢN GỐC)
# ============================================================
# ĐÃ XOÁ ROI: detect trên TOÀN BỘ ảnh.
# Giữ roi_x/roi_y = 0 để phần vẽ bbox không phải sửa offset.
roi_x, roi_y = 0, 0

# Hysteresis 2 ngưỡng
THRESH_STRONG = 35
THRESH_WEAK = 20

CHROMA_TOL = 0.04
SHADOW_DIM = (0.3, 0.95)
SHADOW_MAX_DIFF = 70

MIN_AREA = 1000
PADDING = 0        # đệm quanh bbox. 0 = ôm SÁT mép mask. Tăng nếu muốn chừa lề.

# Lấp thủng / bọc lồi
# SOLIDITY_MIN: blob solidity >= mức này sẽ bị thay bằng convex hull (phình lồi,
# bo góc, lấp chỗ lõm). Để ~0.97-1.0 -> giữ viền THẬT, ôm sát vật. Hạ xuống nếu
# muốn làm liền các blob rỗ/đứt đoạn.
SOLIDITY_MIN = 0.97

# Morphology trong build_mask:
#   OPEN  = dọn chấm nhiễu lẻ (kernel càng lớn càng dọn mạnh, nhưng ăn mòn mép).
#   CLOSE = nối khe hở/nét đứt (kernel càng lớn càng nở & bo tròn mép vật).
# Muốn ôm SÁT hơn -> giảm CLOSE về (3,3). Nhiễu nhiều -> tăng OPEN.
MORPH_OPEN_KSIZE = (5, 5)
MORPH_CLOSE_KSIZE = (3, 3)

# Cách vẽ kết quả:
#   VE_CONTOUR = True  -> vẽ đường viền ÔM SÁT vật (theo mask), màu đỏ.
#   VE_BBOX    = True  -> vẽ thêm hình chữ nhật bao (vàng). Tắt nếu chỉ cần viền.
#   CONTOUR_SMOOTH: độ MƯỢT của viền. Gồm 2 tầng: blur nhẹ mask + lọc trung bình
#                   trượt trên các điểm viền. Số càng lớn viền càng trơn.
#                   Thử 15 / 25 / 41... 0 = tắt (sát pixel, sần).
#   APPROX_CONTOUR: rút gọn viền thành các đoạn thẳng (0 = tắt). Hợp với vật
#                   nhiều cạnh thẳng; thường để 0 khi đã dùng CONTOUR_SMOOTH.
# (bboxes vẫn luôn được tính để đưa vào SAM, bất kể vẽ hay không.)
VE_CONTOUR = True
VE_BBOX = False
CONTOUR_SMOOTH = 25
APPROX_CONTOUR = 0.0

# Xác nhận blob bằng tracking
N_CONFIRM = 2
MAX_MOVE = 180
MAX_MISSED = 2

# Watershed (bước cuối)
SPLIT_MIN_AREA = 10000   # blob nhỏ hơn mức này không bao giờ tách
MIN_PEAK_DISTANCE = 60
BORDER_GRAD_MIN = 45

# Học nền
WARMUP_FRAMES = 60
BG_FRAMES = 60
BG_MAX_RESIDUAL = 8.0

# Camera Hikrobot: ưu tiên chọn model có chuỗi này (để trống "" = lấy camera đầu tiên)
CAMERA_MODEL_HINT = "MV-CS200"

# Tỉ lệ thu nhỏ frame trước khi detect. Camera Hikrobot độ phân giải rất lớn;
# đặt RESIZE_DIV lớn hơn để frame nhỏ lại, chạy nhanh + cửa sổ hiển thị vừa màn
# hình. Lưu ý: MIN_AREA / SPLIT_MIN_AREA đang canh theo ảnh nhỏ, nếu để frame
# quá lớn có thể phải chỉnh lại các ngưỡng diện tích cho hợp.
RESIZE_DIV = 4


# ================================


def smooth_contour(c, k):
    """Làm mượt đường viền bằng trung bình trượt VÒNG (contour là vòng kín).
    k = cửa sổ trung bình (điểm), càng lớn càng mượt. Trả contour đã mượt."""
    k = int(k) | 1                     # ép số lẻ
    pts = c[:, 0, :].astype(np.float32)
    n = len(pts)
    if k < 3 or n < k:
        return c
    pad = k // 2
    kernel = np.ones(k, np.float32) / k
    xp = np.concatenate([pts[-pad:, 0], pts[:, 0], pts[:pad, 0]])
    yp = np.concatenate([pts[-pad:, 1], pts[:, 1], pts[:pad, 1]])
    xs = np.convolve(xp, kernel, mode="valid")
    ys = np.convolve(yp, kernel, mode="valid")
    return np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)


def get_roi(frame):
    # ĐÃ XOÁ ROI: xử lý toàn bộ ảnh, không cắt, không áp mask đa giác.
    return frame


# ============================================================
#  WRAPPER CAMERA HIKROBOT
#  Cung cấp .read() trả về 1 frame BGR full-res (hoặc None),
#  để get_resized_frame() / learn_background() dùng y như bản Irayple.
# ============================================================
class HikrobotCamera:
    def __init__(self, model_hint=CAMERA_MODEL_HINT):
        self.model_hint = model_hint
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = 0
        self.stFrameInfo = None
        self._convert_buf = None   # buffer tái sử dụng cho MV_CC_ConvertPixelType

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

        # GigE: tối ưu packet size
        if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if nPacketSize > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        # Chế độ chụp liên tục (không trigger)
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
        """Lấy 1 frame, chuyển sang BGR full-res. Trả None nếu lỗi frame.

        Màu: KHÔNG tự demosaic Bayer bằng OpenCV (dễ chọn nhầm pattern -> sai
        màu). Để CHÍNH SDK MVS convert sang BGR8_Packed, nó biết đúng sensor."""
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000)
        if ret != 0:
            return None

        w = self.stFrameInfo.nWidth
        h = self.stFrameInfo.nHeight
        n = self.stFrameInfo.nFrameLen
        pt = self.stFrameInfo.enPixelType

        # Mono8: ảnh xám -> BGR
        if pt == PixelType_Gvsp_Mono8:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n).reshape(h, w)
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        # Đã sẵn BGR8 packed: dùng luôn
        if pt == PixelType_Gvsp_BGR8_Packed:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n)
            return raw.reshape(h, w, 3).copy()

        # Đã sẵn RGB8 packed -> đảo về BGR
        if pt == PixelType_Gvsp_RGB8_Packed:
            raw = np.frombuffer(self.data_buf, dtype=np.uint8, count=n)
            return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        # Còn lại (Bayer...): nhờ SDK demosaic -> BGR8 để MÀU CHUẨN
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


def show_fit(winname, img, max_w=1280, max_h=720):
    """Hiển thị co vừa màn hình (CHỈ ảnh preview, không đụng dữ liệu detect).
    Ảnh sau resize để xử lý vẫn có thể lớn hơn màn hình -> co nhỏ khi show để
    thấy TRỌN khung hình (giống MVS), tránh cảm giác 'bị zoom'."""
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imshow(winname, img)


def get_resized_frame(camera):
    """Đọc 1 frame từ camera + thu nhỏ 1/RESIZE_DIV cho nhẹ và vừa màn hình."""
    img = camera.read()
    if img is not None:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // RESIZE_DIV, h // RESIZE_DIV))
    return img


def learn_background(camera, num_frames=BG_FRAMES, warmup_frames=WARMUP_FRAMES):
    print(f"Chờ camera ổn định ({warmup_frames} frames)...")
    skipped = 0
    while skipped < warmup_frames:
        img = get_resized_frame(camera)
        if img is None:
            continue
        skipped += 1
        disp = img.copy()
        cv2.putText(disp, f"Camera warming up... {skipped}/{warmup_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        show_fit("Detection", disp)
        cv2.waitKey(1)

    print("Học nền... giữ ROI trống")
    frames = []
    while len(frames) < num_frames:
        img = get_resized_frame(camera)
        if img is None:
            continue
        frames.append(cv2.GaussianBlur(get_roi(img), (7, 7), 0).astype(np.float32))
        disp = img.copy()
        cv2.putText(disp, f"Learning BG... {len(frames)}/{num_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        show_fit("Detection", disp)
        cv2.waitKey(1)
    print("Xong!")
    return np.mean(frames, axis=0)


def learn_background_verified(camera):
    warmup = WARMUP_FRAMES
    while True:
        bg = learn_background(camera, warmup_frames=warmup)
        img = None
        while img is None:
            img = get_resized_frame(camera)
        cur = cv2.GaussianBlur(get_roi(img), (7, 7), 0).astype(np.float32)
        residual = np.mean(np.abs(cur - bg))
        print(f"Độ lệch nền vs frame hiện tại: {residual:.1f}")
        if residual < BG_MAX_RESIDUAL:
            return bg
        print("Nền chưa ổn định, học lại...")
        warmup = 30


def fill_holes(mask):
    """Flood-fill từ góc → vùng không tràn tới = lỗ kín → tô đầy."""
    h, w = mask.shape
    ff = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def solidify(mask, solidity_min=SOLIDITY_MIN):
    """Convex hull lấp khuyết góc/hông — chỉ khi blob đã gần lồi."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and area / hull_area >= solidity_min:
            cv2.drawContours(out, [hull], -1, 255, -1)
        else:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


NECK_RATIO = 0.6   # bề dày tại cổ < 60% bề dày 2 đỉnh → có eo thắt thật
def has_neck(dist, p1, p2):
    """Kiểm tra đường nối 2 đỉnh có đi qua chỗ thắt (mỏng hơn hẳn) không.
    p1, p2: (y, x). Trả về True nếu có eo thắt thật."""
    n_samples = 50
    ys = np.linspace(p1[0], p2[0], n_samples).astype(int)
    xs = np.linspace(p1[1], p2[1], n_samples).astype(int)
    profile = dist[ys, xs]                    # bề dày dọc đường nối

    peak_thickness = min(dist[p1[0], p1[1]], dist[p2[0], p2[1]])
    neck_thickness = profile.min()

    return neck_thickness < peak_thickness * NECK_RATIO


def watershed_split(blob_mask, roi_bgr):
    """CHỈ tách khi có eo thắt hình học thật sự.
    Vật dài/to không có cổ → không bao giờ bị cắt, bất kể bao nhiêu đỉnh."""
    dist = ndimage.distance_transform_edt(blob_mask)
    coords = peak_local_max(dist, min_distance=MIN_PEAK_DISTANCE,
                            labels=blob_mask.astype(bool))
    if len(coords) <= 1:
        return [blob_mask]

    # === KIỂM TRA EO THẮT: chỉ giữ các đỉnh thật sự bị ngăn cách bởi cổ ===
    # Gom đỉnh: 2 đỉnh không có cổ giữa chúng = cùng 1 vật → giữ 1 đại diện
    kept = []
    for c in coords:
        merged = False
        for k in kept:
            if not has_neck(dist, tuple(c), tuple(k)):
                merged = True          # không có cổ → cùng vật với đỉnh đã giữ
                break
        if not merged:
            kept.append(tuple(c))

    if len(kept) <= 1:
        return [blob_mask]             # mọi đỉnh cùng 1 vật → không tách

    # === Có >= 2 đỉnh bị cổ ngăn cách → tách bằng watershed trên distance ===
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(kept):
        markers[y, x] = i + 1

    labels = watershed(-dist, markers, mask=blob_mask.astype(bool))

    sub_masks = []
    for lb in range(1, labels.max() + 1):
        sub = ((labels == lb) * 255).astype(np.uint8)
        if cv2.countNonZero(sub) >= MIN_AREA:
            sub_masks.append(sub)

    return sub_masks if sub_masks else [blob_mask]


class Detector:
    """Bước [1]→[3]: tạo STABLE MASK hoàn chỉnh. KHÔNG watershed ở đây."""

    def __init__(self, roi_shape, bg):
        self.bg = bg.astype(np.float32)
        self.bg_uint8 = bg.astype(np.uint8)
        self.tracks = []

        # Tối ưu hóa: tính sẵn chroma và mean của background một lần duy nhất bằng các hàm OpenCV
        bg_b, bg_g, bg_r = cv2.split(self.bg)
        self.bg_sum = bg_b + bg_g + bg_r + 3.0
        self.bg_sum_3d = cv2.merge([self.bg_sum, self.bg_sum, self.bg_sum])
        self.bg_plus_1 = self.bg + 1.0

        # Tính sẵn các ngưỡng độ sáng cho shadow removal
        self.shadow_dim_low = SHADOW_DIM[0] * self.bg_sum
        self.shadow_dim_high = SHADOW_DIM[1] * self.bg_sum

    def _decay_tracks(self, roi_shape):
        """Frame không có vật: già hoá track (missed++) & trả mask rỗng."""
        kept = []
        for t in self.tracks:
            t['missed'] += 1
            if t['missed'] <= MAX_MISSED:
                kept.append(t)
        self.tracks = kept
        return np.zeros((roi_shape[0], roi_shape[1]), dtype=np.uint8)

    def build_mask(self, roi):
        # [1] Kiểm tra nhanh ở dạng uint8 để thoát cực sớm nếu không có vật thể
        roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)
        diff_c_uint8 = cv2.absdiff(roi_blur, self.bg_uint8)
        b_u, g_u, r_u = cv2.split(diff_c_uint8)
        diff_uint8 = cv2.max(cv2.max(b_u, g_u), r_u)

        # Chỉ chạy tiếp nếu có ít nhất 1 pixel biến động đủ lớn (THRESH_STRONG - 1 để bù sai số làm tròn)
        if cv2.minMaxLoc(diff_uint8)[1] < THRESH_STRONG - 1:
            return self._decay_tracks(roi.shape)

        # [2] Xác định vùng hoạt động (weak) ngay trên diff uint8 rẻ, chạy full ROI.
        #     Nhờ vậy toàn bộ toán float khử bóng đắt đỏ chỉ chạy trên vùng crop nhỏ.
        weak_full = (diff_uint8 > THRESH_WEAK).astype(np.uint8)
        x_w, y_w, w_w, h_w = cv2.boundingRect(weak_full)
        if w_w == 0 or h_w == 0:
            return self._decay_tracks(roi.shape)

        # Thêm padding an toàn để tránh mất biên khi chạy Morphology
        pad = 5
        y1 = max(0, y_w - pad)
        y2 = min(roi.shape[0], y_w + h_w + pad)
        x1 = max(0, x_w - pad)
        x2 = min(roi.shape[1], x_w + w_w + pad)

        # [3] Diff float + khử bóng CHỈ trên vùng crop (dùng các mảng nền cắt tương ứng)
        roi_f = roi_blur[y1:y2, x1:x2].astype(np.float32)
        diff_c = cv2.absdiff(roi_f, self.bg[y1:y2, x1:x2])
        b_f, g_f, r_f = cv2.split(diff_c)
        diff = cv2.max(cv2.max(b_f, g_f), r_f)

        # S_roi phải là TỔNG ĐỘ SÁNG của pixel hiện tại (roi_b+roi_g+roi_r),
        # KHÔNG phải tổng độ lệch |roi-bg| — nếu không mô hình bóng sẽ sai hoàn toàn.
        rb, rg, rr = cv2.split(roi_f)
        S_roi = rb + rg + rr + 3.0
        S_roi_3d = cv2.merge([S_roi, S_roi, S_roi])

        # so khớp màu sắc: | (I_i + 1) * S_bg - (BG_i + 1) * S_roi | < CHROMA_TOL * S_roi * S_bg
        term_roi = (roi_f + 1.0) * self.bg_sum_3d[y1:y2, x1:x2]
        term_bg = self.bg_plus_1[y1:y2, x1:x2] * S_roi_3d

        chroma_diff = cv2.absdiff(term_roi, term_bg)
        bc, gc, rc = cv2.split(chroma_diff)
        max_diff_term = cv2.max(cv2.max(bc, gc), rc)

        threshold_map = (CHROMA_TOL * self.bg_sum[y1:y2, x1:x2]) * S_roi
        same_color = max_diff_term < threshold_map

        # kiểm tra tỉ lệ độ sáng: SHADOW_DIM[0] * S_bg < S_roi < SHADOW_DIM[1] * S_bg
        dim_moderate = ((S_roi > self.shadow_dim_low[y1:y2, x1:x2]) &
                        (S_roi < self.shadow_dim_high[y1:y2, x1:x2]))

        # xóa bỏ bóng đổ
        diff[same_color & dim_moderate & (diff < SHADOW_MAX_DIFF)] = 0

        # [4] Hysteresis trên vùng crop
        strong_crop = (diff > THRESH_STRONG).astype(np.uint8)
        weak_crop = (diff > THRESH_WEAK).astype(np.uint8)

        # Sau khi khử bóng, nếu không còn seed strong nào thì coi như không có vật
        if cv2.countNonZero(strong_crop) == 0:
            return self._decay_tracks(roi.shape)

        n_lbl, lbl_crop = cv2.connectedComponents(weak_crop, connectivity=8)
        strong_labels = np.unique(lbl_crop[strong_crop > 0])
        strong_labels = strong_labels[strong_labels != 0]

        # Ánh xạ nhãn nhanh
        map_arr = np.zeros(n_lbl, dtype=bool)
        map_arr[strong_labels] = True
        mask_crop = map_arr[lbl_crop].astype(np.uint8) * 255

        # [5] Morphology trên vùng Crop (nhỏ hơn hàng chục lần so với ảnh gốc)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN_KSIZE)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE_KSIZE)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, k_open)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k_close)

        # [6] Lấp lỗ kín trên vùng Crop (an toàn tuyệt đối nhờ đệm 1px)
        h_c, w_c = mask_crop.shape
        padded = np.zeros((h_c + 2, w_c + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = mask_crop
        ff = padded.copy()
        ff_mask = np.zeros((h_c + 4, w_c + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        filled_padded = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
        mask_crop = filled_padded[1:-1, 1:-1]

        # [7] Tìm contours trên vùng Crop & chuyển đổi tọa độ về ảnh gốc (Chỉ chạy findContours 1 lần!)
        h_full, w_full = roi.shape[:2]
        raw_blobs = []

        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0 and area / hull_area >= SOLIDITY_MIN:
                chosen_c = hull
                chosen_area = hull_area
            else:
                chosen_c = c
                chosen_area = area

            if chosen_area < MIN_AREA or chosen_area > h_full * w_full * 0.85:
                continue

            # Dịch chuyển tọa độ contour từ vùng crop về ROI đầy đủ
            chosen_c[:, 0, 0] += x1
            chosen_c[:, 0, 1] += y1

            x, y, bw, bh = cv2.boundingRect(chosen_c)
            raw_blobs.append((chosen_c, (x + bw / 2, y + bh / 2)))

        # [8] Tracking & Vẽ lên stable mask đầy đủ
        for t in self.tracks:
            t['matched'] = False
        new_tracks = []
        stable = np.zeros((h_full, w_full), dtype=np.uint8)

        for c, center in raw_blobs:
            best_t, best_d = None, MAX_MOVE
            for t in self.tracks:
                if t['matched']:
                    continue
                d = np.hypot(center[0] - t['center'][0], center[1] - t['center'][1])
                if d < best_d:
                    best_d, best_t = d, t
            if best_t is not None:
                best_t['matched'] = True
                best_t['center'] = center
                best_t['age'] += 1
                best_t['missed'] = 0
                new_tracks.append(best_t)
                if best_t['age'] >= N_CONFIRM:
                    cv2.drawContours(stable, [c], -1, 255, -1)
            else:
                new_tracks.append({'center': center, 'age': 1, 'missed': 0,
                                   'matched': True})

        for t in self.tracks:
            if not t['matched']:
                t['missed'] += 1
                if t['missed'] <= MAX_MISSED:
                    new_tracks.append(t)
        self.tracks = new_tracks

        return stable   # mask hoàn chỉnh, chưa tách


def extract_bboxes(stable_mask, roi):
    """[4] BƯỚC CUỐI: watershed trên stable mask hoàn chỉnh → bbox.
    Blob nhỏ hơn SPLIT_MIN_AREA giữ nguyên; chỉ blob to bất thường mới tách."""
    h, w = stable_mask.shape
    bboxes = []
    contours, _ = cv2.findContours(stable_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        if area < SPLIT_MIN_AREA:
            # Lấy trực tiếp bbox từ contour hiện tại, không cần vẽ lại và tìm contour lần 2
            x, y, bw, bh = cv2.boundingRect(c)
            bboxes.append([max(0, x-PADDING), max(0, y-PADDING),
                           min(w, x+bw+PADDING), min(h, y+bh+PADDING)])
        else:
            # Tối ưu hóa: Kiểm tra độ lồi (solidity). Nếu lồi (như hộp đơn lẻ) thì không cần chạy Watershed
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            if solidity >= 0.92:
                # Vật thể đơn lẻ dạng lồi -> Bỏ qua hoàn toàn Watershed để tiết kiệm CPU
                x, y, bw, bh = cv2.boundingRect(c)
                bboxes.append([max(0, x-PADDING), max(0, y-PADDING),
                               min(w, x+bw+PADDING), min(h, y+bh+PADDING)])
            else:
                # Nghi ngờ nhiều vật thể chạm nhau -> Chỉ chạy Watershed trên vùng ảnh crop nhỏ của riêng blob này
                x, y, bw, bh = cv2.boundingRect(c)

                # Tạo mask nhỏ vừa khít với blob
                blob_mask_crop = np.zeros((bh, bw), dtype=np.uint8)
                c_shifted = c.copy()
                c_shifted[:, 0, 0] -= x
                c_shifted[:, 0, 1] -= y
                cv2.drawContours(blob_mask_crop, [c_shifted], -1, 255, -1)

                roi_crop = roi[y:y+bh, x:x+bw]

                # Chạy watershed trên mask nhỏ
                subs_crop = watershed_split(blob_mask_crop, roi_crop)

                for sub_crop in subs_crop:
                    sub_contours, _ = cv2.findContours(sub_crop, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    for sc in sub_contours:
                        if cv2.contourArea(sc) < MIN_AREA:
                            continue
                        x_s, y_s, bw_s, bh_s = cv2.boundingRect(sc)
                        abs_x = x + x_s
                        abs_y = y + y_s
                        bboxes.append([max(0, abs_x-PADDING), max(0, abs_y-PADDING),
                                       min(w, abs_x+bw_s+PADDING), min(h, abs_y+bh_s+PADDING)])
    return bboxes


if __name__ == "__main__":
    camera = HikrobotCamera(model_hint=CAMERA_MODEL_HINT)
    if not camera.open():
        print("Failed to open Hikrobot camera.")
        sys.exit(1)
    if not camera.start():
        camera.close()
        sys.exit(1)

    # model = SAM(PATH_MOBILE_SAM)
    # model = FastSAM(PAHT_FASTSAM)

    bg = learn_background_verified(camera)
    detector = Detector(bg.shape[:2], bg)

    try:
        while True:
            img = get_resized_frame(camera)
            if img is None:
                continue

            roi = get_roi(img)
            start_cr_mask = time.perf_counter()
            # Bước 1-3: tạo stable mask hoàn chỉnh
            stable_mask = detector.build_mask(roi)

            # Bước 4 (cuối): watershed trên mask hoàn chỉnh → bbox
            bboxes = extract_bboxes(stable_mask, roi)
            end_cr_mask = time.perf_counter()
            # print(f"Time creat mask: {(end_cr_mask - start_cr_mask):.3f}")
            # print(len(bboxes))

            # Bbox -> MobileSAM segment (nguyên bản) — chèn ở đây khi bật model:
            # if bboxes:
            #     results = model(roi, bboxes=bboxes, verbose=False)
            #     masks = results[0].masks

            disp = img.copy()

            # [Vẽ] Đường viền ôm SÁT vật (contour của mask) và/hoặc bbox chữ nhật
            if VE_CONTOUR:
                mask_draw = stable_mask
                if CONTOUR_SMOOTH >= 3:
                    ksz = CONTOUR_SMOOTH | 1   # ép về số lẻ
                    mask_draw = cv2.GaussianBlur(stable_mask, (ksz, ksz), 0)
                    _, mask_draw = cv2.threshold(mask_draw, 127, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(mask_draw, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) < MIN_AREA:
                        continue
                    if CONTOUR_SMOOTH >= 3:
                        c = smooth_contour(c, CONTOUR_SMOOTH)
                    if APPROX_CONTOUR > 0:
                        eps = APPROX_CONTOUR * cv2.arcLength(c, True)
                        c = cv2.approxPolyDP(c, eps, True)
                    c_shift = c + [roi_x, roi_y]
                    cv2.drawContours(disp, [c_shift], -1, (0, 0, 255), 2)

            if VE_BBOX:
                for b in bboxes:
                    cv2.rectangle(disp, (b[0]+roi_x, b[1]+roi_y), (b[2]+roi_x, b[3]+roi_y),
                                  (0, 255, 255), 2)

            cv2.putText(disp, f"Objects: {len(bboxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            show_fit("Detection", disp)
            show_fit("Mask", stable_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                bg = learn_background_verified(camera)
                detector = Detector(bg.shape[:2], bg)
    finally:
        camera.close()
        cv2.destroyAllWindows()