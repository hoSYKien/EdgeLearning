# -*- coding: utf-8 -*-
"""
================================================================================
                      HỆ THỐNG ĐỒNG BỘ ĐA CAMERA 2.0 (MULTICAMERA 2.0)
                         HỖ TRỢ 2 ĐẾN 4 CAMERA GIGE / USB CÔNG NGHIỆP
================================================================================
Các tính năng cốt lõi:
  1. Driver đa camera tối ưu (GigE Network Optimizer):
     - Khóa cứng 10.0 FPS cho mỗi camera, tổng băng thông < 1Gbps, không bao giờ xước ảnh.
     - Packet Size = 1500, Packet Delay (GevSCPD) = 1200 ticks, Host Resend = True.
     - Giải mã màu Bayer/Mono qua SDK Hikrobot MV_CC_ConvertPixelTypeEx -> BGR chuẩn 100%.
     - Tự động bật Cân bằng trắng (Auto White Balance).

  2. Đồng bộ ID đối tượng xuyên suốt 2 - 4 Camera (Multi-Camera ID Coordinator):
     - Chuyển đổi tọa độ ảnh (cx, cy) của mọi camera sang hệ tọa độ phẳng mặt bàn (X, Y) cm qua Ma trận Homography (H).
     - Gộp các phát hiện từ các góc nhìn khác nhau thành DUY NHẤT 1 VẬT THỂ (cùng 1 ID).
     - Chia sẻ Barcode chéo giữa các camera: Chỉ cần 1 camera đọc được barcode, TẤT CẢ các camera khác đang nhìn vật đó đều tự động hiển thị cùng ID và cùng Barcode!
     - Tracking vị trí theo thời gian giữ nguyên ID khi vật di chuyển trên bàn.

  3. Giao diện trực quan (Thread-safe OpenCV GUI):
     - Tự động ghép cửa sổ dạng Grid (1x2 cho 2 camera, 2x2 cho 4 camera).
     - Hiển thị Bounding Box, ID đồng bộ, Barcode, Tọa độ thực (X, Y) và Live FPS.
     - Phím tắt:
       * 'q' hoặc ESC : Thoát an toàn
       * 's'          : Chụp lưu ảnh đồng thời tất cả camera vào thư mục captures_multicam2/
       * 'r'          : Học lại nền (Re-learn background)
       * 'c'          : Hiệu chuẩn Homography (Calibrate)
       * 'g'          : Chuyển đổi giữa chế độ Grid View và Cửa sổ riêng
================================================================================
"""

import os
import sys
import json
import time
import threading
from ctypes import *

# Cấu hình UTF-8 cho Windows Console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

# ============================================================
# 1. CẤU HÌNH HỆ THỐNG
# ============================================================
TARGET_FPS = 10.0            # FPS mỗi camera (10.0 FPS đảm bảo 2-4 camera không quá tải switch 1Gbps)
GEV_PACKET_SIZE = 1500       # Kích thước MTU chuẩn Ethernet
GEV_PACKET_DELAY = 1200      # Độ trễ giữa các packet (ticks) chống nghẽn Switch
IMAGE_NODE_NUM = 15          # Buffer frame trong RAM máy tính

MERGE_DIST_CM = 8.0          # Khoảng cách tối đa trên mặt bàn (cm) để gộp phát hiện từ nhiều camera thành 1 vật
TRACK_DIST_CM = 15.0         # Khoảng cách tối đa (cm) giữa 2 frame liên tiếp để giữ nguyên ID
FORGET_AFTER_SEC = 1.5       # Thời gian (giây) xóa vật thể nếu tất cả camera mất dấu

HOMOGRAPHY_FILE = "homographies_multicam2.json"
CAPTURES_DIR = "captures_multicam2"

# Mốc tọa độ vật lý mẫu trên mặt bàn (cm) dùng khi Calibrate
# Hình chữ nhật 25cm (ngang) x 20cm (dọc)
CALIB_TABLE_POINTS = [
    (0.0, 0.0),     # 1: Trên - Trái (Gốc tọa độ O)
    (25.0, 0.0),    # 2: Trên - Phải
    (25.0, 20.0),   # 3: Dưới - Phải
    (0.0, 20.0),    # 4: Dưới - Trái
]

# ============================================================
# 2. KHỞI TẠO SDK HIKROBOT (MVS) & BARCODE ENGINES
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_HIK_DLL = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(_HIK_DLL):
    try:
        os.add_dll_directory(_HIK_DLL)
    except Exception:
        pass

_SDK_PATHS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "MvImport"),
    os.path.join(_THIS_DIR, "multiCamera", "MvImport"),
    os.path.join(_THIS_DIR, "MachNap", "datasetBegin", "MvImport"),
    os.path.join(_THIS_DIR, "MachNap", "MvImport"),
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
]
for p in _SDK_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from PixelType_header import *
    HIK_SDK_OK = True
except Exception as e:
    HIK_SDK_OK = False
    print(f"[Cảnh báo] Chưa thể import SDK Hikrobot: {e}")

# Nạp engine giải mã Barcode
_HAS_ZXING = False
try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:
    pass

_HAS_PYZBAR = False
try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    pass


# ============================================================
# 3. DRIVER CAMERA CÔNG NGHIỆP GIGE / USB (THREAD AN TOÀN)
# ============================================================
def get_camera_device_info(dev_info):
    """Trích xuất Model, Serial, IP từ MV_CC_DEVICE_INFO."""
    info = {"model": "Unknown", "serial": "Unknown", "ip": "Unknown", "type": "GigE"}
    try:
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            gige = dev_info.SpecialInfo.stGigEInfo
            info["model"] = "".join(chr(c) for c in gige.chModelName if c != 0).strip()
            info["serial"] = "".join(chr(c) for c in gige.chSerialNumber if c != 0).strip()
            nip1 = ((gige.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((gige.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((gige.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (gige.nCurrentIp & 0x000000ff)
            info["ip"] = f"{nip1}.{nip2}.{nip3}.{nip4}"
        elif dev_info.nTLayerType == MV_USB_DEVICE:
            usb = dev_info.SpecialInfo.stUsb3VInfo
            info["model"] = "".join(chr(c) for c in usb.chModelName if c != 0).strip()
            info["serial"] = "".join(chr(c) for c in usb.chSerialNumber if c != 0).strip()
            info["ip"] = "USB"
            info["type"] = "USB"
    except Exception:
        pass
    return info


class IndustrialCamera:
    """Quản lý kết nối, cấu hình mạng và thu nhận ảnh từ camera."""

    def __init__(self, cam_id, dev_info=None, name=None, target_fps=TARGET_FPS):
        self.cam_id = cam_id
        self.dev_info = dev_info
        self.name = name if name else f"cam{cam_id + 1}"
        self.target_fps = target_fps
        self.cam = MvCamera()
        
        self.cam_info = get_camera_device_info(dev_info) if dev_info else {}
        self.is_running = False
        self.thread = None
        self._lock = threading.Lock()
        
        self.latest_frame = None
        self.latest_frame_id = 0
        self.fps = 0.0
        self._fps_count = 0
        self._fps_time = time.time()
        
        self._convert_buf = None
        self._convert_buf_size = 0

    def open(self):
        if self.dev_info is None:
            return False
        
        ret = self.cam.MV_CC_CreateHandle(self.dev_info)
        if ret != 0:
            print(f"[{self.name}] Lỗi MV_CC_CreateHandle: 0x{ret:08x}")
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            err_msg = f"0x{ret:08x}"
            if ret == 0x80000203:
                err_msg += " (Camera đang bị MVS.exe hoặc tiến trình khác chiếm)"
            elif ret == 0x80000204:
                err_msg += " (Thiết bị bận hoặc mất mạng)"
            print(f"[{self.name}] Lỗi MV_CC_OpenDevice: {err_msg}")
            self.cam.MV_CC_DestroyHandle()
            return False

        # Tối ưu hóa GigE Vision
        if self.dev_info.nTLayerType == MV_GIGE_DEVICE:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", GEV_PACKET_SIZE)
            self.cam.MV_CC_SetIntValue("GevSCPD", GEV_PACKET_DELAY)
            try:
                self.cam.MV_CC_SetBoolValue("GevSCPHostResend", True)
            except Exception:
                pass

        self.cam.MV_CC_SetImageNodeNum(IMAGE_NODE_NUM)

        # Cài đặt FPS
        if self.target_fps is not None and self.target_fps > 0:
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))

        # Kiểm tra Exposure Time để không bị bóp tụt FPS
        try:
            stFloat = MVCC_FLOATVALUE()
            memset(byref(stFloat), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloat)
            if ret == 0 and stFloat.fCurValue > 80000.0:
                self.cam.MV_CC_SetFloatValue("ExposureTime", 50000.0)
        except Exception:
            pass

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # Tự động cân bằng trắng cho camera màu
        try:
            self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Continuous")
        except Exception:
            pass

        print(f"[{self.name}] Đã kết nối thành công: {self.cam_info['model']} (IP: {self.cam_info['ip']}, SN: {self.cam_info['serial']})")
        return True

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[{self.name}] Lỗi MV_CC_StartGrabbing: 0x{ret:08x}")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, name=f"CamWorker-{self.name}", daemon=True)
        self.thread.start()
        return True

    def _convert_frame_to_bgr(self, stFrame):
        w = stFrame.stFrameInfo.nWidth
        h = stFrame.stFrameInfo.nHeight
        frame_len = stFrame.stFrameInfo.nFrameLen
        pixel_type = stFrame.stFrameInfo.enPixelType

        buf_addr = cast(stFrame.pBufAddr, c_void_p).value
        if not buf_addr or frame_len == 0 or w == 0 or h == 0:
            return None

        # 1. Ảnh Mono8
        if pixel_type == PixelType_Gvsp_Mono8:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. Ảnh BGR8
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3)).copy()

        # 3. Tất cả định dạng Bayer / RGB / YUV -> Dùng MV_CC_ConvertPixelTypeEx của SDK chuẩn màu tuyệt đối
        else:
            dst_len = w * h * 3
            if self._convert_buf is None or self._convert_buf_size != dst_len:
                self._convert_buf = (c_ubyte * dst_len)()
                self._convert_buf_size = dst_len

            c_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(c_param), 0, sizeof(c_param))
            c_param.nWidth = w
            c_param.nHeight = h
            c_param.enSrcPixelType = pixel_type
            c_param.pSrcData = stFrame.pBufAddr
            c_param.nSrcDataLen = frame_len
            c_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            c_param.pDstBuffer = self._convert_buf
            c_param.nDstBufferSize = dst_len

            ret = self.cam.MV_CC_ConvertPixelTypeEx(c_param)
            if ret == 0:
                return np.frombuffer(self._convert_buf, dtype=np.uint8, count=dst_len).reshape((h, w, 3)).copy()
            else:
                raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                if pixel_type == PixelType_Gvsp_BayerGB8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_GB2BGR)
                elif pixel_type == PixelType_Gvsp_BayerRG8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_RG2BGR)
                elif pixel_type == PixelType_Gvsp_RGB8_Packed:
                    return cv2.cvtColor(raw_arr.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
                return None

    def _grab_loop(self):
        stFrame = MV_FRAME_OUT()
        memset(byref(stFrame), 0, sizeof(stFrame))

        while self.is_running:
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == 0:
                try:
                    bgr_img = self._convert_frame_to_bgr(stFrame)
                    if bgr_img is not None:
                        with self._lock:
                            self.latest_frame = bgr_img
                            self.latest_frame_id += 1
                            self._fps_count += 1

                        now = time.time()
                        dt = now - self._fps_time
                        if dt >= 1.0:
                            self.fps = self._fps_count / dt
                            self._fps_count = 0
                            self._fps_time = now
                except Exception as e:
                    print(f"[{self.name}] Lỗi decode: {e}")
                finally:
                    self.cam.MV_CC_FreeImageBuffer(stFrame)
            else:
                time.sleep(0.005)

    def read(self):
        """Lấy frame BGR mới nhất (Thread-safe)."""
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_frame_with_info(self):
        """Lấy frame, frame_id, FPS."""
        with self._lock:
            if self.latest_frame is None:
                return None, 0, 0.0
            return self.latest_frame.copy(), self.latest_frame_id, self.fps

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        print(f"[{self.name}] Đã đóng thiết bị.")


def auto_discover_and_create_cameras(max_cameras=4):
    """Tự động quét và khởi tạo danh sách tất cả camera có trên mạng."""
    if not HIK_SDK_OK:
        return {}
    MvCamera.MV_CC_Initialize()
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list)
    if ret != 0 or dev_list.nDeviceNum == 0:
        return {}

    count = min(dev_list.nDeviceNum, max_cameras)
    cameras = {}
    for i in range(count):
        dev = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam_name = f"cam{i + 1}"
        cam_worker = IndustrialCamera(cam_id=i, dev_info=dev, name=cam_name, target_fps=TARGET_FPS)
        cameras[cam_name] = cam_worker
    return cameras


# ============================================================
# 4. THUẬT TOÁN PHÁT HIỆN VẬT & ĐỌC BARCODE
# ============================================================
class ObjectDetector:
    """Trừ nền học động để phát hiện vị trí vật và đọc mã vạch."""

    def __init__(self, resize_div=4, thresh=30, min_area_ratio=0.002, max_area_ratio=0.9):
        self.resize_div = resize_div
        self.thresh = thresh
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.bg_model = None

    def learn_background(self, cam, num_frames=30, timeout=10.0):
        """Học nền từ camera (khung hình không chứa sản phẩm)."""
        frames = []
        t0 = time.time()
        while len(frames) < num_frames:
            f = cam.read()
            if f is None:
                if time.time() - t0 > timeout:
                    print(f"[{cam.name}] Timeout khi học nền!")
                    return False
                time.sleep(0.02)
                continue
            h, w = f.shape[:2]
            small = cv2.resize(f, (w // self.resize_div, h // self.resize_div))
            blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
            frames.append(blur)
            time.sleep(0.03)

        self.bg_model = np.mean(frames, axis=0)
        print(f"[{cam.name}] Đã học nền thành công ({len(frames)} frames).")
        return True

    def detect_objects(self, full_bgr):
        """Trả về list ( (cx, cy), bbox(x1,y1,x2,y2) )."""
        if self.bg_model is None or full_bgr is None:
            return []
        h, w = full_bgr.shape[:2]
        small = cv2.resize(full_bgr, (w // self.resize_div, h // self.resize_div))
        blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
        diff = cv2.absdiff(blur, self.bg_model)
        b, g, r = cv2.split(diff)
        d = cv2.max(cv2.max(b, g), r)
        mask = (d > self.thresh).astype(np.uint8) * 255
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        area_img = small.shape[0] * small.shape[1]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area_ratio * area_img <= a <= self.max_area_ratio * area_img:
                x, y, ww, hh = cv2.boundingRect(c)
                x1, y1 = x * self.resize_div, y * self.resize_div
                x2, y2 = (x + ww) * self.resize_div, (y + hh) * self.resize_div
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                objects.append(((cx, cy), (x1, y1, x2, y2)))
        return objects

    @staticmethod
    def decode_barcode(crop_bgr):
        """Giải mã barcode từ vùng crop của vật."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        
        # 1. Thử zxing-cpp
        if _HAS_ZXING:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            for im in (gray, cv2.createCLAHE(3.0, (8, 8)).apply(gray)):
                try:
                    res = zxingcpp.read_barcodes(im)
                    for r in res:
                        if r.text:
                            return r.text.strip()
                except Exception:
                    pass

        # 2. Thử pyzbar
        if _HAS_PYZBAR:
            try:
                res = _pyzbar_decode(crop_bgr)
                for r in res:
                    if r.data:
                        return r.data.decode('utf-8', errors='ignore').strip()
            except Exception:
                pass

        return None


# ============================================================
# 5. BỘ ĐIỀU PHỐI ĐỒNG BỘ ID ĐA CAMERA (MULTI-CAMERA COORDINATOR)
# ============================================================
def image_to_table(H, x, y):
    """Chuyển đổi điểm ảnh (x, y) sang tọa độ mặt bàn (X, Y) qua ma trận H (3x3)."""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


class SingleDetection:
    def __init__(self, cam_name, img_xy, table_xy, barcode, bbox):
        self.cam_name = cam_name
        self.img_xy = img_xy
        self.table_xy = table_xy
        self.barcode = barcode
        self.bbox = bbox


class TrackedObject:
    """Một vật thể duy nhất được đồng bộ trên toàn bộ 2 - 4 camera."""

    def __init__(self, obj_id, table_xy):
        self.id = obj_id
        self.table_xy = table_xy
        self.barcode = None
        self.seen_by = set()
        self.per_cam_bbox = {}
        self.last_seen = time.time()

    def update(self, detections):
        pts = [d.table_xy for d in detections if d.table_xy is not None]
        if pts:
            self.table_xy = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        
        self.seen_by = set(d.cam_name for d in detections)
        self.per_cam_bbox = {d.cam_name: d.bbox for d in detections if d.bbox}

        # CHIA SẺ BARCODE: Nếu có bất kỳ camera nào đọc được barcode -> cập nhật cho cả nhóm
        for d in detections:
            if d.barcode:
                if self.barcode is None:
                    self.barcode = d.barcode
        self.last_seen = time.time()


class MultiCameraCoordinator:
    """Quản lý gộp cụm không gian và duy trì ID đồng bộ cho 2 - 4 camera."""

    def __init__(self, homographies=None, merge_dist=MERGE_DIST_CM, track_dist=TRACK_DIST_CM, forget_after=FORGET_AFTER_SEC):
        self.H = homographies if homographies else {}
        self.merge_dist = merge_dist
        self.track_dist = track_dist
        self.forget_after = forget_after
        self.objects = {}
        self._next_id = 1

    def update(self, raw_cam_detections):
        """
        raw_cam_detections: dict { cam_name: [ ((cx, cy), barcode_or_None, bbox), ... ] }
        Trả về: list[TrackedObject]
        """
        all_dets = []
        for cam_name, det_list in raw_cam_detections.items():
            H = self.H.get(cam_name)
            for ((cx, cy), barcode, bbox) in det_list:
                table_xy = None
                if H is not None:
                    table_xy = image_to_table(H, cx, cy)
                
                # Nếu chưa calibrate Homography, dùng tạm toạ độ ảnh quy chuẩn
                if table_xy is None:
                    table_xy = (cx / 10.0, cy / 10.0)

                all_dets.append(SingleDetection(cam_name, (cx, cy), table_xy, barcode, bbox))

        # 1. Gộp cụm (Clustering) các phát hiện gần nhau trên mặt bàn thành 1 vật
        clusters = []
        for d in all_dets:
            placed = False
            for cl in clusters:
                cx = np.mean([e.table_xy[0] for e in cl])
                cy = np.mean([e.table_xy[1] for e in cl])
                if np.hypot(d.table_xy[0] - cx, d.table_xy[1] - cy) < self.merge_dist:
                    cl.append(d)
                    placed = True
                    break
            if not placed:
                clusters.append([d])

        # 2. Gán cụm vào ID đang theo dõi (Tracking qua các frame)
        now = time.time()
        used_ids = set()
        for cl in clusters:
            cx = float(np.mean([e.table_xy[0] for e in cl]))
            cy = float(np.mean([e.table_xy[1] for e in cl]))

            best_id, best_d = None, self.track_dist
            for oid, obj in self.objects.items():
                if oid in used_ids:
                    continue
                dd = np.hypot(obj.table_xy[0] - cx, obj.table_xy[1] - cy)
                if dd < best_d:
                    best_d, best_id = dd, oid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.objects[best_id] = TrackedObject(best_id, (cx, cy))

            self.objects[best_id].update(cl)
            used_ids.add(best_id)

        # 3. Xóa các vật thể đã ra khỏi khung nhìn quá lâu
        for oid in [o for o, ob in self.objects.items() if now - ob.last_seen > self.forget_after]:
            del self.objects[oid]

        return list(self.objects.values())


# ============================================================
# 6. HIỆU CHUẨN HOMOGRAPHY TƯƠNG TÁC (INTERACTIVE CALIBRATOR)
# ============================================================
def load_homographies(filepath=HOMOGRAPHY_FILE):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float64) for k, v in data.items()}
    except Exception as e:
        print(f"Lỗi đọc {filepath}: {e}")
        return {}


def save_homographies(H_dict, filepath=HOMOGRAPHY_FILE):
    data = {k: v.tolist() for k, v in H_dict.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Đã lưu homography vào: {filepath}")


def calibrate_camera_interactive(cam_name, get_frame_func):
    """Mở cửa sổ cho người dùng bấm 4 điểm mốc để tính ma trận Homography."""
    frame = get_frame_func()
    if frame is None:
        print(f"[{cam_name}] Không lấy được frame để calibrate.")
        return None

    img_points = []
    h_orig, w_orig = frame.shape[:2]
    view_w, view_h = 1280, 720
    scale = min(view_w / w_orig, view_h / h_orig)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(img_points) < len(CALIB_TABLE_POINTS):
                orig_x = x / scale
                orig_y = y / scale
                img_points.append((orig_x, orig_y))
                print(f"[{cam_name}] Điểm {len(img_points)}/{len(CALIB_TABLE_POINTS)}: Ảnh({orig_x:.1f}, {orig_y:.1f}) -> Bàn{CALIB_TABLE_POINTS[len(img_points)-1]}")

    win_name = f"CALIBRATE [{cam_name}] - Bam 4 diem theo thu tu: Tren-Trai -> Tren-Phai -> Duoi-Phai -> Duoi-Trai"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 60)
    print(f"BẮT ĐẦU CALIBRATE HOMOGRAPHY CHO [{cam_name}]:")
    print(f"  Hãy click chuột lần lượt 4 điểm mốc trên ảnh:")
    for idx, pt in enumerate(CALIB_TABLE_POINTS):
        print(f"    Điểm {idx+1}: Tọa độ mặt bàn = {pt} cm")
    print("  Phím điều khiển:")
    print("    - 's': Lưu ma trận H sau khi đã click đủ 4 điểm")
    print("    - 'u': Undo điểm click gần nhất")
    print("    - 'q' hoặc ESC: Bỏ qua camera này")
    print("=" * 60 + "\n")

    H_matrix = None
    while True:
        f = get_frame_func()
        if f is None:
            f = frame
        disp = cv2.resize(f, (int(w_orig * scale), int(h_orig * scale)))

        # Vẽ các điểm đã click
        for idx, (px, py) in enumerate(img_points):
            sx, sy = int(px * scale), int(py * scale)
            cv2.circle(disp, (sx, sy), 6, (0, 0, 255), -1)
            cv2.circle(disp, (sx, sy), 8, (0, 255, 255), 2)
            cv2.putText(disp, f"{idx+1}", (sx + 10, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Nối dây các điểm
        if len(img_points) > 1:
            for idx in range(len(img_points) - 1):
                p1 = (int(img_points[idx][0] * scale), int(img_points[idx][1] * scale))
                p2 = (int(img_points[idx+1][0] * scale), int(img_points[idx+1][1] * scale))
                cv2.line(disp, p1, p2, (255, 255, 0), 2)
            if len(img_points) == len(CALIB_TABLE_POINTS):
                p1 = (int(img_points[-1][0] * scale), int(img_points[-1][1] * scale))
                p2 = (int(img_points[0][0] * scale), int(img_points[0][1] * scale))
                cv2.line(disp, p1, p2, (255, 255, 0), 2)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') or key == ord('U'):
            if img_points:
                removed = img_points.pop()
                print(f"[{cam_name}] Đã Undo điểm: {removed}")
        elif key == ord('s') or key == ord('S'):
            if len(img_points) == len(CALIB_TABLE_POINTS):
                src_pts = np.array(img_points, dtype=np.float32)
                dst_pts = np.array(CALIB_TABLE_POINTS, dtype=np.float32)
                H, status = cv2.findHomography(src_pts, dst_pts)
                if H is not None:
                    H_matrix = H
                    print(f"[{cam_name}] Tính toán Homography thành công!")
                break
            else:
                print(f"[{cam_name}] Chưa click đủ {len(CALIB_TABLE_POINTS)} điểm (Hiện có: {len(img_points)})")
        elif key == ord('q') or key == 27:
            print(f"[{cam_name}] Đã hủy Calibrate.")
            break

    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass
    return H_matrix


# ============================================================
# 7. CHƯƠNG TRÌNH CHÍNH (MAIN THREAD PIPELINE)
# ============================================================
def main():
    print("=" * 70)
    print("      HỆ THỐNG ĐỒNG BỘ ĐA CAMERA 2.0 (2 ĐẾN 4 CAMERA GIGE)")
    print("=" * 70)

    # 1. Quét và tạo kết nối tới các camera
    print("Đang quét tìm các camera công nghiệp Hikrobot...")
    cams = auto_discover_and_create_cameras(max_cameras=4)
    if not cams:
        print("Lỗi: Không tìm thấy camera Hikrobot nào trên mạng.")
        return

    print(f"\nTìm thấy {len(cams)} camera:")
    for name, cam in cams.items():
        print(f"  [{name}] {cam.cam_info['model']} | IP: {cam.cam_info['ip']} | SN: {cam.cam_info['serial']}")

    # Mở và khởi động camera
    opened_cams = {}
    for name, cam in cams.items():
        if cam.open():
            if cam.start():
                opened_cams[name] = cam
            else:
                cam.stop()

    if not opened_cams:
        print("Không mở được camera nào. Đang thoát...")
        return
    cams = opened_cams

    # 2. Nạp hoặc kiểm tra Homography
    homographies = load_homographies(HOMOGRAPHY_FILE)
    missing_calib = [name for name in cams if name not in homographies]
    if missing_calib:
        print(f"\n[Thông báo] Các camera chưa có Homography: {missing_calib}")
        print("Nhấn phím 'c' trong giao diện bất kỳ lúc nào để Calibrate lấy tọa độ mặt bàn.")

    # 3. Khởi tạo Detector & Coordinator
    detectors = {name: ObjectDetector() for name in cams}
    coordinator = MultiCameraCoordinator(homographies=homographies, merge_dist=MERGE_DIST_CM, track_dist=TRACK_DIST_CM)

    # 4. Học nền
    print("\n" + "=" * 60)
    print("ĐANG HỌC NỀN CHO CÁC CAMERA (Vui lòng giữ khung nhìn TRỐNG)...")
    for name, cam in cams.items():
        detectors[name].learn_background(cam, num_frames=30)
    print("Học nền hoàn tất. Đang chuyển sang chế độ Tracking đồng bộ ID.")
    print("=" * 60)

    os.makedirs(CAPTURES_DIR, exist_ok=True)
    show_grid = True

    try:
        while True:
            raw_frames = {}
            per_cam_detections = {}

            # Thu thập frame và phát hiện đối tượng từ từng camera
            for name, cam in cams.items():
                f = cam.read()
                if f is None:
                    continue
                raw_frames[name] = f

                # Trừ nền -> Bounding Boxes
                obj_list = detectors[name].detect_objects(f)
                
                # Quét barcode trên từng crop của vật
                cam_dets = []
                for (cx, cy), (x1, y1, x2, y2) in obj_list:
                    crop = f[y1:y2, x1:x2]
                    barcode = detectors[name].decode_barcode(crop)
                    cam_dets.append(((cx, cy), barcode, (x1, y1, x2, y2)))
                per_cam_detections[name] = cam_dets

            # ĐỒNG BỘ ID & BARCODE QUA COORDINATOR
            tracked_objects = coordinator.update(per_cam_detections)

            # Vẽ kết quả lên từng frame
            display_canvases = []
            for cam_id_idx, (name, cam) in enumerate(cams.items()):
                f = raw_frames.get(name)
                disp_w, disp_h = 640, 480
                
                if f is not None:
                    disp = f.copy()
                    
                    # Vẽ tất cả các object đang được track lên frame này
                    for obj in tracked_objects:
                        if name in obj.per_cam_bbox:
                            x1, y1, x2, y2 = obj.per_cam_bbox[name]
                            
                            # Màu khung: Xanh lá nếu có Barcode, Vàng cam nếu chưa có Barcode
                            color = (0, 255, 0) if obj.barcode else (0, 200, 255)
                            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)

                            # Nhãn ID đồng bộ
                            label = f"ID: {obj.id}"
                            if obj.barcode:
                                label += f" | {obj.barcode}"
                            
                            # Tọa độ mặt bàn
                            pos_label = f"Pos: ({obj.table_xy[0]:.1f}, {obj.table_xy[1]:.1f})cm"

                            # Vẽ nền nhãn
                            cv2.rectangle(disp, (x1, max(0, y1 - 40)), (x1 + max(len(label), len(pos_label)) * 12, y1), (0, 0, 0), -1)
                            cv2.putText(disp, label, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                            cv2.putText(disp, pos_label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Scale cho màn hình
                    h, w = disp.shape[:2]
                    scale = min(disp_w / w, disp_h / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(disp, (nw, nh))

                    canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    xo = (disp_w - nw) // 2
                    yo = (disp_h - nh) // 2
                    canvas[yo:yo+nh, xo:xo+nw] = resized

                    # Header thông tin camera
                    header = f"[{name.upper()}] {cam.cam_info['model']} | FPS: {cam.fps:.1f} | Frame: {cam.latest_frame_id}"
                    cv2.rectangle(canvas, (0, 0), (disp_w, 28), (20, 20, 20), -1)
                    cv2.putText(canvas, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                    display_canvases.append((name, canvas, f))
                else:
                    blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    cv2.putText(blank, f"[{name}] Dang cho frame...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    display_canvases.append((name, blank, None))

            # HIỂN THỊ CỬA SỔ
            if show_grid and len(display_canvases) > 0:
                if len(display_canvases) == 1:
                    grid_view = display_canvases[0][1]
                elif len(display_canvases) == 2:
                    grid_view = np.hstack([display_canvases[0][1], display_canvases[1][1]])
                elif len(display_canvases) <= 4:
                    row1 = np.hstack([display_canvases[0][1], display_canvases[1][1]])
                    row2_items = []
                    for idx in range(2, 4):
                        if idx < len(display_canvases):
                            row2_items.append(display_canvases[idx][1])
                        else:
                            row2_items.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    row2 = np.hstack(row2_items)
                    grid_view = np.vstack([row1, row2])
                else:
                    grid_view = np.hstack([item[1] for item in display_canvases[:3]])

                cv2.imshow("MultiCamera 2.0 - Unified ID Synchronizer", grid_view)
            else:
                for name, canvas, _ in display_canvases:
                    cv2.imshow(f"Camera [{name}]", canvas)

            # Xử lý phím tắt
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('g') or key == ord('G'):
                show_grid = not show_grid
                if show_grid:
                    for name in cams:
                        try:
                            cv2.destroyWindow(f"Camera [{name}]")
                        except Exception:
                            pass
                else:
                    try:
                        cv2.destroyWindow("MultiCamera 2.0 - Unified ID Synchronizer")
                    except Exception:
                        pass
                print(f"-> Chuyển chế độ hiển thị: {'Grid View' if show_grid else 'Cửa sổ riêng'}")
            elif key == ord('s') or key == ord('S'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                saved_count = 0
                for name, _, raw in display_canvases:
                    if raw is not None:
                        fp = os.path.join(CAPTURES_DIR, f"{name}_{ts}.png")
                        cv2.imwrite(fp, raw)
                        saved_count += 1
                print(f"[Snapshot] Đã lưu {saved_count} ảnh vào {CAPTURES_DIR}")
            elif key == ord('r') or key == ord('R'):
                print("\n[Học lại nền] Đang học lại nền cho tất cả camera...")
                for name, cam in cams.items():
                    detectors[name].learn_background(cam, num_frames=20)
                print("[Học lại nền] Hoàn tất.")
            elif key == ord('c') or key == ord('C'):
                print("\n[Calibrate] Mở chế độ Calibrate Homography lần lượt cho các camera...")
                for name, cam in cams.items():
                    H = calibrate_camera_interactive(name, cam.read)
                    if H is not None:
                        homographies[name] = H
                save_homographies(homographies, HOMOGRAPHY_FILE)
                coordinator.H = homographies
                print("[Calibrate] Đã cập nhật ma trận Homography mới cho Coordinator.")

    except KeyboardInterrupt:
        print("\nNhận tín hiệu dừng...")
    finally:
        print("\nĐang dừng tất cả camera và giải phóng tài nguyên...")
        for cam in cams.values():
            try:
                cam.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass
        print("Hoàn tất dọn dẹp và thoát an toàn.")


if __name__ == "__main__":
    main()
