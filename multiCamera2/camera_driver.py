# -*- coding: utf-8 -*-
"""
DRIVER CAMERA CÔNG NGHIỆP HIKROBOT GIGE / USB (THREAD-SAFE & GIGE OPTIMIZED)
"""

import os
import sys
import time
import threading
from ctypes import *

import cv2
import numpy as np

from config import (
    TARGET_FPS, GEV_PACKET_SIZE, GEV_PACKET_DELAY,
    IMAGE_NODE_NUM, MAX_EXPOSURE_TIME,
    SPECIFIC_MASTER_IP, SPECIFIC_MASTER_SERIAL
)

# ============================================================
# CẤU HÌNH SDK MVS HIKROBOT
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
    os.path.join(_THIS_DIR, "..", "MvImport"),
    os.path.join(_THIS_DIR, "..", "MachNap", "datasetBegin", "MvImport"),
    os.path.join(_THIS_DIR, "..", "MachNap", "MvImport"),
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
    print(f"[Driver Warning] Chưa thể import SDK MVS Hikrobot: {e}")


def get_device_info_dict(dev_info):
    """Trích xuất Model, Serial, IP và phân loại Color/Mono từ MV_CC_DEVICE_INFO."""
    info = {
        "model": "Unknown",
        "serial": "Unknown",
        "ip": "Unknown",
        "type": "GigE",
        "is_color": False
    }
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

        # Phân loại Color vs Mono theo tên model (Hikrobot GC/UC/C, Sentech C...)
        model_upper = info["model"].upper()
        if any(c_tag in model_upper for c_tag in ["-10GC", "-20GC", "-10UC", "-20UC", "-13GC", "-13UC", "GC", "UC", "COLOR"]):
            info["is_color"] = True
        elif model_upper.endswith("C"):
            info["is_color"] = True
        else:
            info["is_color"] = False
    except Exception:
        pass
    return info


class IndustrialCamera:
    """Quản lý kết nối, tối ưu mạng GigE, giải mã màu BGR và thu nhận ảnh liên tục."""

    def __init__(self, cam_id, dev_info=None, name=None, is_master=False, target_fps=TARGET_FPS):
        self.cam_id = cam_id
        self.dev_info = dev_info
        self.name = name if name else f"cam{cam_id + 1}"
        self.is_master = is_master
        self.target_fps = target_fps
        self.cam = MvCamera()
        
        self.cam_info = get_device_info_dict(dev_info) if dev_info else {}
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
                err_msg += " (Thiết bị bận hoặc mất kết nối mạng)"
            elif ret in (0x80000215, 0x80000206):
                err_msg += f" (LỆCH DẢI MẠNG: Camera đang ở IP {self.cam_info['ip']}, khác dải với card mạng máy tính)"
            print(f"[{self.name}] Lỗi MV_CC_OpenDevice: {err_msg}")
            self.cam.MV_CC_DestroyHandle()
            return False

        # Tối ưu hóa GigE Vision
        if self.dev_info.nTLayerType == MV_GIGE_DEVICE:
            try:
                opt_packet_size = self.cam.MV_CC_GetOptimalPacketSize()
                if opt_packet_size > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", opt_packet_size)
                else:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", GEV_PACKET_SIZE)
            except Exception:
                try:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", GEV_PACKET_SIZE)
                except Exception:
                    pass

            is_hikrobot = any(k in self.cam_info["model"].upper() for k in ["MV-", "KC", "HIK"])
            if is_hikrobot:
                try:
                    self.cam.MV_CC_SetIntValue("GevSCPD", GEV_PACKET_DELAY)
                    self.cam.MV_CC_SetBoolValue("GevSCPHostResend", True)
                except Exception:
                    pass
            else:
                try:
                    self.cam.MV_CC_SetIntValue("GevSCPD", 400)
                except Exception:
                    pass

        try:
            self.cam.MV_CC_SetImageNodeNum(IMAGE_NODE_NUM)
        except Exception:
            pass

        # Khóa FPS chuẩn cho mọi camera
        if self.target_fps is not None and self.target_fps > 0:
            try:
                self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))
            except Exception:
                try:
                    self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))
                except Exception:
                    pass

        # Kiểm tra Exposure Time để không bị bóp FPS
        try:
            stFloat = MVCC_FLOATVALUE()
            memset(byref(stFloat), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloat)
            max_allowed_exp = (1.0 / self.target_fps) * 1000000.0 * 0.80
            if ret == 0 and stFloat.fCurValue > max_allowed_exp:
                self.cam.MV_CC_SetFloatValue("ExposureTime", max_allowed_exp)
        except Exception:
            pass

        # Tắt Trigger Mode & Chuyển Continuous Mode
        try:
            self.cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        except Exception:
            try:
                self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            except Exception:
                pass

        try:
            self.cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
        except Exception:
            pass

        # Cân bằng trắng tự động cho camera Màu
        if self.cam_info.get("is_color", False):
            try:
                self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Continuous")
            except Exception:
                pass

        role_tag = "MASTER (COLOR)" if self.is_master else ("SLAVE (COLOR)" if self.cam_info.get("is_color") else "SLAVE (MONO)")
        print(f"[{self.name.upper()}: {role_tag}] Đã mở: {self.cam_info['model']} | IP: {self.cam_info['ip']} | Khóa FPS: {self.target_fps:.1f}")
        return True

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[{self.name}] Lỗi MV_CC_StartGrabbing: 0x{ret:08x}")
            return False
        
        # Kích hoạt AcquisitionStart cho camera hãng thứ ba (Sentech)
        try:
            self.cam.MV_CC_SetCommandValue("AcquisitionStart")
        except Exception:
            pass

        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, name=f"CamWorker-{self.name}", daemon=True)
        self.thread.start()
        return True
        
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

        # 3. Bayer & RGB -> Dùng ConvertPixelTypeEx của SDK chuẩn màu tuyệt đối
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
        """Đọc frame mới nhất (Thread-safe). Trả về numpy array BGR hoặc None."""
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_frame_with_info(self):
        """Lấy frame, frame_id, realtime FPS."""
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


def discover_cameras():
    """Liệt kê toàn bộ camera có trên mạng."""
    if not HIK_SDK_OK:
        return []
    MvCamera.MV_CC_Initialize()
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list)
    found = []
    if ret == 0:
        for i in range(dev_list.nDeviceNum):
            dev = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            info = get_device_info_dict(dev)
            info["index"] = i
            found.append(info)
    return found


def build_all_cameras(max_cameras=4):
    """Tự động quét, phân loại Color/Master và khởi tạo danh sách camera."""
    if not HIK_SDK_OK:
        return {}
    MvCamera.MV_CC_Initialize()
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list)
    if ret != 0 or dev_list.nDeviceNum == 0:
        return {}

    count = min(dev_list.nDeviceNum, max_cameras)
    
    # Lấy thông tin sơ bộ
    device_entries = []
    for i in range(count):
        dev = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        info = get_device_info_dict(dev)
        device_entries.append((dev, info))

    # Tìm index camera Color / Master
    master_idx = None
    if SPECIFIC_MASTER_IP != "" or SPECIFIC_MASTER_SERIAL != "":
        for idx, (_, info) in enumerate(device_entries):
            if (SPECIFIC_MASTER_IP and info["ip"] == SPECIFIC_MASTER_IP) or \
               (SPECIFIC_MASTER_SERIAL and info["serial"] == SPECIFIC_MASTER_SERIAL):
                master_idx = idx
                break

    if master_idx is None:
        for idx, (_, info) in enumerate(device_entries):
            if info.get("is_color", False):
                master_idx = idx
                break

    if master_idx is None:
        master_idx = 0

    # Đưa Master Camera lên đầu (cam1)
    sorted_entries = [device_entries[master_idx]] + [device_entries[i] for i in range(len(device_entries)) if i != master_idx]

    cameras = {}
    for i, (dev, info) in enumerate(sorted_entries):
        cam_name = f"cam{i + 1}"
        is_master = (i == 0)
        cameras[cam_name] = IndustrialCamera(
            cam_id=i,
            dev_info=dev,
            name=cam_name,
            is_master=is_master,
            target_fps=TARGET_FPS
        )
    return cameras
