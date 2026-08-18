"""
LỚP CAMERA TRỪU TƯỢNG (đa hãng)
================================
Mọi camera đều expose CÙNG 1 interface:

    cam.open()  -> bool
    cam.start()
    cam.read()  -> np.ndarray BGR  (hoặc None)
    cam.stop()
    cam.name    -> str  (để log / gắn ID nguồn)

Nhờ vậy phần xử lý phía sau (crop + barcode + ghép ID) KHÔNG cần biết
frame đến từ Hikrobot hay KEA hay hãng nào.

Đã có sẵn:
  - HikrobotCamera : dùng SDK MVS (MvImport / MvCameraControl_class)
  - KeaCamera      : dùng SDK KEA (bạn nói đã có Python cho nó - IMVApi).
                     Nếu KEA thực chất cũng là chuẩn GigE Vision, có thể
                     mở bằng Harvester (genicam) thay cho SDK riêng.
  - FakeCamera     : phát ảnh giả để TEST logic mà không cần phần cứng.

>>> Bạn chỉ cần chỉnh 2 lớp thật cho khớp SDK máy bạn; interface giữ nguyên.
"""

import os
import sys
import threading
import time
import numpy as np
import cv2


# ============================================================
# THIẾT LẬP ĐƯỜNG DẪN SDK (giống live_detect.py)
# ============================================================
# Cho phép import MvCameraControl_class (Hikrobot) và IMVApi (KEA)
# dù chúng nằm trong thư mục con cạnh file này.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# DLL runtime của Hikrobot (nếu máy có cài MVS)
_HIK_DLL = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(_HIK_DLL):
    try:
        os.add_dll_directory(_HIK_DLL)
    except Exception:
        pass

# Thêm các thư mục có thể chứa SDK Python vào sys.path.
# >>> CHỈNH đường dẫn cho khớp máy bạn nếu MvImport / IMVApi nằm chỗ khác. <<<
_SDK_DIRS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "MvImport"),      # SDK Hikrobot (MvCameraControl_class)
    os.path.join(_THIS_DIR, "IMVApi"),        # SDK KEA (nếu để trong folder con)
    os.path.join(_THIS_DIR, "..", "MvImport"),
]
for _d in _SDK_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)


# ============================================================
# INTERFACE CHUNG
# ============================================================

class BaseCamera:
    name = "cam"

    def open(self): raise NotImplementedError
    def start(self): raise NotImplementedError
    def read(self): raise NotImplementedError
    def stop(self): raise NotImplementedError


# ============================================================
# HIKROBOT (MVS) - rút gọn từ live_detect của bạn
# ============================================================

# ============================================================
# HIKROBOT (MVS) - BÊ NGUYÊN logic đã chạy được từ live_detect.py
# ============================================================

class HikrobotCamera(BaseCamera):
    """
    Mở bằng SDK MVS. Dùng 'from MvCameraControl_class import *' giống hệt
    live_detect.py để không thiếu hằng số nào. Logic read() bê nguyên bản
    đã chạy được của bạn.
    """

    def __init__(self, name="HIK", model_hint="MV-CS200", serial=None):
        self.name = name
        self.model_hint = model_hint
        self.serial = serial       # nếu đặt -> chọn ĐÚNG camera theo serial
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = 0
        self.stFrameInfo = None
        self._convert_buf = None
        self.is_running = False
        self.current_frame = None
        self.thread = None
        self._lock = threading.Lock()

    def open(self):
        # wildcard import: nạp toàn bộ hằng số + lớp, y như live_detect.py
        import MvCameraControl_class as _mv
        globals().update({k: getattr(_mv, k) for k in dir(_mv)
                          if not k.startswith("__")})
        from ctypes import cast, POINTER, byref

        print("SDK Version:", hex(MvCamera.MV_CC_GetSDKVersion()))
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print(f"[{self.name}] không thấy camera Hikrobot (ret={ret})")
            return False
        print(f"[{self.name}] tìm thấy {deviceList.nDeviceNum} camera")

        def get_serial(info):
            if info.nTLayerType == MV_GIGE_DEVICE:
                return "".join(chr(c) for c in
                               info.SpecialInfo.stGigEInfo.chSerialNumber if c != 0)
            elif info.nTLayerType == MV_USB_DEVICE:
                return "".join(chr(c) for c in
                               info.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0)
            return ""

        def get_name(info):
            if info.nTLayerType == MV_GIGE_DEVICE:
                return "".join(chr(c) for c in
                               info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            elif info.nTLayerType == MV_USB_DEVICE:
                return "".join(chr(c) for c in
                               info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            return ""

        # Chọn camera: ưu tiên SERIAL, sau đó model_hint. Chỉ nhận camera
        # mà SDK Hikrobot thực sự điều khiển được (model chứa model_hint).
        target_index = None
        for i in range(deviceList.nDeviceNum):
            info = cast(deviceList.pDeviceInfo[i],
                        POINTER(MV_CC_DEVICE_INFO)).contents
            nm, sn = get_name(info), get_serial(info)
            print(f"  [{i}] {nm}  (SN={sn})")
            if self.serial and sn == self.serial:
                target_index = i
            elif target_index is None and self.model_hint and self.model_hint in nm:
                target_index = i

        if target_index is None:
            print(f"[{self.name}] KHÔNG tìm thấy camera khớp "
                  f"(serial={self.serial}, hint={self.model_hint})")
            return False

        stDeviceInfo = cast(deviceList.pDeviceInfo[target_index],
                            POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        r = self.cam.MV_CC_CreateHandle(stDeviceInfo)
        if r != 0:
            print(f"[{self.name}] tạo handle lỗi (ret=0x{r & 0xFFFFFFFF:X})")
            return False
        r = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if r != 0:
            print(f"[{self.name}] mở device lỗi (ret=0x{r & 0xFFFFFFFF:X})")
            print(f"        -> lỗi 0x80000203/0x8000000D thường là ĐANG BỊ CHIẾM "
                  f"(đóng MVS Viewer / tiến trình khác giữ camera).")
            self.cam.MV_CC_DestroyHandle(); return False

        if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if nPacketSize > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        return True

    def start(self):
        from ctypes import byref, memset, sizeof, c_ubyte
        if self.cam.MV_CC_StartGrabbing() != 0:
            print(f"[{self.name}] start grabbing lỗi"); return False
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self.nPayloadSize = stParam.nCurValue
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _read_once(self):
        """Bê nguyên logic read() của live_detect.py (đã chạy được)."""
        from ctypes import byref, memset, sizeof, c_ubyte
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

        # convert pixel format khác -> BGR8
        nDstSize = w * h * 3
        if self._convert_buf is None or len(self._convert_buf) != nDstSize:
            self._convert_buf = (c_ubyte * nDstSize)()
        cvt = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(cvt), 0, sizeof(cvt))
        cvt.nWidth = w; cvt.nHeight = h
        cvt.pSrcData = self.data_buf; cvt.nSrcDataLen = n
        cvt.enSrcPixelType = pt
        cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        cvt.pDstBuffer = self._convert_buf; cvt.nDstBufferSize = nDstSize
        if self.cam.MV_CC_ConvertPixelType(cvt) != 0:
            return None
        img = np.frombuffer(self._convert_buf, dtype=np.uint8, count=nDstSize)
        return img.reshape(h, w, 3).copy()

    def _loop(self):
        while self.is_running:
            img = self._read_once()
            if img is not None:
                with self._lock:
                    self.current_frame = img

    def read(self):
        with self._lock:
            return None if self.current_frame is None else self.current_frame.copy()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()


# ============================================================
# KEA (IMVApi) - theo file mẫu camera công nghiệp của bạn
# ============================================================

class KeaCamera(BaseCamera):
    """
    Mở bằng SDK KEA (IMVApi) - đúng luồng file mẫu CameraIndustrial bạn có.
    Nếu KEA hỗ trợ GigE Vision chuẩn, có thể thay bằng Harvester (genicam).
    """

    def __init__(self, name="KEA", index=0, model_hint="KEA", serial=None,
                 exposure=1000.0, gain=2.41, packet_size=1500):
        self.name = name
        self.index = index          # dùng khi không khớp được model/serial
        self.model_hint = model_hint  # tự tìm đúng con KEA trong danh sách
        self.serial = serial        # nếu đặt -> chọn chính xác theo serial
        self.exposure = exposure
        self.gain = gain
        # GevSCPSPacketSize: KEA mặc định 8044 (jumbo). NIC không bật jumbo
        # frame -> rớt hết gói ảnh -> GetFrame timeout. 1500 là an toàn.
        self.packet_size = packet_size
        self.cam = None
        self.is_running = False
        self.current_frame = None
        self.thread = None
        self._lock = threading.Lock()

    def open(self):
        from IMVApi import (MvCamera, IMV_DeviceList, IMV_EInterfaceType,
                            IMV_ECreateHandleMode)
        from ctypes import byref, c_void_p
        self.cam = MvCamera()
        dl = IMV_DeviceList()
        MvCamera.IMV_EnumDevices(dl, IMV_EInterfaceType.interfaceTypeAll)
        if dl.nDevNum == 0:
            print(f"[{self.name}] không thấy camera KEA")
            return False

        # IMVApi có thể liệt kê CHUNG cả camera hãng khác (vd Hikrobot). Phải
        # chọn ĐÚNG con KEA theo serial/model, không dùng index cứng kẻo trỏ
        # nhầm sang con đang bị SDK khác chiếm (Open lỗi -101/-102).
        def _s(v):
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            return str(v).strip("\x00")

        chosen = None
        print(f"[{self.name}] IMVApi thấy {dl.nDevNum} camera:")
        for i in range(dl.nDevNum):
            info = dl.pDevInfo[i]
            model = _s(getattr(info, "modelName", ""))
            vendor = _s(getattr(info, "vendorName", ""))
            sn = _s(getattr(info, "serialNumber", ""))
            print(f"    [{i}] {vendor} {model} (SN={sn})")
            if self.serial and sn == self.serial:
                chosen = i
                break
            if (chosen is None and self.model_hint
                    and self.model_hint.upper() in model.upper()):
                chosen = i
        if chosen is None:
            chosen = self.index
            print(f"[{self.name}] không khớp model/serial -> dùng index {chosen}")
        self.index = chosen
        print(f"[{self.name}] chọn index {self.index}")

        r = self.cam.IMV_CreateHandle(
            IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(self.index)))
        if r != 0:
            print(f"[{self.name}] CreateHandle lỗi ret={r}")
            return False
        r = self.cam.IMV_Open()
        if r != 0:
            print(f"[{self.name}] Open lỗi ret={r} "
                  f"(-101/-102 thường là camera đang bị chiếm / trỏ nhầm con)")
            return False

        # ÉP packet size TRƯỚC khi grab (đây là chìa khoá để ra được frame).
        try:
            rp = self.cam.IMV_SetIntFeatureValue(
                "GevSCPSPacketSize", self.packet_size)
            if rp != 0:
                print(f"[{self.name}] set GevSCPSPacketSize={self.packet_size} "
                      f"ret={rp} (bỏ qua nếu không phải GigE)")
        except Exception as e:
            print(f"[{self.name}] set packet size EXC: {e}")
        try:
            self.cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
            self.cam.IMV_SetEnumFeatureSymbol("BalanceWhiteAuto", "Once")
            self.cam.IMV_SetDoubleFeatureValue("ExposureTime", self.exposure)
            self.cam.IMV_SetDoubleFeatureValue("GainRaw", self.gain)
        except Exception:
            pass
        return True

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        from IMVApi import (IMV_Frame, IMV_PixelConvertParam, IMV_EPixelType,
                            IMV_EBayerDemosaic, IMV_OK)
        from ctypes import (byref, memset, memmove, sizeof, c_ubyte, c_buffer)
        rg = self.cam.IMV_StartGrabbing()
        if rg != 0:
            print(f"[{self.name}] StartGrabbing lỗi ret={rg} -> không ra ảnh.")
            return
        while self.is_running:
            frame = IMV_Frame()
            cvt = IMV_PixelConvertParam()
            if self.cam.IMV_GetFrame(frame, 1000) != IMV_OK:
                continue
            if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
                nDst = frame.frameInfo.width * frame.frameInfo.height
            else:
                nDst = frame.frameInfo.width * frame.frameInfo.height * 3
            pDst = (c_ubyte * nDst)()
            memset(byref(cvt), 0, sizeof(cvt))
            cvt.nWidth = frame.frameInfo.width
            cvt.nHeight = frame.frameInfo.height
            cvt.ePixelFormat = frame.frameInfo.pixelFormat
            cvt.pSrcData = frame.pData
            cvt.nSrcDataLen = frame.frameInfo.size
            cvt.nPaddingX = frame.frameInfo.paddingX
            cvt.nPaddingY = frame.frameInfo.paddingY
            cvt.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
            cvt.pDstBuf = pDst
            cvt.nDstBufSize = nDst
            cvt.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
            if self.cam.IMV_PixelConvert(cvt) == IMV_OK:
                buf = c_buffer(b'\0', cvt.nDstBufSize)
                memmove(buf, cvt.pDstBuf, cvt.nDstBufSize)
                img = np.array(bytearray(buf)).reshape(
                    cvt.nHeight, cvt.nWidth, 3)
                with self._lock:
                    self.current_frame = img
                del pDst
            self.cam.IMV_ReleaseFrame(frame)

    def read(self):
        with self._lock:
            return None if self.current_frame is None else self.current_frame.copy()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        # dọn sạch: StopGrabbing -> Close -> DestroyHandle. Thiếu DestroyHandle
        # có thể khiến lần chạy sau bị khoá thiết bị (CreateHandle -103).
        for fn in ("IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"):
            try:
                getattr(self.cam, fn)()
            except Exception:
                pass


# ============================================================
# FAKE (để test không cần phần cứng)
# ============================================================

class FakeCamera(BaseCamera):
    """Phát 1 ảnh tĩnh (hoặc callback sinh ảnh) - dùng test logic."""

    def __init__(self, name, frame_provider):
        self.name = name
        self._fp = frame_provider   # callable -> BGR image

    def open(self): return True
    def start(self): pass
    def read(self):
        img = self._fp()
        return None if img is None else img.copy()
    def stop(self): pass