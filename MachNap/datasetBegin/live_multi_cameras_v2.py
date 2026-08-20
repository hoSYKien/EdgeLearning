# -*- coding: utf-8 -*-
"""
LIVE MULTI-CAMERAS V2 - HỆ THỐNG 3 CAMERA CÔNG NGHIỆP HIKROBOT (1 COLOR CHÍNH + 2 MONO PHỤ)
========================================================================================
Phiên bản V2 tối ưu riêng cho hệ thống 3 Camera:
  1. Tự động nhận diện Camera Màu (Color) và Camera Đen Trắng (Mono) qua Model Name / PixelFormat.
  2. Tự động ưu tiên xếp Camera Color làm [CAM 0: MAIN (CHÍNH)], 2 camera Mono làm [CAM 1 & 2: SUB (PHỤ)].
  3. Cấu hình phần cứng chuyên biệt:
     - Color Camera : Bật Auto White Balance (Cân bằng trắng liên tục), tối ưu ma trận màu BGR.
     - Mono Camera  : Bỏ qua White Balance (tránh lỗi ngầm), xử lý ảnh xám siêu tốc Zero-Overhead.
  4. Tối ưu mạng GigE cho 3 Camera chạy đồng thời:
     - Chèn Inter-Packet Delay (GevSCPD) và Packet Size chuẩn chống nghẽn Switch và sọc ngang.
     - Tăng RAM Image Buffer Node chống rớt frame.
  5. Đa dạng chế độ hiển thị (Nhấn 'g' hoặc 'v' để đổi chế độ):
     - Chế độ 1: "Master-Sub" (1 Cam Chính lớn bên trái + 2 Cam Phụ nhỏ xếp dọc bên phải).
     - Chế độ 2: "Horizontal 1x3" (3 Camera dàn hàng ngang).
     - Chế độ 3: "Grid 2x2" (Lưới 4 ô chuẩn tỉ lệ).
     - Chế độ 4: "Separate" (Từng cửa sổ riêng biệt).
  6. Lưu ảnh chụp đồng thời ('s'):
     - Lưu ảnh độ phân giải gốc có gắn nhãn rõ ràng: MAIN_COLOR_cam0_..., SUB1_MONO_cam1_..., SUB2_MONO_cam2_...
     - Lưu kèm ảnh tổng hợp Grid của cả 3 camera.

PHÍM TẮT ĐIỀU KHIỂN:
  - 'q' hoặc ESC : Thoát an toàn
  - 's'          : Chụp và lưu ảnh đồng thời từ tất cả camera
  - 'g' hoặc 'v' : Chuyển đổi qua lại giữa 4 chế độ hiển thị
  - 'w'          : Cân bằng trắng lại cho Cam Màu (White Balance Once)
  - 'h'          : In hướng dẫn phím tắt ra màn hình Console
"""

import os
import sys
import time
import threading
from ctypes import *

# Cấu hình UTF-8 cho console Windows tránh lỗi Unicode
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

# ============================================================
# 1. CẤU HÌNH DLL RUNTIME VÀ IMPORT SDK HIKROBOT (MVS)
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(dll_dir):
    try:
        os.add_dll_directory(dll_dir)
    except Exception as e:
        print(f"[Warning] Không thể add_dll_directory: {e}")

# Thêm đường dẫn MvImport
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mvs_sample_dir = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"

for p in [os.path.join(current_dir, "MvImport"), os.path.join(parent_dir, "MvImport"), mvs_sample_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from PixelType_header import *
    HIKROBOT_SDK_OK = True
except Exception as e:
    print(f"[Error] Không thể import SDK Hikrobot (MVS): {e}")
    HIKROBOT_SDK_OK = False


# ============================================================
# 2. CẤU HÌNH THÔNG SỐ HỆ THỐNG
# ============================================================
# Cấu hình IP hoặc Serial cố định cho Camera Chính (Nếu để trống "", hệ thống sẽ tự động tìm Cam Màu)
SPECIFIC_MAIN_CAM_IP = ""       # Ví dụ: "192.168.1.64" hoặc để trống "" để auto
SPECIFIC_MAIN_CAM_SERIAL = ""   # Ví dụ: "DA1234567" hoặc để trống "" để auto

TARGET_FPS = 3.3                 # FPS mục tiêu cho từng camera (Khuyến nghị: 5 - 15 FPS trên Switch 1Gbps)
GEV_PACKET_SIZE = 1500          # MTU chuẩn Ethernet (1500)
GEV_PACKET_DELAY = 1500         # Độ trễ packet tối ưu (1500 ticks) để 3 camera không tranh chấp băng thông
IMAGE_NODE_NUM = 15             # Số lượng buffer khung hình trong RAM (chống rớt frame)

# Thư mục lưu ảnh snapshot
SAVE_DIR = os.path.join(current_dir, "captures_multi_cam_v2")


# ============================================================
# 3. HÀM PHÂN TÍCH VÀ NHẬN DIỆN CAMERA
# ============================================================
def get_camera_info(dev_info):
    """Trích xuất thông tin model, IP, Serial và phân loại Color/Mono."""
    info = {
        "model": "Unknown",
        "serial": "Unknown",
        "ip": "Unknown",
        "type": "GigE" if dev_info.nTLayerType == MV_GIGE_DEVICE else "USB",
        "is_color": False,
        "role": "SUB (MONO)"
    }
    try:
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            gige_info = dev_info.SpecialInfo.stGigEInfo
            info["model"] = "".join(chr(c) for c in gige_info.chModelName if c != 0).strip()
            info["serial"] = "".join(chr(c) for c in gige_info.chSerialNumber if c != 0).strip()
            nip1 = ((gige_info.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((gige_info.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((gige_info.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (gige_info.nCurrentIp & 0x000000ff)
            info["ip"] = f"{nip1}.{nip2}.{nip3}.{nip4}"
        elif dev_info.nTLayerType == MV_USB_DEVICE:
            usb_info = dev_info.SpecialInfo.stUsb3VInfo
            info["model"] = "".join(chr(c) for c in usb_info.chModelName if c != 0).strip()
            info["serial"] = "".join(chr(c) for c in usb_info.chSerialNumber if c != 0).strip()
            info["ip"] = "USB"

        # Phân loại Color vs Mono theo quy ước đặt tên Hikrobot:
        # Camera Color thường có đuôi GC, UC, C (ví dụ: MV-CS050-10GC, MV-CA013-21UC, MV-CS200-10GC)
        # Camera Mono thường có đuôi GM, UM, M (ví dụ: MV-CS050-10GM, MV-CA013-21UM)
        model_upper = info["model"].upper()
        if any(c_tag in model_upper for c_tag in ["-10GC", "-20GC", "-10UC", "-20UC", "-13GC", "-13UC", "GC", "UC", "COLOR"]):
            info["is_color"] = True
        elif model_upper.endswith("C"):
            info["is_color"] = True
        else:
            info["is_color"] = False

    except Exception as e:
        print(f"[Warning] Lỗi khi trích xuất thông tin camera: {e}")
    return info


# ============================================================
# 4. WORKER THU HÌNH CAMERA ĐỘC LẬP
# ============================================================
class CameraWorker:
    def __init__(self, dev_info, cam_id, is_main=False, target_fps=TARGET_FPS):
        self.cam_id = cam_id
        self.dev_info = dev_info
        self.cam_info = get_camera_info(dev_info)
        self.is_main = is_main
        self.target_fps = target_fps
        self.cam = MvCamera()
        
        self.is_running = False
        self.thread = None
        self._lock = threading.Lock()
        
        self.latest_frame = None
        self.latest_frame_id = 0
        self.fps = 0.0
        self._fps_count = 0
        self._fps_time = time.time()
        
        # Buffer chuyển đổi màu
        self._convert_buf = None
        self._convert_buf_size = 0

    def open(self):
        # 1. Tạo Handle
        ret = self.cam.MV_CC_CreateHandle(self.dev_info)
        if ret != 0:
            print(f"[Cam {self.cam_id}] Lỗi MV_CC_CreateHandle: 0x{ret:08x}")
            return False

        # 2. Mở Camera chế độ Exclusive
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            err_msg = f"0x{ret:08x}"
            if ret == 0x80000203:
                err_msg += " (MV_E_ACCESS_DENIED: Camera đang bị chiếm bởi phần mềm khác như MVS.exe)"
            elif ret == 0x80000204:
                err_msg += " (MV_E_BUSY: Thiết bị đang bận hoặc ngắt kết nối mạng)"
            elif ret == 0x80000221:
                err_msg += " (MV_E_IP_CONFLICT: Trùng địa chỉ IP camera)"
            elif ret == 0x80000215 or ret == 0x80000206:
                err_msg += f" (LỆCH DẢI MẠNG / SUBNET MISMATCH: Camera đang ở IP {self.cam_info['ip']}, khác dải mạng với Card mạng máy tính. Vui lòng mở MVS đổi IP về cùng dải 192.168.1.x)"
            print(f"[Cam {self.cam_id}] Lỗi MV_CC_OpenDevice: {err_msg}")
            self.cam.MV_CC_DestroyHandle()
            return False

        # 3. Tối ưu mạng GigE: Chia đều băng thông và tránh xung đột gói tin giữa các camera
        if self.dev_info.nTLayerType == MV_GIGE_DEVICE:
            # Tự động đàm phán Packet Size tối ưu từ SDK nếu hỗ trợ
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

            # Chỉ chèn Inter-Packet Delay cho camera Hikrobot (tránh làm chậm camera hãng khác như Sentech/Basler)
            is_hikrobot = any(k in self.cam_info["model"].upper() for k in ["MV-", "KC", "HIK"])
            if is_hikrobot:
                try:
                    self.cam.MV_CC_SetIntValue("GevSCPD", GEV_PACKET_DELAY)
                    self.cam.MV_CC_SetBoolValue("GevSCPHostResend", True)
                except Exception:
                    pass
            else:
                # Với camera Sentech/Hãng khác: Không ép GevSCPD quá lớn
                try:
                    self.cam.MV_CC_SetIntValue("GevSCPD", 400)
                except Exception:
                    pass

        # 4. Tăng bộ đệm khung hình trong RAM (chống rớt frame)
        try:
            self.cam.MV_CC_SetImageNodeNum(IMAGE_NODE_NUM)
        except Exception:
            pass

        # 5. CHIA ĐỀU VÀ KHÓA CHẶT FPS MỤC TIÊU CHO TẤT CẢ CAMERA
        if self.target_fps is not None and self.target_fps > 0:
            try:
                self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))
            except Exception:
                try:
                    # Một số camera hãng khác dùng node "AcquisitionFrameRateControl" hoặc float trực tiếp
                    self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))
                except Exception:
                    pass

        # 6. Kiểm soát Exposure Time trên tất cả camera để không camera nào bị kéo tụt FPS
        try:
            stFloat = MVCC_FLOATVALUE()
            memset(byref(stFloat), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloat)
            max_allowed_exp = (1.0 / self.target_fps) * 1000000.0 * 0.80
            if ret == 0 and stFloat.fCurValue > max_allowed_exp:
                print(f"[Cam {self.cam_id}] Phơi sáng ({stFloat.fCurValue:.0f} us) quá dài sẽ làm tụt FPS dưới {self.target_fps}. Đang giảm về {max_allowed_exp:.0f} us...")
                self.cam.MV_CC_SetFloatValue("ExposureTime", max_allowed_exp)
        except Exception:
            pass

        # 7. Tắt Trigger Mode & Bật Thu hình liên tục (Dùng chuỗi GenICam chuẩn cho mọi hãng)
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

        # 8. Cấu hình riêng cho Camera Màu vs Camera Đen Trắng
        if self.cam_info["is_color"]:
            try:
                # Bật Tự Động Cân Bằng Trắng liên tục cho camera Màu
                self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Continuous")
            except Exception:
                pass
        
        role_str = "MAIN (COLOR)" if self.is_main else ("SUB (COLOR)" if self.cam_info["is_color"] else "SUB (MONO)")
        print(f"[Cam {self.cam_id}] [{role_str}] Đã mở thành công (Đã khóa chuẩn {self.target_fps:.1f} FPS): {self.cam_info['model']} | IP: {self.cam_info['ip']}")
        return True

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[Cam {self.cam_id}] Lỗi MV_CC_StartGrabbing: 0x{ret:08x}")
            return False
        
        # Gửi lệnh AcquisitionStart để kích hoạt stream trên camera hãng thứ ba (như Sentech)
        try:
            self.cam.MV_CC_SetCommandValue("AcquisitionStart")
        except Exception:
            pass

        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, name=f"CamWorker-{self.cam_id}")
        self.thread.daemon = True
        self.thread.start()
        return True

    def set_white_balance_once(self):
        """Kích hoạt cân bằng trắng 1 lần (dành cho Camera Màu)."""
        if self.cam_info["is_color"]:
            try:
                self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Once")
                print(f"[Cam {self.cam_id}] Đã kích hoạt cân bằng trắng (White Balance Once).")
            except Exception as e:
                print(f"[Cam {self.cam_id}] Lỗi White Balance: {e}")

    def _convert_frame_to_bgr(self, stFrame):
        """Chuyển đổi dữ liệu sang ảnh OpenCV BGR."""
        w = stFrame.stFrameInfo.nWidth
        h = stFrame.stFrameInfo.nHeight
        frame_len = stFrame.stFrameInfo.nFrameLen
        pixel_type = stFrame.stFrameInfo.enPixelType

        buf_addr = cast(stFrame.pBufAddr, c_void_p).value
        if not buf_addr or frame_len == 0 or w == 0 or h == 0:
            return None

        # 1. Ảnh Đen Trắng Mono8 (Xử lý trực tiếp siêu nhanh)
        if pixel_type == PixelType_Gvsp_Mono8:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. Ảnh BGR8_Packed
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3)).copy()

        # 3. Các định dạng Bayer / RGB / YUV -> Dùng hàm chuyển đổi SDK chuẩn xác
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
                # Fallback qua OpenCV nếu SDK convert lỗi
                raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                if pixel_type == PixelType_Gvsp_BayerGB8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_GB2BGR)
                elif pixel_type == PixelType_Gvsp_BayerRG8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_RG2BGR)
                elif pixel_type == PixelType_Gvsp_BayerGR8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_GR2BGR)
                elif pixel_type == PixelType_Gvsp_BayerBG8:
                    return cv2.cvtColor(raw_arr.reshape((h, w)), cv2.COLOR_BAYER_BG2BGR)
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
                    print(f"[Cam {self.cam_id}] Lỗi xử lý frame: {e}")
                finally:
                    self.cam.MV_CC_FreeImageBuffer(stFrame)
            else:
                time.sleep(0.005)

    def get_latest_frame(self):
        """Lấy frame mới nhất (Thread-safe)."""
        with self._lock:
            if self.latest_frame is None:
                return None, 0, 0.0
            return self.latest_frame.copy(), self.latest_frame_id, self.fps

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        print(f"[Cam {self.cam_id}] Đã dừng và giải phóng thành công.")


# ============================================================
# 5. HÀM TẠO CANVAS HIỂN THỊ CÓ BADGE NỔI BẬT
# ============================================================
def create_display_canvas(frame, cam_worker, target_w, target_h):
    """Tạo khung hình hiển thị kèm thanh thông tin nhãn & viền màu phân biệt."""
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    is_main = cam_worker.is_main
    is_color = cam_worker.cam_info["is_color"]
    fps = cam_worker.fps
    frame_id = cam_worker.latest_frame_id

    # Màu sắc chủ đạo: Xanh lá cho Main Cam, Cyan/Vàng cho Sub Cams, Đỏ cho lỗi
    if not cam_worker.is_running:
        accent_color = (0, 0, 255) # Màu đỏ cảnh báo
    else:
        accent_color = (0, 255, 0) if is_main else ((255, 200, 0) if is_color else (0, 200, 255))
    
    role_text = "MAIN (COLOR)" if is_main else ("SUB (COLOR)" if is_color else "SUB (MONO)")

    if cam_worker.is_running:
        if frame is not None:
            h, w = frame.shape[:2]
            scale = min(target_w / w, (target_h - 32) / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))

            x_off = (target_w - nw) // 2
            y_off = 32 + ((target_h - 32 - nh) // 2)
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

            # Vẽ viền cho Cam Chính
            if is_main:
                cv2.rectangle(canvas, (0, 0), (target_w - 1, target_h - 1), accent_color, 2)
        else:
            cv2.putText(canvas, f"Cam {cam_worker.cam_id}: Waiting for frame...", (20, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        # Vẽ cảnh báo nếu camera không mở được (Bị chiếm quyền bởi MVS.exe hoặc mất mạng)
        cv2.rectangle(canvas, (2, 2), (target_w - 3, target_h - 3), (0, 0, 220), 2)
        cv2.putText(canvas, f"[LỖI KHÔNG THỂ MỞ CAMERA]", (20, target_h // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Model: {cam_worker.cam_info['model']} | IP: {cam_worker.cam_info['ip']}", (20, target_h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Nguyên nhân: Đang bị phần mềm MVS.exe chiếm", (20, target_h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "-> Hãy tắt hoàn toàn MVS.exe rồi chạy lại script!", (20, target_h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1, cv2.LINE_AA)

    # Vẽ thanh Header trên cùng
    cv2.rectangle(canvas, (0, 0), (target_w, 30), (20, 20, 20), -1)
    cv2.line(canvas, (0, 30), (target_w, 30), accent_color, 1)

    label_left = f"[CAM {cam_worker.cam_id}: {role_text}] {cam_worker.cam_info['model']}"
    label_right = f"FPS: {fps:4.1f}/{cam_worker.target_fps:.0f} | #{frame_id}" if cam_worker.is_running else "OFFLINE / BUSY"
    
    cv2.putText(canvas, label_left, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent_color, 1, cv2.LINE_AA)
    
    (rw, rh), _ = cv2.getTextSize(label_right, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(canvas, label_right, (target_w - rw - 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ============================================================
# 6. CHƯƠNG TRÌNH CHÍNH (MAIN THREAD GUI LOOP)
# ============================================================
def main():
    if not HIKROBOT_SDK_OK:
        print("[Lỗi] Không tìm thấy thư viện MVS Hikrobot SDK.")
        return

    # Khởi tạo SDK
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"[Cảnh báo] MV_CC_Initialize: 0x{ret:08x}")

    # 1. Liệt kê tất cả các thiết bị GigE và USB
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
    
    if ret != 0 or device_list.nDeviceNum == 0:
        print(f"[Lỗi] Không tìm thấy camera nào trên mạng! (Mã lỗi: 0x{ret:08x})")
        MvCamera.MV_CC_Finalize()
        return

    num_devices = device_list.nDeviceNum
    print("=" * 70)
    print(f"TÌM THẤY {num_devices} CAMERA TRÊN HỆ THỐNG:")
    
    dev_info_list = []
    for i in range(num_devices):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam_info = get_camera_info(dev_info)
        dev_info_list.append((dev_info, cam_info))
        cam_type_tag = "[COLOR]" if cam_info["is_color"] else "[MONO ]"
        print(f"  [{i}] {cam_type_tag} {cam_info['type']} - {cam_info['model']} | IP: {cam_info['ip']} | SN: {cam_info['serial']}")
    print("=" * 70)

    # 2. XÁC ĐỊNH CAMERA CHÍNH (MAIN COLOR CAMERA) & SẮP XẾP CỐ ĐỊNH THỨ TỰ SLOT
    main_idx = None

    # Cách A: Ưu tiên theo IP hoặc Serial nếu người dùng chỉ định
    if SPECIFIC_MAIN_CAM_IP != "" or SPECIFIC_MAIN_CAM_SERIAL != "":
        for idx, (_, info) in enumerate(dev_info_list):
            if (SPECIFIC_MAIN_CAM_IP and info["ip"] == SPECIFIC_MAIN_CAM_IP) or \
               (SPECIFIC_MAIN_CAM_SERIAL and info["serial"] == SPECIFIC_MAIN_CAM_SERIAL):
                main_idx = idx
                break

    # Cách B: Tự động chọn Camera Màu đầu tiên làm Cam Chính
    if main_idx is None:
        for idx, (_, info) in enumerate(dev_info_list):
            if info["is_color"]:
                main_idx = idx
                break

    # Nếu không có camera màu, chọn camera đầu tiên
    if main_idx is None:
        main_idx = 0

    # Đưa Main Camera lên vị trí index 0 (Cố định Slot 0 = MAIN COLOR)
    sorted_dev_info = [dev_info_list[main_idx]] + [dev_info_list[i] for i in range(len(dev_info_list)) if i != main_idx]

    print("\n[PHÂN CHIA VAI TRÒ CỐ ĐỊNH TỪNG SLOT CAMERA]:")
    print(f"  -> SLOT 0 [MAIN - COLOR]: {sorted_dev_info[0][1]['model']} (IP: {sorted_dev_info[0][1]['ip']})")
    for i in range(1, len(sorted_dev_info)):
        print(f"  -> SLOT {i} [SUB  - MONO ]: {sorted_dev_info[i][1]['model']} (IP: {sorted_dev_info[i][1]['ip']})")
    print("=" * 70)

    # 3. Mở và khởi động từng Camera theo đúng Slot
    workers = []
    opened_count = 0
    for i, (dev_info, _) in enumerate(sorted_dev_info):
        is_main = (i == 0)
        worker = CameraWorker(dev_info=dev_info, cam_id=i, is_main=is_main, target_fps=TARGET_FPS)
        if worker.open():
            if worker.start():
                opened_count += 1
            else:
                worker.stop()
        workers.append(worker)

    if opened_count == 0:
        print("[Lỗi] Không thể mở bất kỳ camera nào. Đang thoát...")
        MvCamera.MV_CC_Finalize()
        return

    print(f"\n[OK] Đã khởi động thành công {opened_count}/{num_devices} camera.")
    print(f"     -> Đã chia đều và khóa chuẩn {TARGET_FPS:.1f} FPS cho TẤT CẢ Camera.")
    if opened_count < num_devices:
        print("[CẢNH BÁO] Có camera không mở được do đang bị phần mềm MVS.exe chiếm hoặc lệch dải IP!")
        print("          Vui lòng TẮT MVS.exe để Python có thể kết nối độc quyền.")
    print("=" * 70)
    print("PHÍM TẮT ĐIỀU KHIỂN:")
    print("  - 'q' hoặc ESC : Thoát chương trình")
    print("  - 's'          : Chụp và lưu ảnh đồng thời từ tất cả camera")
    print("  - 'g' hoặc 'v' : Chuyển đổi qua lại giữa 4 chế độ hiển thị")
    print("  - 'w'          : Cân bằng trắng 1 lần cho Cam Màu (White Balance Once)")
    print("  - 'h'          : In lại trợ giúp phím tắt")
    print("=" * 70)

    # Chế độ hiển thị: 0: Master-Sub (1 lớn 2 nhỏ), 1: 1x3 ngang, 2: 2x2 Lưới, 3: Cửa sổ riêng
    layout_mode = 0
    layout_names = ["1 Lớn + 2 Nhỏ (Master-Sub)", "1x3 Ngang (Horizontal)", "2x2 Lưới (Grid)", "Cửa Sổ Riêng Biệt"]

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        while True:
            # Thu thập frame mới nhất từ tất cả worker
            frames_raw = []
            for w in workers:
                frame, frame_id, fps = w.get_latest_frame()
                frames_raw.append(frame)

            # ========================================================
            # BỐ CỤC HIỂN THỊ
            # ========================================================
            if layout_mode == 0:
                # ----------------------------------------------------
                # Chế độ 1: 1 Lớn (Bên trái) + 2 Nhỏ (Bên phải xếp dọc)
                # ----------------------------------------------------
                main_w, main_h = 800, 600
                sub_w, sub_h = 440, 300

                canvas_main = create_display_canvas(frames_raw[0], workers[0], main_w, main_h)
                
                sub_canvases = []
                for idx in range(1, 3):
                    if idx < len(workers):
                        sub_canvas = create_display_canvas(frames_raw[idx], workers[idx], sub_w, sub_h)
                    else:
                        sub_canvas = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
                    sub_canvases.append(sub_canvas)

                right_col = np.vstack(sub_canvases)
                # Ghép cột chính và cột phụ
                full_view = np.hstack([canvas_main, right_col])
                cv2.imshow("Multi-Camera Live View V2 (1 Main + 2 Sub)", full_view)

            elif layout_mode == 1:
                # ----------------------------------------------------
                # Chế độ 2: 1x3 Ngang (3 camera dàn hàng ngang)
                # ----------------------------------------------------
                cell_w, cell_h = 480, 360
                row_canvases = []
                for idx, w in enumerate(workers):
                    c = create_display_canvas(frames_raw[idx], w, cell_w, cell_h)
                    row_canvases.append(c)
                
                full_view = np.hstack(row_canvases)
                cv2.imshow("Multi-Camera Live View V2 (1x3 Horizontal)", full_view)

            elif layout_mode == 2:
                # ----------------------------------------------------
                # Chế độ 3: 2x2 Lưới (Grid 4 ô)
                # ----------------------------------------------------
                cell_w, cell_h = 600, 450
                canvases = []
                for idx in range(4):
                    if idx < len(workers):
                        c = create_display_canvas(frames_raw[idx], workers[idx], cell_w, cell_h)
                    else:
                        c = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                        cv2.putText(c, "[No Camera]", (cell_w // 3, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
                    canvases.append(c)

                top_row = np.hstack([canvases[0], canvases[1]])
                bot_row = np.hstack([canvases[2], canvases[3]])
                full_view = np.vstack([top_row, bot_row])
                cv2.imshow("Multi-Camera Live View V2 (2x2 Grid)", full_view)

            else:
                # ----------------------------------------------------
                # Chế độ 4: Từng Cửa Sổ Riêng Biệt
                # ----------------------------------------------------
                for idx, w in enumerate(workers):
                    c = create_display_canvas(frames_raw[idx], w, 640, 480)
                    role_str = "MAIN_COLOR" if w.is_main else f"SUB_MONO_{idx}"
                    cv2.imshow(f"Camera {idx} - {role_str}", c)

            # ========================================================
            # XỬ LÝ PHÍM BẤM
            # ========================================================
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                break
            
            elif key in [ord('g'), ord('G'), ord('v'), ord('V')]:
                # Đóng các cửa sổ cũ khi chuyển chế độ
                cv2.destroyAllWindows()
                layout_mode = (layout_mode + 1) % 4
                print(f"\n[Layout] -> Đã chuyển sang chế độ hiển thị: [{layout_names[layout_mode]}]")

            elif key in [ord('w'), ord('W')]:
                # Cân bằng trắng cho camera màu
                for w in workers:
                    if w.cam_info["is_color"]:
                        w.set_white_balance_once()

            elif key in [ord('h'), ord('H')]:
                print("\n" + "=" * 50)
                print("HƯỚNG DẪN PHÍM TẮT:")
                print("  - 'q' / ESC : Thoát")
                print("  - 's'       : Chụp lưu ảnh tất cả camera vào thư mục captures_multi_cam_v2")
                print("  - 'g' / 'v' : Chuyển chế độ hiển thị (1 Lớn 2 Nhỏ -> 1x3 Ngang -> 2x2 -> Cửa sổ riêng)")
                print("  - 'w'       : Auto White Balance Once cho Cam Màu")
                print("=" * 50)

            elif key in [ord('s'), ord('S')]:
                # Lưu ảnh gốc chất lượng cao từ tất cả camera
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                saved_files = []
                for idx, w in enumerate(workers):
                    raw_f = frames_raw[idx]
                    if raw_f is not None:
                        role_tag = "MAIN_COLOR" if w.is_main else f"SUB{idx}_MONO"
                        fname = f"{role_tag}_cam{idx}_{w.cam_info['serial']}_{timestamp}.png"
                        fpath = os.path.join(SAVE_DIR, fname)
                        cv2.imwrite(fpath, raw_f)
                        saved_files.append(fname)

                print(f"\n[Snapshot] Đã lưu thành công {len(saved_files)} ảnh vào '{SAVE_DIR}':")
                for fn in saved_files:
                    print(f"   -> {fn}")

    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu dừng từ bàn phím...")
    finally:
        print("\nĐang dừng tất cả camera và giải phóng tài nguyên...")
        for w in workers:
            try:
                w.stop()
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
        print("[Hoàn tất] Đã đóng toàn bộ camera và thoát an toàn.")


if __name__ == "__main__":
    main()
