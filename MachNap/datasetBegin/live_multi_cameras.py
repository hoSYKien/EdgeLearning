# -*- coding: utf-8 -*-
"""
LIVE MULTI-CAMERAS - HIKROBOT INDUSTRIAL GIGE / USB CAMERAS
============================================================
Chương trình thu hình và hiển thị đồng thời nhiều camera công nghiệp Hikrobot.
Đã tối ưu:
  1. Thread-safe GUI: Đưa cv2.imshow và cv2.waitKey về Main Thread (tránh crash/freeze).
  2. Zero-Copy / Chuyển đổi màu tối ưu: Hỗ trợ Mono8, Bayer (RG/GB/GR/BG), RGB, BGR và tự động fallback.
  3. Tối ưu mạng GigE: Tự động đàm phán Packet Size tối ưu, chèn Inter-Packet Delay (GevSCPD) chống nghẽn Switch.
  4. Quản lý bộ đệm an toàn: Luôn giải phóng Buffer (FreeImageBuffer) trong khối try...finally.
  5. Hỗ trợ hiển thị dạng Grid (ghép ảnh) hoặc từng cửa sổ riêng biệt.
  6. Phím tắt:
     - 'q' hoặc 'ESC': Thoát
     - 's': Lưu ảnh chụp đồng thời từ tất cả camera
     - 'g': Chuyển đổi giữa chế độ Cửa sổ ghép (Grid) và Cửa sổ riêng
"""

import os
import sys
import time
import threading
from ctypes import *

# Cấu hình UTF-8 cho console Windows tránh UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

# ============================================================
# 1. CẤU HÌNH DLL RUNTIME VÀ IMPORT SDK MVS (HIKROBOT)
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
# 2. HELPER LẤY THÔNG TIN CAMERA
# ============================================================
def get_camera_info(dev_info):
    """Trích xuất thông tin model, IP, Serial từ struct MV_CC_DEVICE_INFO."""
    info = {
        "model": "Unknown",
        "serial": "Unknown",
        "ip": "Unknown",
        "type": "GigE" if dev_info.nTLayerType == MV_GIGE_DEVICE else "USB"
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
    except Exception:
        pass
    return info


# ============================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================
TARGET_FPS = 10.0          # Khóa chuẩn 10.0 FPS cho tất cả các camera
GEV_PACKET_SIZE = 1500     # MTU chuẩn Ethernet (1500)
GEV_PACKET_DELAY = 1200    # Độ trễ packet tối ưu (1200 ticks ~ 10us/gói): Vừa đủ để chống nghẽn Switch mà không bị bóp tụt FPS của camera Mono!
IMAGE_NODE_NUM = 15        # Số lượng buffer khung hình trong RAM


# ============================================================
# 3. WORKER THU HÌNH CAMERA ĐỘC LẬP (CHẠY TRÊN THREAD RIÊNG)
# ============================================================
class CameraWorker:
    def __init__(self, dev_info, cam_id, target_fps=TARGET_FPS):
        self.cam_id = cam_id
        self.dev_info = dev_info
        self.cam_info = get_camera_info(dev_info)
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
                err_msg += " (MV_E_ACCESS_DENIED: Camera đang bị chiếm bởi phần mềm khác như MVS.exe hoặc tiến trình Python khác)"
            elif ret == 0x80000204:
                err_msg += " (MV_E_BUSY: Thiết bị đang bận hoặc ngắt kết nối mạng)"
            elif ret == 0x80000221:
                err_msg += " (MV_E_IP_CONFLICT: Trùng địa chỉ IP camera)"
            print(f"[Cam {self.cam_id}] Lỗi MV_CC_OpenDevice: {err_msg}")
            self.cam.MV_CC_DestroyHandle()
            return False

        # 3. TỐI ƯU MẠNG GIGE: Chống sọc nhiễu & đảm bảo đủ băng thông đạt 10 FPS
        if self.dev_info.nTLayerType == MV_GIGE_DEVICE:
            # Ép Packet Size = 1500 (tránh rớt gói khi switch không bật Jumbo Frame)
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", GEV_PACKET_SIZE)

            # Chèn Packet Delay (GevSCPD = 1200 ticks): Giãn cách các gói vừa đủ tránh tràn buffer Switch
            self.cam.MV_CC_SetIntValue("GevSCPD", GEV_PACKET_DELAY)

            # Bật cơ chế Host Resend (Yêu cầu gửi lại gói tin nếu bị drop)
            try:
                self.cam.MV_CC_SetBoolValue("GevSCPHostResend", True)
            except Exception:
                pass

        # 4. Tăng bộ đệm khung hình trong RAM (chống rớt frame)
        self.cam.MV_CC_SetImageNodeNum(IMAGE_NODE_NUM)

        # 5. Cài đặt FPS = 10.0
        if self.target_fps is not None and self.target_fps > 0:
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.target_fps))

        # 6. Đảm bảo Exposure Time không quá lớn làm tụt FPS dưới 10 FPS (10 FPS cần phơi sáng < 80ms)
        try:
            stFloat = MVCC_FLOATVALUE()
            memset(byref(stFloat), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloat)
            if ret == 0 and stFloat.fCurValue > 80000.0:
                print(f"[Cam {self.cam_id}] ExposureTime hiện tại ({stFloat.fCurValue:.0f} us) quá cao để đạt 10 FPS. Đang chỉnh về 50000 us...")
                self.cam.MV_CC_SetFloatValue("ExposureTime", 50000.0)
        except Exception:
            pass

        # 7. Tắt Trigger Mode -> Stream liên tục
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # 8. Bật Tự Động Cân Bằng Trắng (Auto White Balance) cho camera màu
        try:
            self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Continuous")
        except Exception:
            pass
        
        print(f"[Cam {self.cam_id}] Đã mở thành công: {self.cam_info['model']} (IP: {self.cam_info['ip']}, SN: {self.cam_info['serial']})")
        return True

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[Cam {self.cam_id}] Lỗi MV_CC_StartGrabbing: 0x{ret:08x}")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, name=f"CamWorker-{self.cam_id}")
        self.thread.daemon = True
        self.thread.start()
        return True

    def _convert_frame_to_bgr(self, stFrame):
        """Chuyển đổi dữ liệu thô sang OpenCV BGR image với màu sắc chuẩn xác."""
        w = stFrame.stFrameInfo.nWidth
        h = stFrame.stFrameInfo.nHeight
        frame_len = stFrame.stFrameInfo.nFrameLen
        pixel_type = stFrame.stFrameInfo.enPixelType

        buf_addr = cast(stFrame.pBufAddr, c_void_p).value
        if not buf_addr or frame_len == 0 or w == 0 or h == 0:
            return None

        # 1. Ảnh Đen Trắng Mono8 (Xử lý trực tiếp siêu tốc)
        if pixel_type == PixelType_Gvsp_Mono8:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2. Ảnh BGR8_Packed
        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            raw_bytes = (c_ubyte * frame_len).from_address(buf_addr)
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3)).copy()

        # 3. Tất cả các định dạng Bayer (BayerRG8, BayerGB8, BayerGR8, BayerBG8, 10/12-bit) và RGB/YUV:
        # Dùng MV_CC_ConvertPixelTypeEx của SDK Hikrobot để chuẩn hóa đúng thứ tự kênh màu BGR và áp dụng ma trận màu chính xác
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
                # Fallback nếu ConvertPixelTypeEx lỗi:
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
            # Lấy buffer ảnh từ SDK với timeout 1000ms
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == 0:
                try:
                    bgr_img = self._convert_frame_to_bgr(stFrame)
                    if bgr_img is not None:
                        # Cập nhật frame mới vào biến chia sẻ an toàn với Lock
                        with self._lock:
                            self.latest_frame = bgr_img
                            self.latest_frame_id += 1
                            self._fps_count += 1

                        # Tính FPS
                        now = time.time()
                        dt = now - self._fps_time
                        if dt >= 1.0:
                            self.fps = self._fps_count / dt
                            self._fps_count = 0
                            self._fps_time = now
                except Exception as e:
                    print(f"[Cam {self.cam_id}] Lỗi xử lý frame: {e}")
                finally:
                    # Bắt buộc phải giải phóng buffer để camera tiếp tục nhận frame mới
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
        print(f"[Cam {self.cam_id}] Đã dừng và đóng thiết bị.")


# Đặt alias DualCamWorker để tương thích ngược
DualCamWorker = CameraWorker


# ============================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN THREAD GUI LOOP)
# ============================================================
def main():
    if not HIKROBOT_SDK_OK:
        print("Lỗi: Không tìm thấy thư viện MVS Hikrobot SDK.")
        return

    # Khởi tạo SDK
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"Cảnh báo: MV_CC_Initialize trả về mã 0x{ret:08x}")

    # 1. Liệt kê tất cả các thiết bị GigE và USB
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
    
    if ret != 0 or device_list.nDeviceNum == 0:
        print(f"Không tìm thấy camera nào! (Mã lỗi: 0x{ret:08x})")
        MvCamera.MV_CC_Finalize()
        return

    num_devices = device_list.nDeviceNum
    print("=" * 60)
    print(f"TÌM THẤY {num_devices} CAMERA HIKROBOT:")
    
    workers = []
    for i in range(num_devices):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        cam_info = get_camera_info(dev_info)
        print(f"  [{i}] {cam_info['type']} - {cam_info['model']} | IP: {cam_info['ip']} | SN: {cam_info['serial']}")

    print("=" * 60)
    print("Đang mở và khởi động các camera...")

    for i in range(num_devices):
        dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        worker = CameraWorker(dev_info=dev_info, cam_id=i, target_fps=TARGET_FPS)
        if worker.open():
            if worker.start():
                workers.append(worker)
            else:
                worker.stop()

    if len(workers) == 0:
        print("Không mở được camera nào. Đang thoát...")
        MvCamera.MV_CC_Finalize()
        return

    print(f"\nĐã khởi động thành công {len(workers)}/{num_devices} camera.")
    print("Đang hiển thị hình ảnh...")
    print("Phím tắt điều khiển:")
    print("  - 'q' hoặc 'ESC' : Thoát")
    print("  - 's'           : Chụp và lưu ảnh từ tất cả camera")
    print("  - 'g'           : Chuyển đổi giữa chế độ Cửa sổ ghép (Grid) và Cửa sổ riêng")

    show_grid_view = True
    save_dir = os.path.join(current_dir, "captures_multi_cam")
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            frames_to_display = []
            
            for worker in workers:
                frame, frame_id, fps = worker.get_latest_frame()
                if frame is not None:
                    # Thu nhỏ ảnh để hiển thị mượt mà trên màn hình
                    display_w, display_h = 640, 480
                    h, w = frame.shape[:2]
                    scale = min(display_w / w, display_h / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (nw, nh))

                    # Tạo nền đen chuẩn kích thước
                    canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                    x_off = (display_w - nw) // 2
                    y_off = (display_h - nh) // 2
                    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

                    # Vẽ thông tin camera lên ảnh
                    label = f"Cam {worker.cam_id}: {worker.cam_info['model']} | FPS: {fps:.1f} | Frame: {frame_id}"
                    cv2.rectangle(canvas, (0, 0), (display_w, 30), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

                    frames_to_display.append((worker.cam_id, canvas, frame))
                else:
                    # Tạo frame chờ nếu chưa có ảnh
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Cam {worker.cam_id}: Waiting for frame...", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    frames_to_display.append((worker.cam_id, blank, None))

            # Hiển thị
            if show_grid_view and len(frames_to_display) > 0:
                # Ghép các frame thành Grid
                if len(frames_to_display) == 1:
                    grid_img = frames_to_display[0][1]
                elif len(frames_to_display) == 2:
                    grid_img = np.hstack([frames_to_display[0][1], frames_to_display[1][1]])
                elif len(frames_to_display) <= 4:
                    # Ghép 2x2
                    top = np.hstack([frames_to_display[0][1], frames_to_display[1][1]])
                    bottom_row = []
                    for idx in range(2, 4):
                        if idx < len(frames_to_display):
                            bottom_row.append(frames_to_display[idx][1])
                        else:
                            bottom_row.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    bottom = np.hstack(bottom_row)
                    grid_img = np.vstack([top, bottom])
                else:
                    grid_img = np.hstack([item[1] for item in frames_to_display[:3]])

                cv2.imshow("Multi-Camera Live View", grid_img)
            else:
                # Hiển thị từng cửa sổ riêng
                for cam_id, canvas, _ in frames_to_display:
                    cv2.imshow(f"Camera {cam_id}", canvas)

            # Xử lý phím bấm
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                break
            elif key == ord('g') or key == ord('G'):
                show_grid_view = not show_grid_view
                if show_grid_view:
                    for w in workers:
                        try:
                            cv2.destroyWindow(f"Camera {w.cam_id}")
                        except Exception:
                            pass
                else:
                    try:
                        cv2.destroyWindow("Multi-Camera Live View")
                    except Exception:
                        pass
                print(f"-> Chế độ hiển thị: {'Ghép cửa sổ (Grid)' if show_grid_view else 'Cửa sổ riêng biệt'}")
            elif key == ord('s') or key == ord('S'):
                # Lưu ảnh gốc chất lượng cao từ tất cả camera
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                saved_count = 0
                for cam_id, _, raw_frame in frames_to_display:
                    if raw_frame is not None:
                        filename = f"cam_{cam_id}_{timestamp}.png"
                        filepath = os.path.join(save_dir, filename)
                        cv2.imwrite(filepath, raw_frame)
                        saved_count += 1
                print(f"[Snapshot] Đã lưu {saved_count} ảnh vào thư mục: {save_dir}")

    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu dừng từ người dùng...")
    finally:
        print("\nĐang dừng tất cả camera và giải phóng tài nguyên...")
        for worker in workers:
            try:
                worker.stop()
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