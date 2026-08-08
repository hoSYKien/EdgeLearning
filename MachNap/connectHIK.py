import sys
import os
import numpy as np
import cv2

# ============================================================
# 1. Trỏ tới thư mục chứa DLL runtime của MVS
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(dll_dir)

# ============================================================
# 2. Trỏ THẲNG vào thư mục MvImport (nằm cùng cấp với file này)
# ============================================================
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)

# ============================================================
# 3. Import kiểu module thường (không có tiền tố MvImport.)
# ============================================================
from MvCameraControl_class import *


def main():
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("SDK Version:", hex(SDKVersion))

    # --- Enum thiết bị ---
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Không tìm thấy camera nào! Mã lỗi:", ret)
        sys.exit()

    print(f"Tìm thấy {deviceList.nDeviceNum} camera")

    # --- In danh sách camera và chọn đúng camera GigE MV-CS200-10GC ---
    target_index = 0
    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            print(f"[{i}] GigE Camera: {strModeName}")
            if "MV-CS200" in strModeName:
                target_index = i
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            print(f"[{i}] USB Camera: {strModeName}")

    # --- Lấy camera đã chọn ---
    stDeviceInfo = cast(deviceList.pDeviceInfo[target_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print("Tạo handle lỗi:", ret)
        sys.exit()

    # --- Mở camera ---
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Mở camera lỗi:", ret)
        cam.MV_CC_DestroyHandle()
        sys.exit()

    # --- Nếu là camera GigE, set packet size tối ưu để tránh mất gói ---
    if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        else:
            print("Cảnh báo: không lấy được packet size tối ưu, mã lỗi:", nPacketSize)

    # --- Tắt trigger mode để chụp liên tục ---
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # --- Bắt đầu grab ảnh ---
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing lỗi:", ret)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    # --- Lấy kích thước buffer cần thiết ---
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * nPayloadSize)()

    print("Đang lấy ảnh... Nhấn 'q' trên cửa sổ ảnh để thoát.")

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                img = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
                img = img.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)

                pixel_type = stFrameInfo.enPixelType

                if pixel_type == PixelType_Gvsp_BayerRG8:
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
                elif pixel_type == PixelType_Gvsp_BayerGB8:
                    # Dùng 2RGB thay vì 2BGR để bù lệch quy ước Bayer giữa
                    # GenICam (Hikrobot) và OpenCV, tránh bị đảo kênh R/B
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
                elif pixel_type == PixelType_Gvsp_BayerGR8:
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
                elif pixel_type == PixelType_Gvsp_BayerBG8:
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                elif pixel_type == PixelType_Gvsp_Mono8:
                    pass  # ảnh xám, không cần convert màu
                else:
                    print("Pixel format chưa được xử lý:", pixel_type)
                    continue

                # Ảnh gốc độ phân giải lớn (5472x3648) -> resize cho vừa màn hình hiển thị
                img_show = cv2.resize(img, (960, 640))
                cv2.imshow("Hikrobot Camera", img_show)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Không lấy được frame, mã lỗi:", ret)
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")


if __name__ == "__main__":
    main()