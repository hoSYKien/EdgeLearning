import threading
import os
import cv2
import numpy as np

from IMVApi import *


class CameraIndustrial:
    def __init__(self, index=0):
        self.cam = MvCamera()
        self.handle = None
        self.is_running = False
        self.current_frame = None
        self.index = index
        self.thread = None

    def open(self):
        # 1. Tìm thiết bị
        deviceList = IMV_DeviceList()
        print("Enumerating devices...")
        MvCamera.IMV_EnumDevices(
            deviceList, IMV_EInterfaceType.interfaceTypeAll)
        if deviceList.nDevNum == 0:
            return False

        # 2. Tạo handle và Mở
        self.cam.IMV_CreateHandle(
            IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(self.index)))
        self.cam.IMV_Open()

        # 3. Tắt trigger để lấy ảnh liên tục
        self.cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")

        # 4. White Balance
        self.cam.IMV_SetEnumFeatureSymbol("BalanceWhiteAuto", "Once")
        self.cam.IMV_SetDoubleFeatureValue("ExposureTime", 1000)
        self.cam.IMV_SetDoubleFeatureValue("GainRaw", 2.41)

        return True

    def _grabbing_thread(self):
        """ Luồng chạy ngầm để lấy ảnh liên tục """
        self.cam.IMV_StartGrabbing()
        while self.is_running:
            frame = IMV_Frame()
            stPixelConvertParam = IMV_PixelConvertParam()
            nRet = self.cam.IMV_GetFrame(frame, 1000)

            if nRet == IMV_OK:
                if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
                    nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
                else:
                    nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3

                pDstBuf = (c_ubyte * nDstBufSize)()
                memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))

                stPixelConvertParam.nWidth = frame.frameInfo.width
                stPixelConvertParam.nHeight = frame.frameInfo.height
                stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
                stPixelConvertParam.pSrcData = frame.pData
                stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
                stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
                stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
                stPixelConvertParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
                stPixelConvertParam.eDstPixelFormat = frame.frameInfo.pixelFormat
                stPixelConvertParam.pDstBuf = pDstBuf
                stPixelConvertParam.nDstBufSize = nDstBufSize

                # convert to BGR24
                stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
                nRet = self.cam.IMV_PixelConvert(stPixelConvertParam)

                if IMV_OK == nRet:
                    rgbBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)
                    memmove(rgbBuff, stPixelConvertParam.pDstBuf,
                            stPixelConvertParam.nDstBufSize)
                    colorByteArray = bytearray(rgbBuff)
                    cvImage = np.array(colorByteArray).reshape(
                        stPixelConvertParam.nHeight, stPixelConvertParam.nWidth, 3)
                    self.current_frame = cvImage
                    if None != pDstBuf:
                        del pDstBuf

                self.cam.IMV_ReleaseFrame(frame)

        self.cam.IMV_StopGrabbing()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._grabbing_thread)
        self.thread.start()

    def stop(self):
        # Chống gọi stop() nhiều lần
        if not self.is_running and self.thread is None:
            return
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.cam.IMV_Close()


# ---------------- Quản lý đường dẫn lưu ảnh ----------------

BASE_DIR = r"D:\TongHop\RTC Technologi\G8\dataset\OOD"


def make_save_dir(part_num):
    """Trả về đường dẫn folder PartN, tạo nếu chưa có."""
    path = os.path.join(BASE_DIR, f"Part{part_num}")
    os.makedirs(path, exist_ok=True)
    return path


def next_index(folder):
    """Đếm số ảnh .jpg đã có trong folder để đánh số tiếp, không ghi đè."""
    existing = [f for f in os.listdir(folder)
                if f.lower().endswith(".jpg")]
    return len(existing)


if __name__ == '__main__':
    my_cam = None
    try:
        my_cam = CameraIndustrial(index=0)

        if my_cam.open():
            my_cam.start()

            part_num = 1
            save_dir = make_save_dir(part_num)
            img_count = next_index(save_dir)
            print(f"[FOLDER] Đang lưu vào: {save_dir}")
            print("Bấm 'c' để chụp, 'f' để đổi folder, 'q' để thoát.")

            while True:
                frame = my_cam.current_frame
                if frame is not None:
                    height, width = frame.shape[0], frame.shape[1]

                    # Hiển thị bản resize, KHÔNG động vào frame gốc
                    disp = cv2.resize(frame, (width // 4, height // 4))
                    cv2.imshow("Industrial Cam Feed", disp)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('c'):
                    if frame is not None:
                        filename = os.path.join(
                            save_dir, f"img_{img_count:04d}.jpg")
                        cv2.imwrite(filename, frame)   # lưu ảnh gốc full-res
                        print(f"[SAVED] {filename}")
                        img_count += 1
                    else:
                        print("[WARN] Chưa có frame để chụp.")

                elif key == ord('f'):
                    part_num += 1
                    save_dir = make_save_dir(part_num)
                    img_count = next_index(save_dir)
                    print(f"[FOLDER] Chuyển sang: {save_dir}")

            my_cam.stop()
            cv2.destroyAllWindows()
        else:
            print("Không tìm thấy camera.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if my_cam is not None:
            my_cam.stop()
        cv2.destroyAllWindows()