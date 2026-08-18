import threading
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

    def open(self):
        # 1. Tìm thiết bị
        deviceList = IMV_DeviceList()
        print("Enumerating devices...")
        MvCamera.IMV_EnumDevices(
            deviceList, IMV_EInterfaceType.interfaceTypeAll)
        if deviceList.nDevNum == 0:
            return False
        # print(f"Found {deviceList.nDevNum} device(s).")

        # 2. Tạo handle và Mở
        self.cam.IMV_CreateHandle(
            IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(self.index)))
        self.cam.IMV_Open()

        # 3. Cài đặt thông số (Tắt trigger để lấy ảnh liên tục)
        self.cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
        
        # 4. White Balance (Tự động cân bằng trắng)
        self.cam.IMV_SetEnumFeatureSymbol("BalanceWhiteAuto", "Once")
        # Edit Exposure/Gain ở đây
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

            # print(f"GetFrame return code: {nRet}")
            if nRet == IMV_OK:
                # Assign values to the parameters required for transcoding
                if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
                    print("Mono8 format detected")
                    nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
                else:
                    # print(f"Non-Mono8 format detected: {frame.frameInfo.pixelFormat}")
                    nDstBufSize = frame.frameInfo.width * frame.frameInfo.height*3

                pDstBuf = (c_ubyte*nDstBufSize)()
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
                # stPixelConvertParam.nDstBufSize=nDstBufSize
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
                
                # # Logic chuyển đổi nhanh sang Numpy (Dành cho ảnh Mono8)
                # img_data = cast(frame.pData, POINTER(
                #     c_ubyte * frame.frameInfo.size)).contents
                # self.current_frame = np.frombuffer(img_data, count=frame.frameInfo.size, dtype=np.uint8).reshape(
                #     frame.frameInfo.height, frame.frameInfo.width)

                # Giải phóng frame ngay lập tức
                self.cam.IMV_ReleaseFrame(frame)

        self.cam.IMV_StopGrabbing()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._grabbing_thread)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.cam.IMV_Close()


if __name__ == '__main__':
    # Khởi tạo
    try:
        my_cam = CameraIndustrial(index=0)

        if my_cam.open():
            my_cam.start()

            while True:
                frame = my_cam.current_frame
                if frame is not None:
                    width, height = frame.shape[1], frame.shape[0]
                    # --- ĐÂY LÀ NƠI BẠN XỬ LÝ ẢNH ---
                    # Ví dụ: Đưa vào model nhận diện, lưu file, v.v.
                    frame = cv2.resize(frame, (width//4, height//4))  # Resize nếu cần
                    cv2.imshow("Industrial Cam Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            my_cam.stop()
            cv2.destroyAllWindows()
    except Exception as e:
        #thoat khi có lỗi xảy ra
        my_cam.stop()
        cv2.destroyAllWindows()
        print(f"An error occurred: {e}")
    finally:
        # Đảm bảo giải phóng tài nguyên
        if 'my_cam' in locals():
            my_cam.stop()
        cv2.destroyAllWindows()
