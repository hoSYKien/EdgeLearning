# -- coding: utf-8 --

import sys
import platform
import os
import numpy as np
import cv2
from ctypes import *

# Initialize path for MvImport depending on platform
currentsystem = platform.system()
if currentsystem == 'Windows':
    mv_import_path = os.getenv('MVCAM_COMMON_RUNENV') + "\\Samples\\Python\\MvImport"
    if mv_import_path not in sys.path:
        sys.path.append(mv_import_path)
else:
    # Demo directory relative path for non-Windows (Linux etc.)
    sys.path.append("./../../../Python/MvImport")

try:
    from MvCameraControl_class import *
except ImportError as e:
    print(f"Warning: Could not import MvCameraControl_class. Make sure MVS SDK is installed. Details: {e}")

# Helper functions
def IsHBPixelFormat(enPixelType=0):
    if enPixelType in (PixelType_Gvsp_HB_Mono8, \
                        PixelType_Gvsp_HB_Mono10,\
                        PixelType_Gvsp_HB_Mono10_Packed,\
                        PixelType_Gvsp_HB_Mono12,\
                        PixelType_Gvsp_HB_Mono12_Packed,\
                        PixelType_Gvsp_HB_Mono16,\
                        PixelType_Gvsp_HB_RGB8_Packed,\
                        PixelType_Gvsp_HB_BGR8_Packed,\
                        PixelType_Gvsp_HB_RGBA8_Packed,\
                        PixelType_Gvsp_HB_BGRA8_Packed,\
                        PixelType_Gvsp_HB_RGB16_Packed,\
                        PixelType_Gvsp_HB_BGR16_Packed,\
                        PixelType_Gvsp_HB_RGBA16_Packed,\
                        PixelType_Gvsp_HB_BGRA16_Packed,\
                        PixelType_Gvsp_HB_YUV422_Packed,\
                        PixelType_Gvsp_HB_YUV422_YUYV_Packed,\
                        PixelType_Gvsp_HB_BayerGR8,\
                        PixelType_Gvsp_HB_BayerRG8,\
                        PixelType_Gvsp_HB_BayerGB8,\
                        PixelType_Gvsp_HB_BayerBG8,\
                        PixelType_Gvsp_HB_BayerRBGG8,\
                        PixelType_Gvsp_HB_BayerGB10,\
                        PixelType_Gvsp_HB_BayerGB10_Packed,\
                        PixelType_Gvsp_HB_BayerBG10,\
                        PixelType_Gvsp_HB_BayerBG10_Packed,\
                        PixelType_Gvsp_HB_BayerRG10,\
                        PixelType_Gvsp_HB_BayerRG10_Packed,\
                        PixelType_Gvsp_HB_BayerGR10,\
                        PixelType_Gvsp_HB_BayerGR10_Packed,\
                        PixelType_Gvsp_HB_BayerGB12,\
                        PixelType_Gvsp_HB_BayerGB12_Packed,\
                        PixelType_Gvsp_HB_BayerBG12,\
                        PixelType_Gvsp_HB_BayerBG12_Packed,\
                        PixelType_Gvsp_HB_BayerRG12,\
                        PixelType_Gvsp_HB_BayerRG12_Packed,\
                        PixelType_Gvsp_HB_BayerGR12,\
                        PixelType_Gvsp_HB_BayerGR12_Packed):
        return True
    else:
        return False

def IsMonoPixelFormat(enPixelType=0):
    if enPixelType in (PixelType_Gvsp_Mono8, \
                        PixelType_Gvsp_Mono10, \
                        PixelType_Gvsp_Mono10_Packed, \
                        PixelType_Gvsp_Mono12, \
                        PixelType_Gvsp_Mono12_Packed, \
                        PixelType_Gvsp_Mono14, \
                        PixelType_Gvsp_Mono16):
        return True
    else:
        return False

class KowaCamera:
    """
    A class to manage the lifecycle and image acquisition of a Kowa camera.
    Can be used as a context manager.
    """
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cam = None
        self.is_opened = False
        self.is_grabbing = False
        self._sdk_initialized = False

    def open(self):
        """Initializes the SDK, connects to the device, and configures settings."""
        if self.is_opened:
            return True

        # Initialize SDK
        try:
            MvCamera.MV_CC_Initialize()
            self._sdk_initialized = True
        except Exception as e:
            print(f"Error: Failed to initialize SDK: {e}")
            return False

        # Enum devices
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                      | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"Error: Enum devices fail! ret[0x{ret:x}]")
            self.close()
            return False

        if deviceList.nDeviceNum == 0:
            print("Error: No camera devices found!")
            self.close()
            return False

        if self.device_index >= deviceList.nDeviceNum:
            print(f"Error: Device index {self.device_index} is out of range. Found {deviceList.nDeviceNum} devices.")
            self.close()
            return False

        # Create camera object
        self.cam = MvCamera()
        stDeviceInfo = cast(deviceList.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceInfo)
        if ret != 0:
            print(f"Error: Create handle fail! ret[0x{ret:x}]")
            self.cam = None
            self.close()
            return False

        # Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"Error: Open device fail! ret[0x{ret:x}]")
            self.close()
            return False
        
        self.is_opened = True

        # Optimal packet size (for GigE)
        if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE or stDeviceInfo.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        # # Set trigger mode to OFF
        # ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # if ret != 0:
        #     print(f"Error: Set trigger mode fail! ret[0x{ret:x}]")
        #     self.close()
        #     return False

        # Start grabbing
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"Error: Start grabbing fail! ret[0x{ret:x}]")
            self.close()
            return False
        
        self.is_grabbing = True
        return True

    def get_image(self, timeout_ms=1000):
        """
        Grabs one frame, decodes and converts it to a OpenCV BGR or Mono8 image.
        Returns:
            numpy.ndarray: The image array or None if failed.
        """
        if not self.is_grabbing:
            if not self.open():
                return None

        stOutFrame = MV_FRAME_OUT()  
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
       
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
        if stOutFrame.pBufAddr is None or ret != 0:
            print(f"Error: Get image buffer fail! ret[0x{ret:x}]")
            return None

        try:
            stDecodeParam = MV_CC_HB_DECODE_PARAM()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            
            # HB Decode if needed
            is_hb = IsHBPixelFormat(stOutFrame.stFrameInfo.enPixelType)
            if is_hb:
                DecodeNeedBufferlen = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
                DecodeBuffer = (c_ubyte * DecodeNeedBufferlen)()
                
                stDecodeParam.pSrcBuf = stOutFrame.pBufAddr
                stDecodeParam.nSrcLen = stOutFrame.stFrameInfo.nFrameLen
                stDecodeParam.pDstBuf = DecodeBuffer
                stDecodeParam.nDstBufSize = DecodeNeedBufferlen
                ret = self.cam.MV_CC_HBDecode(stDecodeParam)
                if ret != 0:
                    print(f"Error: HB Decode fail! ret[0x{ret:x}]")
                    return None

                stConvertParam.pSrcData = stDecodeParam.pDstBuf
                stConvertParam.nSrcDataLen = stDecodeParam.nDstBufLen
                stConvertParam.enSrcPixelType = stDecodeParam.enDstPixelType  
            else:
                stConvertParam.pSrcData = stOutFrame.pBufAddr
                stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType  
            
            # Convert pixel type
            bMono = IsMonoPixelFormat(stConvertParam.enSrcPixelType)
            if bMono:
                enDstPixelType = PixelType_Gvsp_Mono8
                nChannelNum = 1
            else:
                enDstPixelType = PixelType_Gvsp_RGB8_Packed
                nChannelNum = 3
            
            convertdestBufflen = nChannelNum * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight 
            DstBuffer = (c_ubyte * convertdestBufflen)()
            
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stConvertParam.enDstPixelType = enDstPixelType
            stConvertParam.pDstBuffer = DstBuffer
            stConvertParam.nDstBufferSize = convertdestBufflen

            ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
            if ret != 0:
                print(f"Error: Convert pixel fail! ret[0x{ret:x}]")
                return None
        
            # Convert to numpy array
            if nChannelNum == 1:
                numpy_image = np.frombuffer(DstBuffer, dtype=np.ubyte, count=convertdestBufflen).reshape(
                    stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth
                )
            else:
                numpy_image = np.frombuffer(DstBuffer, dtype=np.ubyte, count=convertdestBufflen).reshape(
                    stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, 3
                )
                h, w = numpy_image.shape[:2]
                # Convert RGB to BGR for OpenCV
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                
            return numpy_image

        finally:
            self.cam.MV_CC_FreeImageBuffer(stOutFrame)

    def close(self):
        """Closes connection to camera and cleans up SDK resources."""
        if self.cam:
            if self.is_grabbing:
                self.cam.MV_CC_StopGrabbing()
                self.is_grabbing = False
            if self.is_opened:
                self.cam.MV_CC_CloseDevice()
                self.is_opened = False
            self.cam.MV_CC_DestroyHandle()
            self.cam = None

        if self._sdk_initialized:
            try:
                MvCamera.MV_CC_Finalize()
            except:
                pass
            self._sdk_initialized = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_image(device_index=0, timeout_ms=1000):
    """
    Helper function to instantly open the camera, grab one frame, and close it.
    Returns:
        numpy.ndarray: The captured image (Mono8 or BGR), or None if failed.
    """
    with KowaCamera(device_index=device_index) as cam:
        return cam.get_image(timeout_ms)


# Demonstration code when run as script
if __name__ == "__main__":
    print("Testing get_data.py module...")
    
    # 1. Test single-shot function
    print("Testing single-shot function get_image()...")
    img = get_image(0)
    if img is not None:
        print(f"Successfully grabbed single image of shape: {img.shape}")
        cv2.imwrite("test_single_shot.bmp", img)
        print("Saved test_single_shot.bmp")
    else:
        print("Could not grab single image (check camera connection).")

    # 2. Test context manager / class usage
    print("\nTesting class usage / live display (Press ESC to exit)...")
    with KowaCamera(0) as cam:
        if cam.open():
            print("Camera is open and grabbing. Displaying stream...")
            while True:
                img = cam.get_image()
                if img is not None:
                    # Apply resize/display logic from original GrabImage_Cv.py
                    h, w = img.shape[:2]
                    cv2.imshow("Kowa Camera Live Stream", img)
                    
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                else:
                    break
            cv2.destroyAllWindows()
        else:
            print("Failed to start stream (check camera connection).")
