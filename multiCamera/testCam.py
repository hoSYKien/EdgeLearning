"""
CHẨN ĐOÁN RIÊNG KEA (IMVApi)
=============================
Liệt kê những camera mà SDK KEA nhìn thấy + thử grab từng index.
Chạy: python test_kea.py

>>> ĐÓNG cam Hikrobot (không chạy run_multicam cùng lúc) khi test cái này.
"""
import os, sys, time
_THIS = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "IMVApi", "MvImport"):
    p = os.path.join(_THIS, sub) if sub else _THIS
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2
from ctypes import byref, c_void_p, memset, memmove, sizeof, c_ubyte, c_buffer
from IMVApi import *


def list_devices():
    dl = IMV_DeviceList()
    MvCamera.IMV_EnumDevices(dl, IMV_EInterfaceType.interfaceTypeAll)
    print(f"IMVApi thấy {dl.nDevNum} camera:")
    for i in range(dl.nDevNum):
        info = dl.pDevInfo[i]
        # thử lấy tên/serial (tên field có thể khác tuỳ SDK)
        fields = {}
        for attr in ("cameraName", "modelName", "serialNumber",
                     "cameraKey", "vendorName"):
            try:
                v = getattr(info, attr)
                if isinstance(v, bytes):
                    v = v.decode(errors="ignore")
                fields[attr] = str(v).strip("\x00")
            except Exception:
                pass
        print(f"  [{i}] {fields}")
    return dl.nDevNum


def try_index(idx):
    print(f"\n--- Thử mở index {idx} ---")
    cam = MvCamera()
    r = cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex,
                             byref(c_void_p(idx)))
    print(f"  CreateHandle ret={r}")
    r = cam.IMV_Open()
    print(f"  Open ret={r}")
    r = cam.IMV_StartGrabbing()
    print(f"  StartGrabbing ret={r}")

    got = False
    t0 = time.time()
    while time.time() - t0 < 5:
        frame = IMV_Frame()
        if cam.IMV_GetFrame(frame, 1000) == IMV_OK:
            print(f"  ✅ GetFrame OK: {frame.frameInfo.width}x{frame.frameInfo.height} "
                  f"pixelFormat={frame.frameInfo.pixelFormat}")
            cam.IMV_ReleaseFrame(frame)
            got = True
            break
        time.sleep(0.1)
    if not got:
        print("  ❌ Không GetFrame được trong 5s")
    try:
        cam.IMV_StopGrabbing(); cam.IMV_Close()
    except Exception:
        pass
    return got


if __name__ == "__main__":
    n = list_devices()
    for i in range(n):
        try_index(i)
    print("\nXong. Cho biết index nào ✅ ra frame, và pixelFormat của nó.")