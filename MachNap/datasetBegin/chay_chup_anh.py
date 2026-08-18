r"""
chay_chup_anh.py - CHẾ ĐỘ CHỤP ẢNH.

Mở camera Hikrobot xem trực tiếp. Nhấn 'c' để chụp: ảnh chạy qua đủ các
bước crop + mask (xu_ly_anh.py) rồi mới hiện lên cho vẽ ROI. Vẽ xong bấm
Enter là ảnh tự cắt part và chia vào dataset.

    python chay_chup_anh.py

Ảnh crop được lưu vào THU_MUC_ANH_FULL, nên hôm sau mở chay_doc_file.py là
vẽ tiếp / sửa lại được.

Cần: thư mục MvImport nằm CÙNG CẤP với file này, và MVS đã cài đúng
DLL_DIR_MVS trong cauhinh.py.
"""

import os
import sys
import time

import cv2
import numpy as np

import cauhinh as cf
import chia_part as cp
import giao_dien as gd
import kho_dulieu as kd
import xu_ly_anh as xl


def nap_sdk():
    """Import SDK Hikrobot (để trong hàm cho máy không cài MVS vẫn import
    được module này)."""
    if not os.path.isdir(cf.DLL_DIR_MVS):
        raise RuntimeError(
            f"Không thấy thư mục DLL của MVS: {cf.DLL_DIR_MVS}\n"
            "-> Cài MVS, hoặc sửa DLL_DIR_MVS trong cauhinh.py.")
    os.add_dll_directory(cf.DLL_DIR_MVS)

    thu_muc = cf.DUONG_DAN_MVIMPORT or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "MvImport")
    if not os.path.isfile(os.path.join(thu_muc, "MvCameraControl_class.py")):
        raise RuntimeError(
            f"Không thấy MvCameraControl_class.py trong: {thu_muc}\n"
            "-> Sửa DUONG_DAN_MVIMPORT trong cauhinh.py cho trỏ đúng thư mục "
            "MvImport, hoặc copy thư mục MvImport vào cạnh chay_chup_anh.py.")

    sys.path.insert(0, thu_muc)
    import MvCameraControl_class as mv
    return mv


def mo_camera(mv):
    """Trả về (cam, payload_size) hoặc (None, 0)."""
    print("SDK Version:", hex(mv.MvCamera.MV_CC_GetSDKVersion()))
    ds = mv.MV_CC_DEVICE_INFO_LIST()
    ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE, ds)
    if ret != 0 or ds.nDeviceNum == 0:
        print("Không tìm thấy camera nào! Mã lỗi:", ret)
        return None, 0
    print(f"Tìm thấy {ds.nDeviceNum} camera")

    chon = 0
    for i in range(ds.nDeviceNum):
        info = mv.cast(ds.pDeviceInfo[i], mv.POINTER(mv.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == mv.MV_GIGE_DEVICE:
            ten = "".join(chr(c) for c in info.SpecialInfo.stGigEInfo.chModelName if c)
            print(f"[{i}] GigE Camera: {ten}")
            if cf.TEN_MODEL_CAMERA in ten:
                chon = i
        elif info.nTLayerType == mv.MV_USB_DEVICE:
            ten = "".join(chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c)
            print(f"[{i}] USB Camera: {ten}")

    info = mv.cast(ds.pDeviceInfo[chon], mv.POINTER(mv.MV_CC_DEVICE_INFO)).contents
    cam = mv.MvCamera()
    if cam.MV_CC_CreateHandle(info) != 0:
        print("Tạo handle lỗi")
        return None, 0
    if cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0) != 0:
        print("Mở camera lỗi")
        cam.MV_CC_DestroyHandle()
        return None, 0

    if info.nTLayerType == mv.MV_GIGE_DEVICE:
        n = cam.MV_CC_GetOptimalPacketSize()
        if n > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", n)
    cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF)

    if cam.MV_CC_StartGrabbing() != 0:
        print("Start grabbing lỗi")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None, 0

    p = mv.MVCC_INTVALUE()
    mv.memset(mv.byref(p), 0, mv.sizeof(mv.MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", p)
    return cam, p.nCurValue


def frame_sang_bgr(mv, raw, pixel_type):
    bang = {
        mv.PixelType_Gvsp_BayerRG8: cv2.COLOR_BAYER_RG2BGR,
        mv.PixelType_Gvsp_BayerGB8: cv2.COLOR_BAYER_GB2RGB,
        mv.PixelType_Gvsp_BayerGR8: cv2.COLOR_BAYER_GR2BGR,
        mv.PixelType_Gvsp_BayerBG8: cv2.COLOR_BAYER_BG2BGR,
        mv.PixelType_Gvsp_Mono8: cv2.COLOR_GRAY2BGR,
    }
    ma = bang.get(pixel_type)
    return None if ma is None else cv2.cvtColor(raw, ma)


def main():
    nhan_all = kd.doc_nhan()

    # kích thước ảnh crop suy ra từ mask -> biết cách chia trước khi chụp
    w, h = xl.kich_thuoc_output_chuan()
    cp.thiet_lap(w, h)
    kd.tao_san_thu_muc()
    cp.canh_bao_doi_cach_chia()
    cp.luu_cau_hinh()
    os.makedirs(cf.THU_MUC_ANH_FULL, exist_ok=True)

    dem = kd.dem_hien_co()
    print(f"\nDataset hiện có (train/val = {cf.TI_LE_TRAIN:.0%}/"
          f"{1 - cf.TI_LE_TRAIN:.0%}, ngưỡng box = {cf.TI_LE_BOX_TOI_THIEU:.2f}):")
    kd.in_thong_ke(dem)

    print("Đang chuẩn bị ảnh chuẩn (dự phòng cho nhánh SIFT)...")
    anh_chuan = xl.chuan_bi_anh_chuan()
    print(f"  Ảnh chuẩn có tỉ lệ lõm {anh_chuan['ti_le_lom']:.1f}\n")

    mv = nap_sdk()
    cam, payload = mo_camera(mv)
    if cam is None:
        return

    frame_info = mv.MV_FRAME_OUT_INFO_EX()
    mv.memset(mv.byref(frame_info), 0, mv.sizeof(frame_info))
    buf = (mv.c_ubyte * payload)()

    CUA_SO_LIVE = "Camera - 'c' chup & ve ROI, 'r' dung lai dataset, 'q' thoat"
    print("Đang xem trực tiếp... 'c' = chụp + vẽ ROI, 'q' = thoát.")

    so_thu_tu, ket_thuc = 0, "thoat"
    try:
        while True:
            if cam.MV_CC_GetOneFrameTimeout(buf, payload, frame_info, 1000) != 0:
                continue
            raw = np.frombuffer(buf, dtype=np.uint8, count=frame_info.nFrameLen)
            raw = raw.reshape(frame_info.nHeight, frame_info.nWidth)
            img = frame_sang_bgr(mv, raw, frame_info.enPixelType)
            if img is None:
                print("Pixel format chưa được xử lý:", frame_info.enPixelType)
                continue

            cv2.imshow(CUA_SO_LIVE, cv2.resize(img, (960, 640)))
            phim = cv2.waitKey(1) & 0xFF

            if phim in (ord('q'), 27):
                break
            if phim == ord('r'):
                ket_thuc = "dung_lai"
                break
            if phim != ord('c'):
                continue

            # ---- chụp: crop + mask rồi mới cho vẽ ROI ----
            so_thu_tu += 1
            crop, thong_bao, mask_align = xl.crop_va_dong_bo_huong(
                img, anh_chuan, tra_mask=True)
            if crop is None:
                print(f"\n[Chụp #{so_thu_tu}] KHÔNG crop được: {thong_bao}")
                continue
            print(f"\n[Chụp #{so_thu_tu}] {thong_bao}")

            ten_file = (f"capture_{time.strftime('%Y%m%d_%H%M%S')}"
                        f"_{so_thu_tu:03d}_crop.png")
            cv2.imwrite(os.path.join(cf.THU_MUC_ANH_FULL, ten_file), crop)
            print(f"  -> Đã lưu ảnh crop: {ten_file}")

            if cf.LUU_MASK_ALIGN and mask_align is not None:
                os.makedirs(cf.THU_MUC_MASK_ALIGN, exist_ok=True)
                cv2.imwrite(os.path.join(cf.THU_MUC_MASK_ALIGN, ten_file),
                            mask_align)

            hd = gd.ve_roi_mot_anh(crop, ten_file, nhan_all, dem,
                                   f"[Chup #{so_thu_tu}] {ten_file}",
                                   cho_dieu_huong=False)
            cv2.destroyWindow(gd.TEN_CUA_SO)
            if hd in ("dung_lai", "thoat"):
                ket_thuc = hd
                break
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("\nĐã đóng camera.")

    if ket_thuc == "dung_lai":
        dem = kd.xuat_lai_toan_bo(nhan_all)
    else:
        print("Kết quả dataset:")
        kd.in_thong_ke(dem)
    print(f"Ảnh crop lưu tại  : {cf.THU_MUC_ANH_FULL}")
    print(f"Nhãn lưu tại      : {cf.FILE_NHAN}")


if __name__ == "__main__":
    main()
