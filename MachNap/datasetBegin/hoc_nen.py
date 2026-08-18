r"""
hoc_nen.py - Chụp nền trống rồi lưu lại, để tim_vat_the.py trừ nền.

    python hoc_nen.py                 -> học từ CAMERA (dọn trống bàn trước)
    python hoc_nen.py D:\anh_nen      -> học từ thư mục ảnh nền đã chụp sẵn

Quy trình khi học từ camera:
    1. Dọn SẠCH vùng chụp - không vật, không tay, không giấy tờ. Đồ gá cố
       định thì CỨ ĐỂ NGUYÊN (nó là một phần của nền).
    2. Chờ warmup cho camera ổn định phơi sáng/cân bằng trắng.
    3. Gom BG_SO_FRAME frame, lấy TRUNG VỊ.
    4. Tự kiểm tra: so nền vừa học với 1 frame mới. Lệch > BG_MAX_RESIDUAL
       nghĩa là cảnh chưa đứng yên (đèn nhấp nháy, có người đi qua...) ->
       học lại.

PHẢI HỌC LẠI NỀN KHI: đổi đèn, đụng vào camera, đổi đồ gá, đổi vị trí đặt
vật, hoặc ánh sáng phòng đổi nhiều (sáng/tối trong ngày).
Kiểm tra nhanh xem nền còn dùng được không:
    python tim_vat_the.py mot_anh_moi_chup.png
(nó in ra độ lệch so với nền)
"""

import os
import sys

import cv2
import numpy as np

import cauhinh as cf
import tim_vat_the as tv


def _hien(img, chu, mau=(0, 255, 0)):
    xem = img.copy()
    ti_le = min(1.0, 1100 / max(xem.shape[:2]))
    if ti_le < 1.0:
        xem = cv2.resize(xem, None, fx=ti_le, fy=ti_le)
    cv2.putText(xem, chu, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mau, 2)
    cv2.imshow("Hoc nen - 'q' de huy", xem)
    return cv2.waitKey(1) & 0xFF


def hoc_tu_thu_muc(thu_muc):
    ten = sorted(f for f in os.listdir(thu_muc)
                 if f.lower().endswith(cf.VALID_EXT))
    if not ten:
        raise SystemExit(f"Không có ảnh nào trong: {thu_muc}")
    print(f"Đọc {len(ten)} ảnh nền từ {thu_muc}")
    frames = []
    for t in ten:
        img = cv2.imread(os.path.join(thu_muc, t))
        if img is None:
            print(f"  bỏ qua (không đọc được): {t}")
            continue
        if frames and img.shape != frames[0].shape:
            print(f"  bỏ qua (khác kích thước): {t}")
            continue
        frames.append(img)
    if not frames:
        raise SystemExit("Không đọc được ảnh nền nào.")
    print(f"Dùng {len(frames)} ảnh, lấy trung vị...")
    return tv.hoc_nen_tu_frames(frames)


def hoc_tu_camera():
    import chay_chup_anh as cc      # dùng lại phần mở camera đã có

    mv = cc.nap_sdk()
    cam, payload = cc.mo_camera(mv)
    if cam is None:
        raise SystemExit("Không mở được camera.")

    frame_info = mv.MV_FRAME_OUT_INFO_EX()
    mv.memset(mv.byref(frame_info), 0, mv.sizeof(frame_info))
    buf = (mv.c_ubyte * payload)()

    def lay_frame():
        if cam.MV_CC_GetOneFrameTimeout(buf, payload, frame_info, 1000) != 0:
            return None
        raw = np.frombuffer(buf, dtype=np.uint8, count=frame_info.nFrameLen)
        raw = raw.reshape(frame_info.nHeight, frame_info.nWidth)
        return cc.frame_sang_bgr(mv, raw, frame_info.enPixelType)

    try:
        while True:
            print(f"\n*** DỌN TRỐNG vùng chụp (đồ gá cố định thì để nguyên). ***")
            print(f"Warmup {cf.BG_WARMUP} frame...")
            i = 0
            while i < cf.BG_WARMUP:
                img = lay_frame()
                if img is None:
                    continue
                i += 1
                if _hien(img, f"Warming up {i}/{cf.BG_WARMUP}", (0, 165, 255)) == ord('q'):
                    raise SystemExit("Đã huỷ.")

            print(f"Gom {cf.BG_SO_FRAME} frame nền...")
            frames = []
            while len(frames) < cf.BG_SO_FRAME:
                img = lay_frame()
                if img is None:
                    continue
                frames.append(img)
                if _hien(img, f"Learning BG {len(frames)}/{cf.BG_SO_FRAME}") == ord('q'):
                    raise SystemExit("Đã huỷ.")

            bg = tv.hoc_nen_tu_frames(frames)

            # tự kiểm tra bằng 1 frame mới
            kiem = None
            while kiem is None:
                kiem = lay_frame()
            lech = tv.do_lech_nen(kiem, bg)
            print(f"Độ lệch nền vs frame mới: {lech:.1f} "
                  f"(ngưỡng {cf.BG_MAX_RESIDUAL})")
            if lech < cf.BG_MAX_RESIDUAL:
                return bg
            print("Cảnh chưa ổn định (đèn nhấp nháy? có người đi qua?) -> học lại.")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1:
        bg = hoc_tu_thu_muc(sys.argv[1])
    else:
        bg = hoc_tu_camera()

    tv.luu_nen(bg)
    png = os.path.splitext(cf.FILE_NEN)[0] + ".png"
    print(f"\nĐã lưu nền: {cf.FILE_NEN}")
    print(f"           {png}  (mở ra soi bằng mắt xem có sạch không)")
    print(f"Kích thước: {bg.shape[1]}x{bg.shape[0]}")
    print("\nKiểm tra thử:  python tim_vat_the.py <ảnh có vật>.png")


if __name__ == "__main__":
    main()
