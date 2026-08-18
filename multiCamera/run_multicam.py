"""
CHẠY ĐA CAMERA: đọc barcode + GHÉP ID CHUNG theo mặt bàn (homography)
====================================================================
Luồng:
  - Mỗi camera (Hikrobot / KEA) grab frame trong luồng riêng.
  - Với mỗi frame: trừ nền -> nhiều bbox vật -> mỗi vật lấy TÂM + đọc barcode.
  - Gửi (tâm, barcode, bbox) cho COORDINATOR.
  - Coordinator quy về mặt bàn, gộp chéo camera -> ID chung, điền barcode.
  - Hiển thị TỪNG camera với ID nhất quán + barcode.

CHUẨN BỊ:
  1) calibrate.py -> tạo homographies.json cho từng camera (chạy 1 lần).
  2) Sửa phần "TẠO CAMERA" bên dưới cho khớp phần cứng của bạn.

PHÍM: q/ESC thoát.

GHI CHÚ QUAN TRỌNG:
  - Phần đọc barcode ở đây dùng lại ý tưởng file live_barcode_threaded
    (gradient + cv2.barcode + decode nhiều engine). Để gọn, bản này gọi
    hàm read_barcode_in_region nếu bạn để chung thư mục; nếu không, thay
    bằng bộ đọc của bạn. Điểm mấu chốt của file này là GHÉP ID, không phải
    thuật toán đọc.
"""

import os
import sys
import time
import numpy as np
import cv2


# ==== ÉP ĐƯỜNG DẪN SDK NGAY TỪ ĐẦU (trước khi import cameras) ====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "MvImport", "IMVApi"):
    _p = os.path.join(_THIS_DIR, _sub) if _sub else _THIS_DIR
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
_HIK_DLL = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(_HIK_DLL):
    try:
        os.add_dll_directory(_HIK_DLL)
    except Exception:
        pass
# =================================================================

from coordinator import MultiCamCoordinator
from calibrate import load_homographies


# ============================================================
# TRỪ NỀN ĐƠN GIẢN -> NHIỀU BBOX VẬT (mỗi frame)
# ============================================================

RESIZE_DIV = 4
THRESH = 30
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.9


def learn_bg(cam, n=40, timeout=15.0):
    """
    Học nền: gom n frame. Có TIMEOUT + in tiến độ để KHÔNG treo vô tận
    nếu camera không trả frame.
    """
    frames = []
    t0 = time.time()
    last_print = 0
    while len(frames) < n:
        f = cam.read()
        if f is None:
            if time.time() - t0 > timeout:
                print(f"  [{cam.name}] KHÔNG lấy được frame sau {timeout:.0f}s "
                      f"(mới có {len(frames)}/{n}). Bỏ qua học nền cam này.")
                return None
            time.sleep(0.02)
            continue
        h, w = f.shape[:2]
        s = cv2.resize(f, (w // RESIZE_DIV, h // RESIZE_DIV))
        frames.append(cv2.GaussianBlur(s, (7, 7), 0).astype(np.float32))
        # in tiến độ mỗi 10 frame
        if len(frames) - last_print >= 10:
            print(f"  [{cam.name}] học nền {len(frames)}/{n}")
            last_print = len(frames)
    print(f"  [{cam.name}] học nền xong.")
    return np.mean(frames, axis=0)


def detect_objects(full_bgr, bg_float):
    """Trả list bbox (x1,y1,x2,y2) trên ẢNH GỐC cho MỌI vật khác nền."""
    h, w = full_bgr.shape[:2]
    small = cv2.resize(full_bgr, (w // RESIZE_DIV, h // RESIZE_DIV))
    blur = cv2.GaussianBlur(small, (7, 7), 0).astype(np.float32)
    diff = cv2.absdiff(blur, bg_float)
    b, g, r = cv2.split(diff)
    d = cv2.max(cv2.max(b, g), r)
    mask = (d > THRESH).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    area_img = small.shape[0] * small.shape[1]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if MIN_AREA_RATIO * area_img <= a <= MAX_AREA_RATIO * area_img:
            x, y, ww, hh = cv2.boundingRect(c)
            boxes.append((x * RESIZE_DIV, y * RESIZE_DIV,
                          (x + ww) * RESIZE_DIV, (y + hh) * RESIZE_DIV))
    return boxes


# ============================================================
# ĐỌC BARCODE TRONG 1 CROP (rút gọn - thay bằng bộ của bạn nếu muốn)
# ============================================================

_HAS_ZXING = False
try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:
    pass


def read_barcode(crop_bgr):
    """Trả chuỗi EAN-13 hợp lệ đầu tiên, hoặc None. (Bản gọn dùng zxing.)"""
    if not _HAS_ZXING or crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    for im in (cl, cv2.bitwise_not(cl), gray):
        try:
            for r in zxingcpp.read_barcodes(im):
                t = r.text
                if t and t.isdigit() and len(t) == 13:
                    return t
        except Exception:
            pass
    return None


# ============================================================
# XỬ LÝ 1 FRAME CỦA 1 CAMERA -> danh sách phát hiện thô
# ============================================================

def process_camera_frame(full_bgr, bg_float):
    """Trả list ( (cx,cy) tâm ảnh, barcode_or_None, bbox )."""
    dets = []
    for (x1, y1, x2, y2) in detect_objects(full_bgr, bg_float):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        crop = full_bgr[y1:y2, x1:x2]
        bc = read_barcode(crop)
        dets.append(((cx, cy), bc, (x1, y1, x2, y2)))
    return dets


# ============================================================
# MAIN
# ============================================================

def build_cameras():
    """
    >>> CHỈNH Ở ĐÂY cho khớp phần cứng của bạn. <<<
    Trả dict {cam_name: camera_object}. Tên phải KHỚP khoá trong homographies.json.
    """
    from cameras import HikrobotCamera, KeaCamera
    cams = {}
    # Dùng SERIAL để chắc chắn bắt đúng con MV-CS200 (không nhầm sang KEA).
    cams["cam1"] = HikrobotCamera(name="cam1", model_hint="MV-CS200",
                                  serial="DB0780533")
    # KEA-1200GC2E: chọn theo SERIAL để không nhầm sang con Hikrobot mà
    # IMVApi cũng liệt kê. packet_size=1500 vì NIC chưa bật jumbo frame.
    cams["cam2"] = KeaCamera(name="cam2", model_hint="KEA",
                             serial="7L0552DPAK00056", packet_size=1500)
    return cams


def main():
    if not os.path.exists("homographies.json"):
        print("Chưa có homographies.json - hãy chạy calibrate.py trước.")
        return
    H = load_homographies()
    print("Đã nạp homography cho:", list(H.keys()))

    cams = build_cameras()
    opened = {}
    for name, cam in cams.items():
        try:
            if not cam.open():
                print(f"Không mở được {name} - bỏ qua."); continue
        except Exception as e:
            print(f"Lỗi mở {name}: {e} - bỏ qua."); continue
        cam.start()
        # chờ frame đầu (nới 8s)
        got = False
        for _ in range(160):
            if cam.read() is not None:
                got = True; break
            time.sleep(0.05)
        if got:
            print(f"[{name}] đã ra ảnh.")
            opened[name] = cam
        else:
            print(f"[{name}] mở được nhưng KHÔNG ra ảnh sau 8s -> bỏ qua.")
            print(f"        (kiểm tra: đang bị MVS/IMV Viewer chiếm? cáp/nguồn?)")
            cam.stop()

    if not opened:
        print("Không camera nào ra ảnh. Dừng.")
        return
    cams = opened

    print("\nHọc nền (giữ khung TRỐNG)...")
    bg = {}
    for name, cam in list(cams.items()):
        b = learn_bg(cam)
        if b is None:
            print(f"[{name}] học nền thất bại -> loại khỏi phiên chạy.")
            cam.stop()
            del cams[name]
        else:
            bg[name] = b

    if not cams:
        print("Không camera nào học nền được. Dừng.")
        return
    print("Xong. Đang chạy. q/ESC thoát.\n")

    coord = MultiCamCoordinator(
        H, merge_dist=8.0, track_dist=15.0, forget_after=1.0)

    try:
        while True:
            per_cam = {}
            frames = {}
            for name, cam in cams.items():
                f = cam.read()
                if f is None:
                    continue
                frames[name] = f
                per_cam[name] = process_camera_frame(f, bg[name])

            objs = coord.update(per_cam)

            # map: cam_name -> {bbox: (id, barcode)} để vẽ
            for name, f in frames.items():
                disp = f.copy()
                for o in objs:
                    if name in o.per_cam_bbox:
                        x1, y1, x2, y2 = o.per_cam_bbox[name]
                        col = (0, 255, 0) if o.barcode else (0, 200, 255)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 3)
                        txt = f"ID{o.id}"
                        if o.barcode:
                            txt += f" {o.barcode}"
                        cv2.putText(disp, txt, (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
                h, w = disp.shape[:2]
                cv2.imshow(name, cv2.resize(disp, (w // RESIZE_DIV, h // RESIZE_DIV)))

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()