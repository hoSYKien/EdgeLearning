"""
CHẠY CALIBRATE CHO 2 CAMERA THẬT
=================================
Trình tự:
  1) Dán 4 (hoặc nhiều hơn) điểm mốc lên mặt bàn, đo kích thước THẬT,
     rồi khai vào TABLE_POINTS trong calibrate.py.
  2) Đảm bảo CẢ 2 camera đều nhìn thấy cả 4 điểm đó.
  3) Chạy: python do_calibrate.py
     - Cửa sổ camera 1 hiện lên -> bấm 4 điểm ĐÚNG THỨ TỰ -> 's' để lưu.
     - Rồi tới camera 2 -> làm y hệt, bấm ĐÚNG 4 điểm vật lý ấy.
  4) Xong sẽ tạo homographies.json.

Sau đó chạy: python run_multicam.py
"""

# ==== ÉP ĐƯỜNG DẪN SDK NGAY TỪ ĐẦU (trước mọi import camera) ====
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# thêm MvImport (Hikrobot) + IMVApi (KEA) nằm cạnh file này
for _sub in ("", "MvImport", "IMVApi"):
    _p = os.path.join(_THIS_DIR, _sub) if _sub else _THIS_DIR
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# DLL runtime Hikrobot
_HIK_DLL = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
if os.path.exists(_HIK_DLL):
    try:
        os.add_dll_directory(_HIK_DLL)
    except Exception:
        pass
# ================================================================

from cameras import HikrobotCamera, KeaCamera
from calibrate import calibrate_one, save_homographies


def main():
    # >>> CHỈNH cho khớp phần cứng của bạn (tên phải khớp run_multicam.py) <<<
    cam_defs = [
        ("cam1", HikrobotCamera(name="cam1", model_hint="MV-CS200")),
        ("cam2", KeaCamera(name="cam2", model_hint="KEA",
                           serial="7L0552DPAK00056", packet_size=1500)),
    ]

    results = {}
    for name, cam in cam_defs:
        print(f"\n=== Mở {name} ===")
        if not cam.open():
            print(f"[{name}] mở KHÔNG được - bỏ qua.")
            continue
        cam.start()

        # chờ có frame đầu tiên
        import time
        for _ in range(50):
            if cam.read() is not None:
                break
            time.sleep(0.05)

        print(f"[{name}] Bấm {4} điểm mốc theo thứ tự, rồi 's' để lưu "
              f"(u=undo, q=bỏ qua cam này).")
        H = calibrate_one(name, cam.read)
        if H is not None:
            results[name] = H
            print(f"[{name}] OK - đã lấy homography.")
        else:
            print(f"[{name}] bỏ qua (không lưu).")
        cam.stop()

    if results:
        save_homographies(results)
        print(f"\nHOÀN TẤT: {list(results.keys())} -> homographies.json")
        print("Giờ chạy: python run_multicam.py")
    else:
        print("\nKhông có homography nào được lưu.")


if __name__ == "__main__":
    main()