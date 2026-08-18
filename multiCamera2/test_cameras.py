# -*- coding: utf-8 -*-
"""
TOOL CHẨN ĐOÁN & KIỂM TRA TỐC ĐỘ 2 - 4 CAMERA GIGE / USB
=========================================================
Chạy:
    python test_cameras.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from camera_driver import discover_cameras, build_all_cameras


def main():
    print("=" * 60)
    print("KIỂM TRA & CHẨN ĐOÁN TOÀN BỘ CAMERA CÔNG NGHIỆP")
    print("=" * 60)

    devices = discover_cameras()
    if not devices:
        print("Không tìm thấy camera nào.")
        return

    print(f"Tìm thấy {len(devices)} camera:")
    for d in devices:
        print(f"  [{d['index']}] {d['type']} - {d['model']} | IP: {d['ip']} | SN: {d['serial']}")

    print("\nĐang mở và stream thử trong 5 giây...")
    cams = build_all_cameras(max_cameras=4)
    opened = {}
    for name, cam in cams.items():
        if cam.open():
            if cam.start():
                opened[name] = cam
            else:
                cam.stop()

    if not opened:
        print("Không mở được camera nào.")
        return

    # Stream 5s
    t0 = time.time()
    while time.time() - t0 < 5.0:
        time.sleep(0.5)
        for name, cam in opened.items():
            frame, fid, fps = cam.get_frame_with_info()
            status = "OK" if frame is not None else "WAITING"
            shape = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "None"
            print(f"  [{name}] {status} | Frame: {fid} | FPS: {fps:.1f} | Res: {shape}")

    print("\nDừng và đóng camera...")
    for cam in opened.values():
        cam.stop()
    print("Xong!")


if __name__ == "__main__":
    main()
