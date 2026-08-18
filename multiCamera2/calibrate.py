# -*- coding: utf-8 -*-
"""
MODULE HIỆU CHUẨN HOMOGRAPHY 2D (ẢNH -> MẶT BÀN) TƯƠNG TÁC CHO TỪNG CAMERA
"""

import os
import sys
import json
import numpy as np
import cv2

# Thêm thư mục hiện tại vào sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import HOMOGRAPHY_FILE, CALIB_TABLE_POINTS


def load_homographies(filepath=HOMOGRAPHY_FILE):
    """Đọc ma trận Homography từ file JSON."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float64) for k, v in data.items()}
    except Exception as e:
        print(f"Lỗi đọc file homography {filepath}: {e}")
        return {}


def save_homographies(H_dict, filepath=HOMOGRAPHY_FILE):
    """Lưu ma trận Homography của tất cả camera vào file JSON."""
    data = {k: v.tolist() for k, v in H_dict.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"-> Đã lưu homography thành công vào: {filepath}")


def calibrate_camera_interactive(cam_name, get_frame_func):
    """Giao diện bấm 4 điểm mốc hiệu chuẩn 2D chuẩn xác 1:1 không lệch pixel."""
    frame = get_frame_func()
    if frame is None:
        print(f"[{cam_name}] Không lấy được frame để calibrate.")
        return None

    img_points = []
    h_orig, w_orig = frame.shape[:2]
    view_w, view_h = 1280, 720
    scale = min(view_w / w_orig, view_h / h_orig)

    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(img_points) < len(CALIB_TABLE_POINTS):
                orig_x = max(0.0, min(float(w_orig - 1), x / scale))
                orig_y = max(0.0, min(float(h_orig - 1), y / scale))
                idx = len(img_points)
                img_points.append((orig_x, orig_y))
                print(f"[{cam_name}] Điểm {idx+1}/{len(CALIB_TABLE_POINTS)}: Ảnh({orig_x:.1f}, {orig_y:.1f}) -> Bàn{CALIB_TABLE_POINTS[idx]}")

    win_name = f"CALIBRATE 2D [{cam_name}] - Bam 4 diem: Tren-Trai -> Tren-Phai -> Duoi-Phai -> Duoi-Trai"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 60)
    print(f"BẮT ĐẦU HIỆU CHUẨN 2D CHO CAMERA [{cam_name.upper()}]:")
    for idx, pt in enumerate(CALIB_TABLE_POINTS):
        print(f"  Điểm mốc {idx+1}: Tọa độ mặt bàn = {pt} cm")
    print("  Phím điều khiển:")
    print("    - 's': Lưu ma trận H khi đã click đủ 4 điểm")
    print("    - 'u': Undo điểm click gần nhất")
    print("    - 'r': Reset xóa làm lại từ đầu")
    print("    - 'q' hoặc ESC: Hủy bỏ / Bỏ qua camera này")
    print("=" * 60 + "\n")

    H_matrix = None
    disp_w = int(w_orig * scale)
    disp_h = int(h_orig * scale)

    while True:
        f = get_frame_func()
        if f is None:
            f = frame
        disp = cv2.resize(f, (disp_w, disp_h))

        # 1. Vẽ các điểm đã click
        for idx, (px, py) in enumerate(img_points):
            sx, sy = int(px * scale), int(py * scale)
            cv2.circle(disp, (sx, sy), 6, (0, 0, 255), -1)
            cv2.circle(disp, (sx, sy), 8, (255, 255, 255), 2)
            cv2.putText(disp, f"{idx+1}", (sx + 10, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # 2. Nối dây các điểm
        if len(img_points) > 1:
            for idx in range(len(img_points) - 1):
                p1 = (int(img_points[idx][0] * scale), int(img_points[idx][1] * scale))
                p2 = (int(img_points[idx+1][0] * scale), int(img_points[idx+1][1] * scale))
                cv2.line(disp, p1, p2, (255, 255, 0), 2)
            if len(img_points) == len(CALIB_TABLE_POINTS):
                p1 = (int(img_points[-1][0] * scale), int(img_points[-1][1] * scale))
                p2 = (int(img_points[0][0] * scale), int(img_points[0][1] * scale))
                cv2.line(disp, p1, p2, (255, 255, 0), 2)

        # 3. Tâm ngắm chuột (Crosshair)
        mx, my = mouse_pos
        if 0 <= mx < disp_w and 0 <= my < disp_h:
            cv2.line(disp, (mx - 15, my), (mx + 15, my), (0, 255, 255), 1)
            cv2.line(disp, (mx, my - 15), (mx, my + 15), (0, 255, 255), 1)
            cv2.circle(disp, (mx, my), 3, (0, 0, 255), -1)

        # 4. Thanh hướng dẫn vẽ trực tiếp trên ảnh
        cv2.rectangle(disp, (0, 0), (disp_w, 38), (0, 0, 0), -1)
        if len(img_points) < len(CALIB_TABLE_POINTS):
            msg = f"Click diem [{len(img_points)+1}/4]: Ban {CALIB_TABLE_POINTS[len(img_points)]}cm"
            cv2.putText(disp, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            msg = "DA CLICK DU 4 DIEM! Nhan 's' de Luu | 'u' de Undo | 'r' de Reset"
            cv2.putText(disp, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('u'), ord('U')):
            if img_points:
                removed = img_points.pop()
                print(f"[{cam_name}] Đã Undo điểm: {removed}")
        elif key in (ord('r'), ord('R')):
            img_points.clear()
            print(f"[{cam_name}] Đã reset toàn bộ điểm.")
        elif key in (ord('s'), ord('S')):
            if len(img_points) == len(CALIB_TABLE_POINTS):
                src_pts = np.array(img_points, dtype=np.float32)
                dst_pts = np.array(CALIB_TABLE_POINTS, dtype=np.float32)
                H, status = cv2.findHomography(src_pts, dst_pts)
                if H is not None:
                    H_matrix = H
                    print(f"[{cam_name}] Tính toán Homography thành công!")
                break
            else:
                print(f"[{cam_name}] Chưa click đủ {len(CALIB_TABLE_POINTS)} điểm!")
        elif key in (ord('q'), 27):
            print(f"[{cam_name}] Đã hủy Calibrate.")
            break

    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass
    return H_matrix


def run_full_calibration():
    """Chạy độc lập để Calibrate 2D toàn bộ camera."""
    from camera_driver import build_all_cameras
    print("=" * 60)
    print("CHƯƠNG TRÌNH HIỆU CHUẨN HOMOGRAPHY 2D CHO TẤT CẢ CAMERA")
    print("=" * 60)

    cams = build_all_cameras(max_cameras=4)
    if not cams:
        print("Không tìm thấy camera nào.")
        return

    results = load_homographies()
    for name, cam in cams.items():
        if not cam.open():
            continue
        cam.start()

        # Chờ frame đầu
        import time
        for _ in range(50):
            if cam.read() is not None:
                break
            time.sleep(0.05)

        H = calibrate_camera_interactive(name, cam.read)
        if H is not None:
            results[name] = H
        cam.stop()

    if results:
        save_homographies(results)
        print("\nHOÀN TẤT HIỆU CHUẨN 2D CHO TẤT CẢ CAMERA.")


if __name__ == "__main__":
    run_full_calibration()
