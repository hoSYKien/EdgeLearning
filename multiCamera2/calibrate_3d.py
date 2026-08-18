# -*- coding: utf-8 -*-
"""
MODULE HIỆU CHUẨN 3D (3D CALIBRATION RIG) CHO TOÀN BỘ CAMERA CÔNG NGHIỆP
========================================================================
Vật mẫu: Khối hộp chữ nhật có kích thước chuẩn đặt cố định trên bàn:
    - P1P2 (Chiều ngang X) = 7.2 cm
    - P1P4 (Chiều sâu Y)   = 4.0 cm
    - P1P5 (Chiều cao Z)   = 8.3 cm

Thuật toán:
    - DLT (Direct Linear Transform) & SVD giải ra Ma trận Chiếu 3D (Projection Matrix P: 3x4).
    - [u*w, v*w, w]^T = P @ [X, Y, Z, 1]^T
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

from config import (
    CALIB_3D_POINTS, BOX_3D_EDGES, POINT_NAMES_3D,
    CALIBRATION_3D_FILE, CALIB_BOX_WIDTH, CALIB_BOX_DEPTH, CALIB_BOX_HEIGHT
)


def compute_3d_projection_matrix_dlt(object_points_3d, image_points_2d):
    """
    Tính Ma trận Chiếu 3D P (3x4) từ N điểm 3D tương ứng N điểm 2D (N >= 6, ở đây N=8).
    Sử dụng thuật toán DLT (Direct Linear Transform) qua phân tích suy biến SVD.
    """
    num_pts = len(object_points_3d)
    if num_pts < 6:
        return None, 999.0

    A = []
    for i in range(num_pts):
        X, Y, Z = object_points_3d[i]
        u, v = image_points_2d[i]
        A.append([X, Y, Z, 1.0, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1.0, -v * X, -v * Y, -v * Z, -v])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Chuẩn hóa để P[2, 3] > 0
    if P[2, 3] < 0:
        P = -P

    # Tính sai số chiếu lại (Reprojection Error - RMSE in pixels)
    errors = []
    for i in range(num_pts):
        X, Y, Z = object_points_3d[i]
        u_real, v_real = image_points_2d[i]
        p_proj = P @ np.array([X, Y, Z, 1.0], dtype=np.float64)
        if abs(p_proj[2]) > 1e-9:
            u_proj = p_proj[0] / p_proj[2]
            v_proj = p_proj[1] / p_proj[2]
            err = np.hypot(u_real - u_proj, v_real - v_proj)
            errors.append(err)

    rmse = float(np.mean(errors)) if errors else 0.0
    return P, rmse


def project_3d_to_2d(P, X, Y, Z):
    """Chiếu 1 điểm 3D (X, Y, Z) sang tọa độ pixel 2D (u, v)."""
    if P is None:
        return None
    p3d = np.array([X, Y, Z, 1.0], dtype=np.float64)
    p2d = P @ p3d
    if abs(p2d[2]) < 1e-9:
        return None
    return float(p2d[0] / p2d[2]), float(p2d[1] / p2d[2])


def load_calibration_3d(filepath=CALIBRATION_3D_FILE):
    """Nạp ma trận chiếu 3D P của các camera từ file JSON."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float64) for k, v in data.items()}
    except Exception as e:
        print(f"Lỗi đọc {filepath}: {e}")
        return {}


def save_calibration_3d(P_dict, filepath=CALIBRATION_3D_FILE):
    """Lưu ma trận chiếu 3D P của các camera vào file JSON."""
    data = {k: v.tolist() for k, v in P_dict.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"-> Đã lưu Ma trận Chiếu 3D thành công vào: {filepath}")


def draw_3d_wireframe(img, pts_2d, color=(0, 255, 0), thickness=2):
    """Vẽ khung hộp 3D (8 đỉnh 12 cạnh) lên ảnh."""
    if len(pts_2d) < 8:
        return
    pts = [(int(p[0]), int(p[1])) for p in pts_2d]

    # Vẽ 12 cạnh
    for (i, j) in BOX_3D_EDGES:
        if i < len(pts) and j < len(pts):
            if i < 4 and j < 4:
                edge_color = (255, 255, 0)      # Đáy (Cyan)
            elif i >= 4 and j >= 4:
                edge_color = (0, 255, 255)      # Đỉnh (Yellow)
            else:
                edge_color = (255, 0, 255)      # Cột đứng (Magenta)
            cv2.line(img, pts[i], pts[j], edge_color, thickness, cv2.LINE_AA)

    # Vẽ 8 đỉnh
    for idx, (px, py) in enumerate(pts):
        pt_color = (0, 0, 255) if idx < 4 else (0, 255, 0)
        cv2.circle(img, (px, py), 6, pt_color, -1)
        cv2.circle(img, (px, py), 8, (255, 255, 255), 2)
        cv2.putText(img, f"P{idx+1}", (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def calibrate_camera_3d_interactive(cam_name, get_frame_func):
    """Giao diện tương tác bấm 8 đỉnh khối hộp 3D trên ảnh camera (Chuẩn 1:1 không lệch pixel)."""
    frame = get_frame_func()
    if frame is None:
        print(f"[{cam_name}] Không lấy được frame để calibrate 3D.")
        return None

    img_points = []
    h_orig, w_orig = frame.shape[:2]
    view_w, view_h = 1280, 720
    scale = min(view_w / w_orig, view_h / h_orig)

    # Lưu tọa độ chuột hiện tại để vẽ tâm ngắm (Crosshair)
    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(img_points) < len(CALIB_3D_POINTS):
                # Quy chuẩn chính xác 1:1 không lệch
                orig_x = max(0.0, min(float(w_orig - 1), x / scale))
                orig_y = max(0.0, min(float(h_orig - 1), y / scale))
                idx = len(img_points)
                img_points.append((orig_x, orig_y))
                print(f"[{cam_name}] Click {POINT_NAMES_3D[idx]}: Ảnh ({orig_x:.1f}, {orig_y:.1f})")

    win_name = f"CALIBRATE 3D [{cam_name}] - Click 8 dinh: P1->P4 (Day) roi P5->P8 (Dinh)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU HIỆU CHUẨN 3D CHO CAMERA [{cam_name.upper()}]:")
    print(f"Kích thước hộp: X={CALIB_BOX_WIDTH}cm (P1P2), Y={CALIB_BOX_DEPTH}cm (P1P4), Z={CALIB_BOX_HEIGHT}cm (P1P5)")
    print("Hãy click chuột lần lượt 8 đỉnh theo thứ tự sau:")
    for idx, name in enumerate(POINT_NAMES_3D):
        print(f"  [{idx+1}/8] {name}")
    print("\nPhím điều khiển:")
    print("  - 's': Lưu Ma trận 3D P (sau khi đã click đủ 8 điểm)")
    print("  - 'u': Undo điểm click gần nhất")
    print("  - 'r': Reset xóa toàn bộ để click lại từ đầu")
    print("  - 'q' hoặc ESC: Hủy bỏ / Bỏ qua camera này")
    print("=" * 70 + "\n")

    P_matrix = None
    disp_w = int(w_orig * scale)
    disp_h = int(h_orig * scale)

    while True:
        f = get_frame_func()
        if f is None:
            f = frame
        disp = cv2.resize(f, (disp_w, disp_h))

        # 1. Vẽ các điểm đã click
        scaled_points = [(px * scale, py * scale) for (px, py) in img_points]
        for idx, (sx, sy) in enumerate(scaled_points):
            pt_col = (0, 0, 255) if idx < 4 else (0, 255, 0)
            cv2.circle(disp, (int(sx), int(sy)), 6, pt_col, -1)
            cv2.circle(disp, (int(sx), int(sy)), 8, (255, 255, 255), 2)
            cv2.putText(disp, f"P{idx+1}", (int(sx) + 10, int(sy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Nối dây các cạnh của hộp 3D
        if len(scaled_points) >= 2:
            for (i, j) in BOX_3D_EDGES:
                if i < len(scaled_points) and j < len(scaled_points):
                    p1 = (int(scaled_points[i][0]), int(scaled_points[i][1]))
                    p2 = (int(scaled_points[j][0]), int(scaled_points[j][1]))
                    edge_col = (255, 255, 0) if i < 4 and j < 4 else ((0, 255, 255) if i >= 4 and j >= 4 else (255, 0, 255))
                    cv2.line(disp, p1, p2, edge_col, 2, cv2.LINE_AA)

        # 3. Vẽ tâm ngắm chuột (Crosshair) để người dùng canh chuẩn từng pixel
        mx, my = mouse_pos
        if 0 <= mx < disp_w and 0 <= my < disp_h:
            cv2.line(disp, (mx - 15, my), (mx + 15, my), (0, 255, 255), 1)
            cv2.line(disp, (mx, my - 15), (mx, my + 15), (0, 255, 255), 1)
            cv2.circle(disp, (mx, my), 3, (0, 0, 255), -1)

        # 4. Thanh hướng dẫn vẽ TRỰC TIẾP TRÊN ẢNH (Không dùng vstack để tránh lệch tọa độ chuột)
        cv2.rectangle(disp, (0, 0), (disp_w, 40), (0, 0, 0), -1)
        if len(img_points) < 8:
            next_pt = POINT_NAMES_3D[len(img_points)]
            msg = f"Click [{len(img_points)+1}/8]: {next_pt}"
            cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            msg = "DA CLICK DU 8 DINH! Nhan 's' de Luu | 'u' de Undo | 'r' de Reset"
            cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

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
            if len(img_points) == 8:
                P, rmse = compute_3d_projection_matrix_dlt(CALIB_3D_POINTS, img_points)
                if P is not None:
                    P_matrix = P
                    print("\n" + "*" * 60)
                    print(f"[{cam_name}] TÍNH TOÁN MA TRẬN CHIẾU 3D THÀNH CÔNG!")
                    print(f"  Sai số tái tạo (Reprojection Error RMSE): {rmse:.2f} pixels")
                    print(f"  Ma trận P (3x4):\n{P}")
                    print("*" * 60 + "\n")
                break
            else:
                print(f"[{cam_name}] Chưa click đủ 8 điểm! (Hiện có: {len(img_points)}/8)")
        elif key in (ord('q'), 27):
            print(f"[{cam_name}] Đã hủy Calibrate 3D.")
            break

    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass
    return P_matrix


def run_full_3d_calibration():
    """Chạy độc lập để Calibrate 3D cho toàn bộ camera."""
    from camera_driver import build_all_cameras
    print("=" * 70)
    print("   CHƯƠNG TRÌNH HIỆU CHUẨN 3D (3D CALIBRATION RIG) CHO TẤT CẢ CAMERA")
    print(f"   Vật mẫu: Hộp X={CALIB_BOX_WIDTH}cm, Y={CALIB_BOX_DEPTH}cm, Z={CALIB_BOX_HEIGHT}cm")
    print("=" * 70)

    cams = build_all_cameras(max_cameras=4)
    if not cams:
        print("Lỗi: Không tìm thấy camera nào.")
        return

    results = load_calibration_3d()
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

        P = calibrate_camera_3d_interactive(name, cam.read)
        if P is not None:
            results[name] = P
        cam.stop()

    if results:
        save_calibration_3d(results)
        print("\nHOÀN TẤT HIỆU CHUẨN 3D CHO TẤT CẢ CAMERA.")


if __name__ == "__main__":
    run_full_3d_calibration()
