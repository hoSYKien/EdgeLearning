# -*- coding: utf-8 -*-
"""
MODULE HIỆU CHUẨN HOMOGRAPHY 2D (ẢNH -> MẶT BÀN) TƯƠNG TÁC CHO TỪNG CAMERA
==========================================================================
Tính năng nâng cấp:
  1. Hỗ trợ ZOOM IN / ZOOM OUT mượt mà:
     - Cuộn chuột (Mouse Wheel) hoặc phím '+' / '-' để phóng to / thu nhỏ.
     - Phím '0' hoặc 'f' để Reset Zoom về toàn khung hình (1.0x).
  2. Hỗ trợ PAN (Kéo di chuyển khung nhìn khi đang phóng to):
     - Giữ chuột phải (Right Click Drag) hoặc phím Mũi tên để kéo di chuyển.
  3. KÍNH LÚP PHÓNG ĐẠI ĐIỂM NGẮM (Picture-in-Picture Magnifier Loupe):
     - Cửa sổ kính lúp 4x - 8x ở góc màn hình hiển thị trực quan từng pixel thực dưới đầu chuột.
  4. ĐIỂM CHẤM SIÊU NHỎ & TÂM NGẮM RỖNG (Pixel-Perfect Precision):
     - Điểm chấm hiển thị dạng hạt siêu nhỏ (bán kính 2px) kèm vòng tròn mỏng, không che khuất góc nhọn của vật thể/mép giấy.
     - Tâm ngắm Hairline rỗng ở giữa (hollow crosshair) giúp căn chuẩn xác từng pixel.
  5. Đồng bộ tọa độ tuyệt đối 1:1 với ảnh gốc không bao giờ bị lệch.

PHÍM TẮT TRONG GIAO DIỆN CALIBRATE:
  - Chuột Trái (Left Click) : Chấm điểm mốc
  - Cuộn Chuột / Phím '+', '-' : Phóng to / Thu nhỏ (Zoom)
  - Giữ Chuột Phải & Kéo    : Di chuyển vùng nhìn (Pan)
  - 'u'                     : Undo điểm vừa chấm
  - 'r'                     : Reset làm lại từ đầu
  - '0' hoặc 'f'            : Reset Zoom về 1.0x (Fit Screen)
  - 's'                     : Lưu ma trận Homography khi đã chấm đủ 4 điểm
  - 'q' hoặc ESC            : Hủy / Bỏ qua camera này
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
    """
    Giao diện hiệu chuẩn 2D chuyên nghiệp với Zoom & Pan, kính lúp PiP
    và tâm ngắm siêu nhỏ chuẩn xác từng pixel.
    """
    frame = get_frame_func()
    if frame is None:
        print(f"[{cam_name}] Không lấy được frame để calibrate.")
        return None

    h_orig, w_orig = frame.shape[:2]
    disp_w, disp_h = 1280, 720

    img_points = []  # Lưu tọa độ thực trên ảnh gốc [(ox, oy), ...]

    # Trạng thái Zoom & Pan
    zoom_level = 1.0
    view_center = [w_orig / 2.0, h_orig / 2.0]  # Tọa độ tâm vùng nhìn trên ảnh gốc
    mouse_screen = [disp_w // 2, disp_h // 2]
    is_panning = False
    pan_start_mouse = [0, 0]
    pan_start_center = [0.0, 0.0]

    def get_view_roi():
        """Tính toán ROI hiển thị trên ảnh gốc dựa trên zoom_level và view_center."""
        vw = w_orig / zoom_level
        vh = h_orig / zoom_level
        
        cx, cy = view_center
        x0 = max(0.0, min(float(w_orig - vw), cx - vw / 2.0))
        y0 = max(0.0, min(float(h_orig - vh), cy - vh / 2.0))
        x1 = x0 + vw
        y1 = y0 + vh
        return x0, y0, x1, y1

    def screen_to_orig(sx, sy):
        """Chuyển tọa độ màn hình (sx, sy) -> tọa độ thực ảnh gốc (ox, oy)."""
        x0, y0, x1, y1 = get_view_roi()
        ox = x0 + (sx / float(disp_w)) * (x1 - x0)
        oy = y0 + (sy / float(disp_h)) * (y1 - y0)
        return max(0.0, min(float(w_orig - 1), ox)), max(0.0, min(float(h_orig - 1), oy))

    def orig_to_screen(ox, oy):
        """Chuyển tọa độ ảnh gốc (ox, oy) -> tọa độ màn hình (sx, sy)."""
        x0, y0, x1, y1 = get_view_roi()
        if (x1 - x0) == 0 or (y1 - y0) == 0:
            return 0, 0
        sx = int(((ox - x0) / float(x1 - x0)) * disp_w)
        sy = int(((oy - y0) / float(y1 - y0)) * disp_h)
        return sx, sy

    def apply_zoom(factor, cursor_screen_pos=None):
        nonlocal zoom_level, view_center
        old_zoom = zoom_level
        new_zoom = max(1.0, min(20.0, zoom_level * factor))
        if new_zoom == old_zoom:
            return

        if cursor_screen_pos is not None:
            # Zoom tập trung quanh vị trí con trỏ chuột
            ox, oy = screen_to_orig(cursor_screen_pos[0], cursor_screen_pos[1])
            zoom_level = new_zoom
            # Điều chỉnh view_center để (ox, oy) vẫn nằm ở vị trí chuột cũ
            vw_new = w_orig / zoom_level
            vh_new = h_orig / zoom_level
            ratio_x = cursor_screen_pos[0] / float(disp_w)
            ratio_y = cursor_screen_pos[1] / float(disp_h)
            view_center[0] = ox - (ratio_x - 0.5) * vw_new
            view_center[1] = oy - (ratio_y - 0.5) * vh_new
        else:
            zoom_level = new_zoom

        # Giới hạn view_center hợp lệ
        vw = w_orig / zoom_level
        vh = h_orig / zoom_level
        view_center[0] = max(vw / 2.0, min(w_orig - vw / 2.0, view_center[0]))
        view_center[1] = max(vh / 2.0, min(h_orig - vh / 2.0, view_center[1]))

    def on_mouse(event, x, y, flags, param):
        nonlocal is_panning, pan_start_mouse, pan_start_center, view_center
        mouse_screen[0] = x
        mouse_screen[1] = y

        # 1. Bấm chuột trái -> Chấm điểm
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(img_points) < len(CALIB_TABLE_POINTS):
                ox, oy = screen_to_orig(x, y)
                idx = len(img_points)
                img_points.append((ox, oy))
                print(f"[{cam_name}] Điểm {idx+1}/{len(CALIB_TABLE_POINTS)}: Ảnh({ox:.2f}, {oy:.2f}) -> Bàn {CALIB_TABLE_POINTS[idx]} cm")

        # 2. Cuộn chuột -> Phóng to / Thu nhỏ
        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0: Cuộn lên (Phóng to), flags < 0: Cuộn xuống (Thu nhỏ)
            if flags > 0:
                apply_zoom(1.25, (x, y))
            else:
                apply_zoom(0.80, (x, y))

        # 3. Giữ chuột phải hoặc chuột giữa -> Bắt đầu kéo (Pan)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            is_panning = True
            pan_start_mouse = [x, y]
            pan_start_center = [view_center[0], view_center[1]]

        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            is_panning = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if is_panning and zoom_level > 1.0:
                dx_screen = x - pan_start_mouse[0]
                dy_screen = y - pan_start_mouse[1]
                x0, y0, x1, y1 = get_view_roi()
                vw = x1 - x0
                vh = y1 - y0
                dx_orig = (dx_screen / float(disp_w)) * vw
                dy_orig = (dy_screen / float(disp_h)) * vh
                view_center[0] = pan_start_center[0] - dx_orig
                view_center[1] = pan_start_center[1] - dy_orig

                view_center[0] = max(vw / 2.0, min(w_orig - vw / 2.0, view_center[0]))
                view_center[1] = max(vh / 2.0, min(h_orig - vh / 2.0, view_center[1]))

    win_name = f"CALIBRATE 2D [{cam_name}] - Cuon chuot de ZOOM, Giu chuot phai de PAN"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU HIỆU CHUẨN 2D CHO CAMERA [{cam_name.upper()}]:")
    for idx, pt in enumerate(CALIB_TABLE_POINTS):
        print(f"  Điểm mốc {idx+1}: Tọa độ mặt bàn = {pt} cm")
    print("----------------------------------------------------------------------")
    print("HƯỚNG DẪN ĐIỀU KHIỂN ĐỘ CHÍNH XÁC CAO:")
    print("  - Chuột Trái (Left Click) : Chấm điểm mốc")
    print("  - Cuộn Chuột / Phím '+', '-' : Phóng to / Thu nhỏ (Zoom)")
    print("  - Giữ Chuột Phải & Kéo    : Di chuyển khung nhìn (Pan)")
    print("  - '0' hoặc 'f'            : Reset Zoom về 1.0x")
    print("  - 'u'                     : Undo điểm vừa chấm")
    print("  - 'r'                     : Reset xóa toàn bộ điểm làm lại")
    print("  - 's'                     : Lưu ma trận Homography (khi đã click đủ 4 điểm)")
    print("  - 'q' hoặc ESC            : Hủy / Bỏ qua camera này")
    print("=" * 70 + "\n")

    H_matrix = None

    while True:
        f = get_frame_func()
        if f is None:
            f = frame

        # 1. Trích xuất ROI và Scale theo Zoom
        x0, y0, x1, y1 = get_view_roi()
        ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        ix1 = max(ix0 + 1, min(w_orig, ix1))
        iy1 = max(iy0 + 1, min(h_orig, iy1))

        crop = f[iy0:iy1, ix0:ix1]
        disp = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # 2. Vẽ các điểm mốc đã chấm (Dạng hạt siêu nhỏ bán kính 2px + vòng mỏng)
        for idx, (px, py) in enumerate(img_points):
            sx, sy = orig_to_screen(px, py)
            if 0 <= sx < disp_w and 0 <= sy < disp_h:
                # Chấm hạt tâm siêu nhỏ (Radius 2px - Đỏ tươi)
                cv2.circle(disp, (sx, sy), 2, (0, 0, 255), -1, cv2.LINE_AA)
                # Vòng tròn mảnh định vị (Radius 5px - Vàng)
                cv2.circle(disp, (sx, sy), 5, (0, 255, 255), 1, cv2.LINE_AA)
                # Số thứ tự điểm dời lệch sang góc để không che khuất tâm
                cv2.putText(disp, f"{idx+1}", (sx + 8, sy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Vẽ đường nối các điểm đã chấm
        if len(img_points) > 1:
            for idx in range(len(img_points) - 1):
                p1 = orig_to_screen(img_points[idx][0], img_points[idx][1])
                p2 = orig_to_screen(img_points[idx+1][0], img_points[idx+1][1])
                cv2.line(disp, p1, p2, (255, 200, 0), 1, cv2.LINE_AA)
            if len(img_points) == len(CALIB_TABLE_POINTS):
                p1 = orig_to_screen(img_points[-1][0], img_points[-1][1])
                p2 = orig_to_screen(img_points[0][0], img_points[0][1])
                cv2.line(disp, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

        # 4. Tâm ngắm Hairline rỗng ở giữa (Hollow Crosshair)
        mx, my = mouse_screen
        if 0 <= mx < disp_w and 0 <= my < disp_h:
            gap = 3    # Khoảng hở rỗng ở tâm để nhìn rõ pixel dưới chuột
            ch_len = 16 # Độ dài vạch ngắm
            # Đường ngang
            cv2.line(disp, (mx - ch_len, my), (mx - gap, my), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(disp, (mx + gap, my), (mx + ch_len, my), (0, 255, 255), 1, cv2.LINE_AA)
            # Đường dọc
            cv2.line(disp, (mx, my - ch_len), (mx, my - gap), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(disp, (mx, my + gap), (mx, my + ch_len), (0, 255, 255), 1, cv2.LINE_AA)

        # 5. CỬA SỔ KÍNH LÚP PHÓNG ĐẠI (PICTURE-IN-PICTURE LOUPE)
        cur_ox, cur_oy = screen_to_orig(mx, my)
        loupe_radius = 24   # Bán kính vùng crop trên ảnh gốc
        lx0 = max(0, int(cur_ox - loupe_radius))
        ly0 = max(0, int(cur_oy - loupe_radius))
        lx1 = min(w_orig, int(cur_ox + loupe_radius))
        ly1 = min(h_orig, int(cur_oy + loupe_radius))

        if lx1 > lx0 and ly1 > ly0:
            loupe_crop = f[ly0:ly1, lx0:lx1]
            loupe_size = 180  # Kích thước khung kính lúp trên màn hình
            loupe_view = cv2.resize(loupe_crop, (loupe_size, loupe_size), interpolation=cv2.INTER_NEAREST)

            # Vẽ tâm ngắm kính lúp
            lc_x = int(((cur_ox - lx0) / float(lx1 - lx0)) * loupe_size)
            lc_y = int(((cur_oy - ly0) / float(ly1 - ly0)) * loupe_size)
            cv2.drawMarker(loupe_view, (lc_x, lc_y), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)

            # Vị trí đặt kính lúp ở góc dưới bên phải màn hình
            pos_x = disp_w - loupe_size - 12
            pos_y = disp_h - loupe_size - 12
            disp[pos_y:pos_y+loupe_size, pos_x:pos_x+loupe_size] = loupe_view

            # Viền và tiêu đề kính lúp
            cv2.rectangle(disp, (pos_x, pos_y), (pos_x + loupe_size, pos_y + loupe_size), (0, 255, 255), 1)
            cv2.rectangle(disp, (pos_x, pos_y - 20), (pos_x + loupe_size, pos_y), (20, 20, 20), -1)
            cv2.putText(disp, "KINH LUP (LOUPE)", (pos_x + 10, pos_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 6. Thanh trạng thái trên cùng (Header)
        cv2.rectangle(disp, (0, 0), (disp_w, 36), (15, 15, 15), -1)
        cv2.line(disp, (0, 36), (disp_w, 36), (0, 200, 255), 1)

        if len(img_points) < len(CALIB_TABLE_POINTS):
            next_idx = len(img_points)
            step_text = f"Diem [{next_idx+1}/4]: {CALIB_TABLE_POINTS[next_idx]}cm"
            color_step = (0, 255, 255)
        else:
            step_text = "DA DU 4 DIEM -> Nhan 's' de Luu ma tran"
            color_step = (0, 255, 0)

        coord_text = f"Toa do thuc: ({cur_ox:.1f}, {cur_oy:.1f}) | Zoom: {zoom_level:.1f}x"
        help_text = "Cuon chuot: Zoom | Chuot phai: Pan | 'u': Undo | 'r': Reset | '0': Fit | 's': Luu"

        cv2.putText(disp, step_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_step, 1, cv2.LINE_AA)
        cv2.putText(disp, coord_text, (380, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, help_text, (760, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(15) & 0xFF

        # Xử lý phím tắt
        if key in (ord('u'), ord('U')):
            if img_points:
                removed = img_points.pop()
                print(f"[{cam_name}] Undo điểm vừa chấm: {removed}")
        elif key in (ord('r'), ord('R')):
            img_points.clear()
            print(f"[{cam_name}] Đã reset toàn bộ điểm.")
        elif key in (ord('0'), ord('f'), ord('F')):
            zoom_level = 1.0
            view_center = [w_orig / 2.0, h_orig / 2.0]
            print(f"[{cam_name}] Reset Zoom về 1.0x (Toàn cảnh).")
        elif key in (ord('+'), ord('=')):
            apply_zoom(1.30, mouse_screen)
        elif key in (ord('-'), ord('_')):
            apply_zoom(0.75, mouse_screen)
        elif key in (ord('s'), ord('S')):
            if len(img_points) == len(CALIB_TABLE_POINTS):
                src_pts = np.array(img_points, dtype=np.float32)
                dst_pts = np.array(CALIB_TABLE_POINTS, dtype=np.float32)
                H, status = cv2.findHomography(src_pts, dst_pts)
                if H is not None:
                    H_matrix = H
                    print(f"\n[{cam_name}] -> TÍNH TOÁN HOMOGRAPHY 2D THÀNH CÔNG!")
                    print(f"  Ma trận H ({cam_name}):\n{H_matrix}")
                break
            else:
                print(f"[{cam_name}] Cảnh báo: Chưa chấm đủ {len(CALIB_TABLE_POINTS)} điểm!")
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
    print("=" * 70)
    print("CHƯƠNG TRÌNH HIỆU CHUẨN HOMOGRAPHY 2D CHÍNH XÁC CAO CHO TẤT CẢ CAMERA")
    print("=" * 70)

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
