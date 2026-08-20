# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH TỰ ĐỘNG GÁN NHÃN DATASET YOLO (1 CLASS) QUA CHIẾU HOMOGRAPHY MASTER-SLAVE
======================================================================================
Nguyên lý hoạt động:
  1. Master Camera (CAM 1 - COLOR):
     - Chỉ duy nhất Cam 1 (Màu) chạy thuật toán Trừ nền & Khử bóng chuyên sâu (Shadow Removal).
     - Phát hiện trọn vẹn 100% hình dạng vật thể, không bị đứt đoạn do chóa/lóa sáng.
  2. Slave Cameras (CAM 2, CAM 3 - MONO):
     - KHÔNG cần tự trừ nền (tránh tình trạng chóa sáng/phản quang nylon bị cắt vụn Bounding Box).
     - Toàn bộ Bounding Box từ Cam 1 được CHIẾU TOÁN HỌC CHÍNH XÁC sang Cam 2 và Cam 3 qua ma trận Homography:
           H_{1 -> i} = inv(H_i) @ H_1
  3. Bấm phím 'c' (hoặc Phím Cách Space):
     - Lưu 3 file ảnh riêng biệt từ 3 góc chụp vào `dataset_yolo/images/`.
     - Tự động sinh 3 file nhãn `.txt` chuẩn YOLO vào `dataset_yolo/labels/` (tất cả đều chuẩn 1 class `0`).
     - Tự động cập nhật `data.yaml`.

PHÍM TẮT ĐIỀU KHIỂN:
  - 'c' hoặc SPACE : Chụp & Tự động lưu 3 ảnh + 3 file nhãn YOLO
  - 'r'            : Học lại nền cho Master Camera (Cam 1) khi mặt bàn TRỐNG
  - 'k'            : Hiệu chuẩn Homography 2D trực tiếp cho các camera
  - 'm'            : Bật / Tắt xem trước Mask khử bóng trên Cam 1
  - 'g'            : Chuyển đổi Grid View / Cửa sổ riêng
  - 'q' hoặc ESC   : Thoát chương trình
"""

import os
import sys
import time
import cv2
import numpy as np

# Thêm thư mục hiện tại vào sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    MAX_CAMERAS, MASTER_CAM, HOMOGRAPHY_FILE,
    DRAW_QUAD_POLYGON, TRACK_DIST_CM
)
from camera_driver import build_all_cameras, discover_cameras
from detector import ObjectDetector
from coordinator import get_cam_to_cam_matrix, project_bbox_to_slave_cam
from calibrate import load_homographies, save_homographies, calibrate_camera_interactive

# Thư mục lưu Dataset YOLO
DATASET_DIR = os.path.join(_THIS_DIR, "dataset_yolo")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
CLASS_NAME = "object"
CLASS_ID = 0


def ensure_dataset_structure():
    """Tạo cấu trúc thư mục dataset và file data.yaml cho YOLO."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    abs_dataset_dir = os.path.abspath(DATASET_DIR).replace("\\", "/")
    yaml_content = f"""# CẤU HÌNH TRAIN YOLO CHO DATASET 1 CLASS
path: {abs_dataset_dir}
train: images
val: images

names:
  0: {CLASS_NAME}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    classes_path = os.path.join(DATASET_DIR, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write(f"{CLASS_NAME}\n")

    print(f"-> Thư mục Dataset YOLO: {DATASET_DIR}")
    print(f"-> File cấu hình: {yaml_path}")


def convert_to_yolo_bbox(bbox, img_w, img_h):
    """
    Chuyển đổi Bounding Box (x1, y1, x2, y2) sang định dạng chuẩn YOLO:
    <class_id> <x_center> <y_center> <width> <height> (chuẩn hóa 0.0 -> 1.0)
    """
    x1, y1, x2, y2 = bbox
    # Giới hạn trong kích thước ảnh
    x1 = max(0.0, min(float(img_w), float(x1)))
    y1 = max(0.0, min(float(img_h), float(y1)))
    x2 = max(0.0, min(float(img_w), float(x2)))
    y2 = max(0.0, min(float(img_h), float(y2)))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 5 or bh <= 5:
        return None

    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0

    xc_norm = xc / float(img_w)
    yc_norm = yc / float(img_h)
    bw_norm = bw / float(img_w)
    bh_norm = bh / float(img_h)

    # Đảm bảo giá trị nằm trong khoảng [0, 1]
    xc_norm = max(0.0, min(1.0, xc_norm))
    yc_norm = max(0.0, min(1.0, yc_norm))
    bw_norm = max(0.0, min(1.0, bw_norm))
    bh_norm = max(0.0, min(1.0, bh_norm))

    return f"{CLASS_ID} {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"


def draw_label(img, text, pos, color=(0, 255, 0), font_scale=0.75, thickness=2):
    """Vẽ nhãn chữ có nền đen tương phản cao."""
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 8, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def main():
    print("=" * 70)
    print("  CHƯƠNG TRÌNH TỰ ĐỘNG GÁN NHÃN YOLO QUA CHIẾU HOMOGRAPHY MASTER -> SLAVES")
    print("=" * 70)

    ensure_dataset_structure()

    # 1. Quét và mở tất cả camera
    available_devices = discover_cameras()
    if not available_devices:
        print("Lỗi: Không tìm thấy camera công nghiệp nào trên mạng.")
        return

    print(f"\nTìm thấy {len(available_devices)} camera:")
    for dev in available_devices:
        print(f"  [{dev['index']}] {dev['type']} - {dev['model']} | IP: {dev['ip']} | SN: {dev['serial']}")

    cams = build_all_cameras(max_cameras=MAX_CAMERAS)
    opened_cams = {}
    for name, cam in cams.items():
        if cam.open():
            if cam.start():
                opened_cams[name] = cam
            else:
                cam.stop()

    if not opened_cams:
        print("Lỗi: Không thể mở camera nào.")
        return
    cams = opened_cams

    # Xác định Master Camera (Cam 1 Màu) và các Slave Camera (Mono)
    master_name = MASTER_CAM if MASTER_CAM in cams else list(cams.keys())[0]
    slave_names = [name for name in cams if name != master_name]

    print(f"\n[PHÂN CHIA VAI TRÒ]")
    print(f"  - Master Camera (Khử bóng & Detect chuẩn): [{master_name.upper()}] ({cams[master_name].cam_info.get('model', '')})")
    print(f"  - Slave Cameras (Nhận BBox chiếu Homography): {[s.upper() for s in slave_names]}")

    # 2. Nạp ma trận Homography
    homographies = load_homographies(HOMOGRAPHY_FILE)
    missing = [name for name in cams if name not in homographies]
    if missing:
        print(f"\n[CẢNH BÁO] Các camera chưa có ma trận Homography: {missing}")
        print("-> Nhấn phím 'k' để mở giao diện bấm 4 điểm mốc hiệu chuẩn mặt bàn.")

    # 3. Khởi tạo Detector (Chỉ chạy trên Master Camera)
    detector_master = ObjectDetector()

    # 4. Học nền ban đầu cho Master Camera
    print("\n" + "=" * 60)
    print(f"ĐANG HỌC NỀN CHO MASTER CAMERA [{master_name.upper()}]...")
    print("Vui lòng giữ khung nhìn mặt bàn TRỐNG trong 3-5 giây...")
    detector_master.learn_background(cams[master_name], num_frames=15)
    print("Học nền hoàn tất! Đang kích hoạt luồng chiếu Bounding Box Homography...")
    print("=" * 60 + "\n")

    show_grid = True
    show_mask = True
    total_saved_samples = len([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    flash_frames = 0

    print("HỆ THỐNG ĐÃ SẴN SÀNG:")
    print("  - Đặt sản phẩm lên mặt bàn.")
    print("  - Nhấn phím 'c' hoặc Phím Cách (SPACE) để CHỤP & GÁN NHÃN YOLO ĐỒNG THỜI.")
    print("  - Nhấn phím 'r' để học lại nền khi mặt bàn trống.")
    print("  - Nhấn phím 'k' để hiệu chuẩn lại Homography 2D.")
    print("  - Nhấn phím 'm' để bật/tắt xem trước Mask trên Master Cam.")
    print("  - Nhấn phím 'q' để thoát.")
    print("=" * 70 + "\n")

    try:
        while True:
            raw_frames = {}
            for name, cam in cams.items():
                f = cam.read()
                if f is not None:
                    raw_frames[name] = f

            if master_name not in raw_frames:
                time.sleep(0.01)
                continue

            frame_master = raw_frames[master_name]
            h_m, w_m = frame_master.shape[:2]

            # 1. PHÁT HIỆN VẬT THỂ DUY NHẤT TRÊN MASTER CAMERA (CAM 1 MÀU)
            master_obj_candidates = detector_master.detect_objects(frame_master)
            master_bboxes = [bbox for (cx, cy), bbox in master_obj_candidates]

            # 2. CHIẾU BOUNDING BOX SANG TOÀN BỘ SLAVE CAMERAS QUA HOMOGRAPHY
            H_master = homographies.get(master_name)
            camera_bboxes = {master_name: master_bboxes}
            camera_quads = {master_name: []}

            for s_name in slave_names:
                H_slave = homographies.get(s_name)
                H_m2s = get_cam_to_cam_matrix(H_master, H_slave)
                
                s_bboxes = []
                s_quads = []

                if s_name in raw_frames:
                    sh, sw = raw_frames[s_name].shape[:2]
                    for m_bbox in master_bboxes:
                        if H_m2s is not None:
                            quad, aabb, center = project_bbox_to_slave_cam(H_m2s, m_bbox)
                            if aabb is not None:
                                ax1, ay1, ax2, ay2 = aabb
                                # Giới hạn trong kích thước ảnh của camera phụ
                                ax1 = max(0, min(sw - 1, ax1))
                                ay1 = max(0, min(sh - 1, ay1))
                                ax2 = max(0, min(sw, ax2))
                                ay2 = max(0, min(sh, ay2))
                                if (ax2 - ax1) > 10 and (ay2 - ay1) > 10:
                                    s_bboxes.append((ax1, ay1, ax2, ay2))
                                    s_quads.append(quad)
                        else:
                            # Nếu chưa calibrate, tạm dùng lại bbox của Cam 1
                            s_bboxes.append(m_bbox)
                            s_quads.append(None)

                camera_bboxes[s_name] = s_bboxes
                camera_quads[s_name] = s_quads

            # 3. VẼ GIAO DIỆN & BOUNDING BOX TRỰC TIẾP LÊN TỪNG CAMERA
            display_canvases = []
            disp_w, disp_h = 640, 480

            for name, cam in cams.items():
                f = raw_frames.get(name)
                if f is None:
                    blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    cv2.putText(blank, f"[{name}] Dang cho frame...", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    display_canvases.append((name, blank))
                    continue

                disp = f.copy()
                is_m = (name == master_name)
                bboxes = camera_bboxes.get(name, [])
                quads = camera_quads.get(name, [])

                # VẼ TRÊN MASTER CAMERA (CAM 1)
                if is_m:
                    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
                        bw, bh = x2 - x1, y2 - y1
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"[0] {CLASS_NAME} #{idx+1} ({bw}x{bh}px)"
                        draw_label(disp, label, (x1, max(30, y1 - 10)), color=(0, 255, 0), font_scale=0.75, thickness=2)

                    # Ô xem trước Mask khử bóng
                    if show_mask and detector_master.last_mask is not None:
                        mask_gray = detector_master.last_mask
                        mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
                        mh, mw = mask_bgr.shape[:2]
                        pw, ph = 180, int(180 * (mh / mw))
                        mask_prev = cv2.resize(mask_bgr, (pw, ph))
                        dh, dw = disp.shape[:2]
                        px0 = dw - pw - 10
                        py0 = dh - ph - 10
                        disp[py0:py0+ph, px0:px0+pw] = mask_prev
                        cv2.rectangle(disp, (px0, py0), (px0 + pw, py0 + ph), (0, 255, 0), 1)
                        cv2.putText(disp, "SHADOW-FREE MASK ('m')", (px0 + 5, py0 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

                # VẼ TRÊN SLAVE CAMERAS (CAM 2 & CAM 3 ĐƯỢC CHIẾU TỪ CAM 1)
                else:
                    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
                        bw, bh = x2 - x1, y2 - y1
                        quad = quads[idx] if idx < len(quads) else None

                        # Vẽ đa giác 4 góc xoay thực tế
                        if quad is not None and DRAW_QUAD_POLYGON:
                            cv2.polylines(disp, [quad], isClosed=True, color=(0, 215, 255), thickness=2)

                        # Vẽ Bounding Box AABB chuẩn YOLO (màu xanh lá)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"[SYNC] [0] {CLASS_NAME} #{idx+1} ({bw}x{bh}px)"
                        draw_label(disp, label, (x1, max(30, y1 - 10)), color=(0, 255, 0), font_scale=0.75, thickness=2)

                # Scale khung hình cho giao diện Grid
                h, w = disp.shape[:2]
                scale = min(disp_w / w, disp_h / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(disp, (nw, nh))

                canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                xo = (disp_w - nw) // 2
                yo = (disp_h - nh) // 2
                canvas[yo:yo+nh, xo:xo+nw] = resized

                # Header thông tin camera
                obj_count = len(bboxes)
                role_str = "MASTER (COLOR)" if is_m else "SLAVE (HOMOGRAPHY)"
                head_color = (0, 255, 0) if obj_count > 0 else (0, 200, 255)
                header = f"[{name.upper()}: {role_str}] {cam.cam_info.get('model', '')} | Vat the: {obj_count} | FPS: {cam.fps:.1f}"
                cv2.rectangle(canvas, (0, 0), (disp_w, 32), (20, 20, 20), -1)
                cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, head_color, 1, cv2.LINE_AA)

                display_canvases.append((name, canvas))

            # 4. HIỂN THỊ GIAO DIỆN GRID
            if show_grid and len(display_canvases) > 0:
                if len(display_canvases) == 1:
                    grid_view = display_canvases[0][1]
                elif len(display_canvases) == 2:
                    grid_view = np.hstack([display_canvases[0][1], display_canvases[1][1]])
                elif len(display_canvases) <= 4:
                    row1 = np.hstack([display_canvases[0][1], display_canvases[1][1]])
                    row2_items = []
                    for idx in range(2, 4):
                        if idx < len(display_canvases):
                            row2_items.append(display_canvases[idx][1])
                        else:
                            row2_items.append(np.zeros((disp_h, disp_w, 3), dtype=np.uint8))
                    row2 = np.hstack(row2_items)
                    grid_view = np.vstack([row1, row2])
                else:
                    grid_view = np.hstack([item[1] for item in display_canvases[:3]])

                # Thanh trạng thái
                gh, gw = grid_view.shape[:2]
                status_bar = np.zeros((38, gw, 3), dtype=np.uint8)
                msg_status = f"Da luu: {total_saved_samples} anh | [SPACE/C]: Chup YOLO | [R]: Hoc nen Cam 1 | [K]: Calib 2D | [M]: Mask | [Q]: Thoat"
                bar_color = (0, 255, 0) if flash_frames > 0 else (0, 200, 255)
                cv2.putText(status_bar, msg_status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 1, cv2.LINE_AA)
                grid_view = np.vstack([grid_view, status_bar])

                # Hiệu ứng nháy xanh khi chụp
                if flash_frames > 0:
                    flash_overlay = np.full_like(grid_view, (0, 100, 0), dtype=np.uint8)
                    grid_view = cv2.addWeighted(grid_view, 0.7, flash_overlay, 0.3, 0)
                    flash_frames -= 1

                cv2.imshow("YOLO 1-Class Auto-Label Dataset Generator [Homography Sync]", grid_view)
            else:
                for name, canvas in display_canvases:
                    cv2.imshow(f"Camera [{name}]", canvas)

            # 5. XỬ LÝ PHÍM TẮT
            key = cv2.waitKey(15) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key in (ord('c'), ord('C'), 32):  # 'c' hoặc Space
                # THỰC HIỆN CHỤP & LƯU NHÃN YOLO CHO TOÀN BỘ 3 CAMERA
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                millis_str = f"{int((time.time() % 1) * 1000):03d}"
                session_id = f"{timestamp_str}_{millis_str}"

                saved_info = []

                for name, frame in raw_frames.items():
                    img_h, img_w = frame.shape[:2]
                    base_filename = f"{name}_{session_id}"
                    img_filename = f"{base_filename}.jpg"
                    label_filename = f"{base_filename}.txt"

                    img_path = os.path.join(IMAGES_DIR, img_filename)
                    label_path = os.path.join(LABELS_DIR, label_filename)

                    # 1. Lưu ảnh gốc
                    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # 2. Tạo nhãn YOLO
                    bboxes = camera_bboxes.get(name, [])
                    yolo_lines = []
                    for bbox in bboxes:
                        yolo_line = convert_to_yolo_bbox(bbox, img_w, img_h)
                        if yolo_line:
                            yolo_lines.append(yolo_line)

                    # 3. Ghi file label .txt
                    with open(label_path, "w", encoding="utf-8") as f_lbl:
                        if yolo_lines:
                            f_lbl.write("\n".join(yolo_lines) + "\n")

                    total_saved_samples += 1
                    saved_info.append(f"  -> [{name.upper()}]: {len(yolo_lines)} vat the -> {img_filename}")

                flash_frames = 4
                print("\n" + "=" * 60)
                print(f"[CHỤP & GÁN NHÃN YOLO THÀNH CÔNG] - Session: {session_id}")
                for info in saved_info:
                    print(info)
                print(f"Tổng số ảnh dataset hiện tại: {total_saved_samples}")
                print("=" * 60)

            elif key in (ord('r'), ord('R')):
                print(f"\n[Học lại nền] Đang học lại nền cho Master Camera [{master_name.upper()}]...")
                detector_master.learn_background(cams[master_name], num_frames=15)
                print("[Học lại nền] Hoàn tất.")

            elif key in (ord('k'), ord('K')):
                print("\n[Hiệu chuẩn Homography 2D] Đang mở giao diện căn chỉnh 4 điểm mốc...")
                for name, cam in cams.items():
                    H = calibrate_camera_interactive(name, cam.read)
                    if H is not None:
                        homographies[name] = H
                save_homographies(homographies, HOMOGRAPHY_FILE)
                print("[Hiệu chuẩn Homography 2D] Đã lưu ma trận mới thành công!")

            elif key in (ord('m'), ord('M')):
                show_mask = not show_mask
                print(f"-> Xem trước Mask: {'BẬT' if show_mask else 'TẮT'}")

            elif key in (ord('g'), ord('G')):
                show_grid = not show_grid
                if show_grid:
                    for name in cams:
                        try:
                            cv2.destroyWindow(f"Camera [{name}]")
                        except Exception:
                            pass
                else:
                    try:
                        cv2.destroyWindow("YOLO 1-Class Auto-Label Dataset Generator [Homography Sync]")
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\nNhận tín hiệu dừng...")
    finally:
        print("\nĐang đóng tất cả camera và giải phóng tài nguyên...")
        for cam in cams.values():
            try:
                cam.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            from MvCameraControl_class import MvCamera
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass
        print(f"Hoàn tất! Toàn bộ dataset đã được lưu tại: {DATASET_DIR}")


if __name__ == "__main__":
    main()
