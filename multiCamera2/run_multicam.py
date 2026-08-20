# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH CHÍNH: ĐỒNG BỘ ID & CHIẾU BBOX 2D TỪ CAM 1 SANG CÁC CAMERA SAU
=============================================================================
Kiến trúc Master - Slave 2D:
  - Cam 1 (Color - Master): Trừ nền / mask khử bóng, đọc Barcode và gán ID.
  - Cam 2, 3, 4 (Slave): Tự động nhận Bounding Box, Đa giác góc nhìn và ID
    được chiếu trực tiếp từ Cam 1 qua Ma trận Homography (H_{1 -> i}).

Chạy:
    python run_multicam.py

Phím tắt điều khiển:
    - 'c'          : Hiệu chuẩn Homography 2D trực tiếp cho từng camera
    - 'r'          : Học lại nền cho Master Camera (Cam 1)
    - 'm'          : Bật / Tắt ô xem trước Mask đã khử sạch bóng
    - 's'          : Chụp lưu ảnh đồng thời tất cả camera vào thư mục captures/
    - 'g'          : Chuyển đổi giữa chế độ Cửa sổ ghép (Grid) và Cửa sổ riêng
    - 'q' hoặc ESC : Thoát an toàn
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
    HOMOGRAPHY_FILE, CAPTURES_DIR, MAX_CAMERAS,
    MASTER_CAM, DRAW_QUAD_POLYGON, TRACK_DIST_CM
)
from camera_driver import build_all_cameras, discover_cameras
from detector import ObjectDetector
from coordinator import MasterSlaveCoordinator
from calibrate import load_homographies, save_homographies, calibrate_camera_interactive


def draw_label(img, text, pos, color=(0, 255, 0), font_scale=0.9, thickness=2):
    """Vẽ nhãn chữ có nền đen tương phản cao, chống lóa."""
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 8, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def main():
    print("=" * 70)
    print("  HỆ THỐNG ĐỒNG BỘ ĐA CAMERA 2D (MASTER-SLAVE BBOX & ID PROJECTION)")
    print("=" * 70)

    # 1. Liệt kê và mở các camera
    available_devices = discover_cameras()
    if not available_devices:
        print("Lỗi: Không tìm thấy camera công nghiệp nào trên mạng.")
        return

    print(f"\nTìm thấy {len(available_devices)} camera công nghiệp:")
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

    # Xác định Master Camera (mặc định là cam1 hoặc camera đầu tiên)
    master_name = MASTER_CAM if MASTER_CAM in cams else list(cams.keys())[0]
    slave_names = [name for name in cams if name != master_name]

    print(f"\n[Cấu hình]")
    print(f"  - Master Camera (Mask Khử Bóng & Detect): [{master_name.upper()}] ({cams[master_name].cam_info['model']})")
    print(f"  - Slave Cameras (Nhận Box & ID): {[s.upper() for s in slave_names]}")

    # 2. Nạp Homography
    homographies = load_homographies(HOMOGRAPHY_FILE)
    missing = [name for name in cams if name not in homographies]
    if missing:
        print(f"\n[Cảnh báo] Các camera chưa được Calibrate Homography: {missing}")
        print("-> Nhấn phím 'c' trong khi chạy để bấm 4 điểm mốc hiệu chuẩn mặt bàn.")

    # 3. Khởi tạo Detector (Chỉ chạy trên Master Camera) & Coordinator
    detector_master = ObjectDetector()
    coordinator = MasterSlaveCoordinator(
        homographies=homographies,
        master_name=master_name,
        track_dist=TRACK_DIST_CM
    )

    # 4. Học nền cho Master Camera
    print("\n" + "=" * 60)
    print(f"ĐANG HỌC NỀN CHO MASTER CAMERA [{master_name.upper()}]...")
    print("Vui lòng giữ khung nhìn TRỐNG không có sản phẩm trong 3-5 giây...")
    detector_master.learn_background(cams[master_name], num_frames=15)
    print("Học nền hoàn tất! Đang chạy luồng nhận diện và đồng bộ Bbox/ID...")
    print("=" * 60)

    os.makedirs(CAPTURES_DIR, exist_ok=True)
    show_grid = True
    show_mask_debug = True

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

            # 1. Phát hiện vật thể & Thử quét Barcode TRÊN MASTER CAMERA
            master_obj_candidates = detector_master.detect_objects(frame_master)
            master_detections = []
            for (cx, cy), (x1, y1, x2, y2) in master_obj_candidates:
                crop_m = frame_master[y1:y2, x1:x2]
                barcode = detector_master.decode_barcode(crop_m)
                decoded_by = master_name if barcode else None
                master_detections.append(((cx, cy), barcode, (x1, y1, x2, y2), decoded_by))

            # 2. COORDINATOR: ĐỒNG BỘ ID & CHIẾU BBOX SANG SLAVE CAMERAS
            tracked_objects = coordinator.update(master_detections, slave_cam_names=slave_names)

            # 3. QUÉT BARCODE BỔ SUNG TRÊN TOÀN BỘ CÁC SLAVE CAMERAS
            for obj in tracked_objects:
                if obj.barcode is None:
                    for s_name in slave_names:
                        if s_name in raw_frames and s_name in obj.slave_projections:
                            quad_pts, aabb, _ = obj.slave_projections[s_name]
                            if aabb is not None:
                                ax1, ay1, ax2, ay2 = aabb
                                s_frame = raw_frames[s_name]
                                sh, sw = s_frame.shape[:2]
                                ax1 = max(0, min(sw - 1, ax1))
                                ay1 = max(0, min(sh - 1, ay1))
                                ax2 = max(0, min(sw, ax2))
                                ay2 = max(0, min(sh, ay2))

                                if (ax2 - ax1) > 20 and (ay2 - ay1) > 20:
                                    crop_s = s_frame[ay1:ay2, ax1:ax2]
                                    bc = detector_master.decode_barcode(crop_s)
                                    if bc:
                                        obj.barcode = bc
                                        obj.decoded_by = s_name
                                        break  # Đã decode thành công từ camera phụ này!

            # 4. VẼ KẾT QUẢ LÊN TỪNG CAMERA
            display_canvases = []
            for name, cam in cams.items():
                f = raw_frames.get(name)
                disp_w, disp_h = 640, 480

                if f is not None:
                    disp = f.copy()

                    # VẼ TRÊN MASTER CAMERA (Cam 1)
                    if name == master_name:
                        for obj in tracked_objects:
                            x1, y1, x2, y2 = obj.master_bbox
                            color = (0, 255, 0) if obj.barcode else (0, 200, 255)
                            
                            # Vẽ Bounding Box
                            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)

                            # Nhãn ID + Barcode (Kèm tên Camera đọc được)
                            label = f"ID: {obj.id}"
                            if obj.barcode:
                                cam_src = f"[{obj.decoded_by.upper()}] " if getattr(obj, 'decoded_by', None) else ""
                                label += f" | {cam_src}{obj.barcode}"
                            pos_label = f"Pos: ({obj.table_xy[0]:.1f}, {obj.table_xy[1]:.1f})cm"

                            draw_label(disp, label, (x1, max(30, y1 - 25)), color=color, font_scale=0.9, thickness=2)
                            draw_label(disp, pos_label, (x1, max(50, y1)), color=(255, 255, 255), font_scale=0.75, thickness=2)

                        # Vẽ ô xem trước Mask khử bóng ở góc dưới Master Cam
                        if show_mask_debug and detector_master.last_mask is not None:
                            mask_bgr = cv2.cvtColor(detector_master.last_mask, cv2.COLOR_GRAY2BGR)
                            mh, mw = mask_bgr.shape[:2]
                            pw, ph = 180, int(180 * (mh / mw))
                            mask_prev = cv2.resize(mask_bgr, (pw, ph))
                            disp_h_orig, disp_w_orig = disp.shape[:2]
                            px0 = disp_w_orig - pw - 12
                            py0 = disp_h_orig - ph - 12
                            disp[py0:py0+ph, px0:px0+pw] = mask_prev
                            cv2.rectangle(disp, (px0, py0), (px0 + pw, py0 + ph), (0, 255, 0), 1)
                            cv2.putText(disp, "SHADOW-FREE MASK ('m')", (px0 + 5, py0 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

                    # VẼ TRÊN SLAVE CAMERAS (Cam 2, Cam 3, Cam 4)
                    else:
                        for obj in tracked_objects:
                            proj_data = obj.slave_projections.get(name)
                            if proj_data is None:
                                continue
                            quad_pts, aabb, center_pt = proj_data
                            color = (0, 255, 0) if obj.barcode else (0, 215, 255)

                            # Nếu có Đa giác 4 góc xoay chiếu từ Cam 1 sang
                            if quad_pts is not None and DRAW_QUAD_POLYGON:
                                # Vẽ đa giác bao quanh vật thể
                                cv2.polylines(disp, [quad_pts], isClosed=True, color=color, thickness=3)
                                if center_pt:
                                    cv2.circle(disp, (int(center_pt[0]), int(center_pt[1])), 5, (0, 0, 255), -1)
                                label_pos = (quad_pts[0][0], quad_pts[0][1])
                            elif aabb is not None:
                                ax1, ay1, ax2, ay2 = aabb
                                cv2.rectangle(disp, (ax1, ay1), (ax2, ay2), color, 3)
                                label_pos = (ax1, ay1)
                            else:
                                continue

                            # Hiển thị đúng ID, Barcode và Tọa độ bàn từ Cam 1
                            label = f"[SYNC] ID: {obj.id}"
                            if obj.barcode:
                                cam_src = f"[{obj.decoded_by.upper()}] " if getattr(obj, 'decoded_by', None) else ""
                                label += f" | {cam_src}{obj.barcode}"
                            pos_label = f"Pos: ({obj.table_xy[0]:.1f}, {obj.table_xy[1]:.1f})cm"

                            draw_label(disp, label, (label_pos[0], max(30, label_pos[1] - 25)), color=color, font_scale=0.9, thickness=2)
                            draw_label(disp, pos_label, (label_pos[0], max(50, label_pos[1])), color=(255, 255, 255), font_scale=0.75, thickness=2)

                    # Scale cho khung hiển thị
                    h, w = disp.shape[:2]
                    scale = min(disp_w / w, disp_h / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(disp, (nw, nh))

                    canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    xo = (disp_w - nw) // 2
                    yo = (disp_h - nh) // 2
                    canvas[yo:yo+nh, xo:xo+nw] = resized

                    # Header thông tin camera
                    is_m = (name == master_name)
                    role_str = "MASTER (COLOR)" if is_m else "SLAVE (MONO)"
                    header_color = (0, 255, 0) if is_m else (0, 200, 255)
                    header = f"[{name.upper()}: {role_str}] {cam.cam_info['model']} | FPS: {cam.fps:.1f}/{cam.target_fps:.1f}"
                    cv2.rectangle(canvas, (0, 0), (disp_w, 30), (20, 20, 20), -1)
                    cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, header_color, 1, cv2.LINE_AA)
                    display_canvases.append((name, canvas, f))
                else:
                    blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    cv2.putText(blank, f"[{name}] Dang cho frame...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    display_canvases.append((name, blank, None))

            # 4. HIỂN THỊ GIAO DIỆN
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
                            row2_items.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    row2 = np.hstack(row2_items)
                    grid_view = np.vstack([row1, row2])
                else:
                    grid_view = np.hstack([item[1] for item in display_canvases[:3]])

                cv2.imshow("MultiCamera 2.0 - Master-Slave ID & Bbox Synchronizer", grid_view)
            else:
                for name, canvas, _ in display_canvases:
                    cv2.imshow(f"Camera [{name}]", canvas)

            # 5. XỬ LÝ PHÍM TẮT
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:
                break
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
                        cv2.destroyWindow("MultiCamera 2.0 - Master-Slave ID & Bbox Synchronizer")
                    except Exception:
                        pass
                print(f"-> Chế độ hiển thị: {'Grid View' if show_grid else 'Cửa sổ riêng'}")
            elif key in (ord('m'), ord('M')):
                show_mask_debug = not show_mask_debug
                print(f"-> Xem trước Mask khử bóng: {'BẬT' if show_mask_debug else 'TẮT'}")
            elif key in (ord('s'), ord('S')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                saved_count = 0
                for name, _, raw in display_canvases:
                    if raw is not None:
                        fp = os.path.join(CAPTURES_DIR, f"{name}_{ts}.png")
                        cv2.imwrite(fp, raw)
                        saved_count += 1
                print(f"[Snapshot] Đã lưu {saved_count} ảnh vào thư mục: {CAPTURES_DIR}")
            elif key in (ord('r'), ord('R')):
                print(f"\n[Học lại nền] Đang học lại nền cho Master Camera [{master_name.upper()}]...")
                detector_master.learn_background(cams[master_name], num_frames=15)
                print("[Học lại nền] Hoàn tất.")
            elif key in (ord('c'), ord('C')):
                print("\n[Calibrate 2D] Bắt đầu hiệu chuẩn Homography 2D cho các camera...")
                for name, cam in cams.items():
                    H = calibrate_camera_interactive(name, cam.read)
                    if H is not None:
                        homographies[name] = H
                save_homographies(homographies, HOMOGRAPHY_FILE)
                coordinator.H = homographies
                print("[Calibrate 2D] Đã cập nhật ma trận Homography mới cho Coordinator.")

    except KeyboardInterrupt:
        print("\nNhận tín hiệu dừng...")
    finally:
        print("\nĐang dừng tất cả camera và giải phóng tài nguyên...")
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
        print("Hoàn tất dọn dẹp và đóng hệ thống an toàn.")


if __name__ == "__main__":
    main()
