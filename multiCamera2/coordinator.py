# -*- coding: utf-8 -*-
"""
BỘ ĐIỀU PHỐI ĐỒNG BỘ ID & CHIẾU BBOX 2D TỪ CAM 1 SANG CÁC CAMERA SAU (HOMOGRAPHY PROJECTION)
============================================================================================
Nguyên lý hoạt động (Master-Slave Architecture):
  1. Camera 1 (Color - Master):
     - Trừ nền / mask vật thể, trích xuất Bounding Box [x1, y1, x2, y2].
     - Đọc mã vạch (Barcode / QR) từ ảnh màu nét.
     - Cấp phát và theo dõi ID ổn định theo thời gian (ID 1, ID 2, ...).

  2. Camera 2, 3, 4 (Slave Cameras):
     - KHÔNG cần phải tự trừ nền (khử sạch 100% nhiễu nền hay bóng).
     - Toàn bộ Bounding Box từ Cam 1 được CHIẾU TRỰC TIẾP sang Cam 2/3/4 qua ma trận:
           H_{1 -> i} = inv(H_i) @ H_1
     - 4 góc của Bounding Box trên Cam 1 được ánh xạ thành 1 Đa giác (Quad Polygon)
       hoặc Hộp bao (AABB) ôm khít lấy vật thể trên Cam 2/3/4.
     - ID và Barcode được đồng bộ 100% khớp với Cam 1!
"""

import time
import numpy as np

from config import MERGE_DIST_CM, TRACK_DIST_CM, FORGET_AFTER_SEC


def image_to_table(H, x, y):
    """Chuyển đổi điểm ảnh (x, y) sang tọa độ mặt bàn (X, Y) qua ma trận H (3x3)."""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


def get_cam_to_cam_matrix(H_src, H_dst):
    """
    Tính ma trận biến đổi trực tiếp từ Camera Nguồn (src) sang Camera Đích (dst):
        H_{src -> dst} = inv(H_dst) @ H_src
    """
    if H_src is None or H_dst is None:
        return None
    try:
        H_dst_inv = np.linalg.inv(H_dst)
        return H_dst_inv @ H_src
    except Exception:
        return None


def project_point(H_mat, x, y):
    """Chiếu 1 điểm ảnh (x, y) qua ma trận biến đổi H_mat."""
    if H_mat is None:
        return None
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H_mat @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


def project_bbox_to_slave_cam(H_master_to_slave, bbox):
    """
    Chiếu Bounding Box [x1, y1, x2, y2] từ Cam Master sang Cam Slave.
    Trả về:
      - quad_pts: 4 góc đa giác (polygon) trên Cam Slave: shape (4, 2)
      - aabb: Hộp bao chữ nhật ngoại tiếp trên Cam Slave: (min_x, min_y, max_x, max_y)
      - center: Tọa độ tâm (cx, cy) trên Cam Slave
    """
    if H_master_to_slave is None or bbox is None:
        return None, None, None

    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    projected = []
    for (px, py) in corners:
        pt = project_point(H_master_to_slave, px, py)
        if pt is None:
            return None, None, None
        projected.append(pt)

    quad_pts = np.array(projected, dtype=np.float32)
    min_x = int(np.min(quad_pts[:, 0]))
    max_x = int(np.max(quad_pts[:, 0]))
    min_y = int(np.min(quad_pts[:, 1]))
    max_y = int(np.max(quad_pts[:, 1]))

    cx = float(np.mean(quad_pts[:, 0]))
    cy = float(np.mean(quad_pts[:, 1]))

    return quad_pts.astype(np.int32), (min_x, min_y, max_x, max_y), (cx, cy)


class MasterTrackedObject:
    """Đối tượng được theo dõi xuất phát từ Master Camera (Cam 1)."""

    def __init__(self, obj_id, img_bbox, table_xy, barcode=None):
        self.id = obj_id
        self.master_bbox = img_bbox          # (x1, y1, x2, y2) trên Cam 1
        self.table_xy = table_xy              # (X, Y) trên mặt bàn cm
        self.barcode = barcode
        self.last_seen = time.time()
        
        # Lưu kết quả chiếu sang các camera phụ: { cam_name: (quad_polygon, aabb_bbox, (cx, cy)) }
        self.slave_projections = {}

    def update(self, img_bbox, table_xy, barcode=None):
        self.master_bbox = img_bbox
        self.table_xy = table_xy
        if barcode and self.barcode is None:
            self.barcode = barcode
        self.last_seen = time.time()


class MasterSlaveCoordinator:
    """
    Bộ điều phối: Nhận phát hiện từ Master Cam (Cam 1), duy trì ID ổn định,
    và tự động chiếu Bounding Box + ID sang toàn bộ các camera phụ (Cam 2, 3, 4).
    """

    def __init__(self, homographies=None, master_name="cam1",
                 track_dist=TRACK_DIST_CM, forget_after=FORGET_AFTER_SEC):
        self.H = homographies if homographies else {}
        self.master_name = master_name
        self.track_dist = track_dist
        self.forget_after = forget_after
        self.objects = {}
        self._next_id = 1

    def update(self, master_detections, slave_cam_names=None):
        now = time.time()
        H_master = self.H.get(self.master_name)

        # 1. Chuyển đổi các phát hiện của Master Cam sang tọa độ mặt bàn
        current_dets = []
        for (cx, cy), barcode, bbox in master_detections:
            t_xy = None
            if H_master is not None:
                t_xy = image_to_table(H_master, cx, cy)
            if t_xy is None:
                t_xy = (cx / 10.0, cy / 10.0)
            current_dets.append((t_xy, barcode, bbox, (cx, cy)))

        # 2. Tracking ID trên Master Camera qua các frame
        used_ids = set()
        for (t_xy, barcode, bbox, (cx, cy)) in current_dets:
            best_id, best_d = None, self.track_dist
            for oid, obj in self.objects.items():
                if oid in used_ids:
                    continue
                dd = np.hypot(obj.table_xy[0] - t_xy[0], obj.table_xy[1] - t_xy[1])
                if dd < best_d:
                    best_d, best_id = dd, oid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.objects[best_id] = MasterTrackedObject(best_id, bbox, t_xy, barcode)
            else:
                self.objects[best_id].update(bbox, t_xy, barcode)

            used_ids.add(best_id)

        # 3. Xóa các vật thể đã rời khỏi khung nhìn quá lâu
        for oid in [o for o, ob in self.objects.items() if now - ob.last_seen > self.forget_after]:
            del self.objects[oid]

        # 4. CHIẾU TỌA ĐỘ VÀ BOUNDING BOX SANG TOÀN BỘ SLAVE CAMERAS
        if slave_cam_names:
            for slave_name in slave_cam_names:
                H_slave = self.H.get(slave_name)
                H_m2s = get_cam_to_cam_matrix(H_master, H_slave)

                for obj in self.objects.values():
                    if H_m2s is not None:
                        quad, aabb, c_slave = project_bbox_to_slave_cam(H_m2s, obj.master_bbox)
                        obj.slave_projections[slave_name] = (quad, aabb, c_slave)
                    else:
                        # Nếu chưa có Homography, dùng lại bbox gốc của cam 1
                        obj.slave_projections[slave_name] = (None, obj.master_bbox, None)

        return list(self.objects.values())
