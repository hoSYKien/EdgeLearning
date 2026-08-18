"""
COORDINATOR GHÉP ID ĐA CAMERA (theo toạ độ MẶT BÀN dùng chung)
==============================================================
Nguyên tắc:
  1) Mỗi camera có 1 ma trận HOMOGRAPHY H quy toạ độ ẢNH -> toạ độ MẶT BÀN
     (đơn vị: mm hoặc cm, do bạn chọn khi calibrate).
  2) Mỗi frame, mỗi camera phát hiện các vật -> gửi lên coordinator:
        (tâm_ảnh_x, tâm_ảnh_y, barcode_or_None)
  3) Coordinator dùng H quy tâm về MẶT BÀN -> gộp các vật GẦN NHAU
     (< MERGE_DIST) từ mọi camera thành 1 vật chung -> 1 ID.
  4) Vật nào có barcode (dù chỉ 1 camera đọc được) -> cả nhóm mang barcode đó.
  5) TRACK theo thời gian: vật ở gần vị trí frame trước -> giữ nguyên ID.

=> Camera bị KHUẤT barcode vẫn biết vật đó là ID mấy nhờ VỊ TRÍ trùng khớp.
"""

import time
import numpy as np


# ============================================================
# HOMOGRAPHY: ảnh -> mặt bàn
# ============================================================

def image_to_table(H, x, y):
    """Quy 1 điểm ảnh (x,y) sang toạ độ mặt bàn qua ma trận H (3x3)."""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


# ============================================================
# 1 PHÁT HIỆN TỪ 1 CAMERA
# ============================================================

class Detection:
    def __init__(self, cam_name, img_xy, table_xy, barcode=None, bbox=None):
        self.cam_name = cam_name
        self.img_xy = img_xy        # (x,y) trên ảnh camera đó
        self.table_xy = table_xy    # (X,Y) trên mặt bàn (đã quy đổi)
        self.barcode = barcode
        self.bbox = bbox            # (x1,y1,x2,y2) trên ảnh camera đó


# ============================================================
# 1 VẬT ĐÃ GỘP (có ID chung)
# ============================================================

class TrackedObject:
    def __init__(self, obj_id, table_xy):
        self.id = obj_id
        self.table_xy = table_xy    # vị trí trung bình trên mặt bàn
        self.barcode = None
        self.seen_by = set()        # tên các camera đang thấy vật này
        self.last_update = time.time()
        self.per_cam_bbox = {}      # cam_name -> bbox (để vẽ trên từng cam)

    def update(self, dets):
        """Cập nhật từ danh sách Detection thuộc cùng vật này (frame hiện tại)."""
        pts = [d.table_xy for d in dets]
        self.table_xy = (float(np.mean([p[0] for p in pts])),
                         float(np.mean([p[1] for p in pts])))
        self.seen_by = set(d.cam_name for d in dets)
        self.per_cam_bbox = {d.cam_name: d.bbox for d in dets if d.bbox}
        # barcode: giữ cái đã có; nếu chưa có mà frame này có -> nhận
        for d in dets:
            if d.barcode:
                # nếu chưa có barcode, hoặc trùng -> ok; nếu khác -> ưu tiên cái cũ
                if self.barcode is None:
                    self.barcode = d.barcode
        self.last_update = time.time()


# ============================================================
# COORDINATOR
# ============================================================

class MultiCamCoordinator:

    def __init__(self, homographies,
                 merge_dist=30.0,      # cm/mm: 2 phát hiện < ngưỡng -> cùng vật
                 track_dist=40.0,      # cm/mm: vật gần vị trí cũ -> giữ ID
                 forget_after=1.0):    # giây: không thấy quá lâu -> xoá
        """
        homographies: dict {cam_name: H (3x3 np.array)}.
        merge_dist  : ngưỡng gộp giữa CÁC CAMERA (nên > sai số dư homography,
                      < khoảng cách giữa 2 vật khác nhau gần nhất).
        """
        self.H = homographies
        self.merge_dist = merge_dist
        self.track_dist = track_dist
        self.forget_after = forget_after
        self.objects = {}     # id -> TrackedObject
        self._next_id = 1

    # --------------------------------------------------------
    def _project(self, cam_name, dets_raw):
        """Quy các phát hiện thô của 1 camera sang toạ độ mặt bàn."""
        H = self.H.get(cam_name)
        out = []
        for (img_xy, barcode, bbox) in dets_raw:
            if H is None:
                continue
            t = image_to_table(H, img_xy[0], img_xy[1])
            if t is None:
                continue
            out.append(Detection(cam_name, img_xy, t, barcode, bbox))
        return out

    # --------------------------------------------------------
    def _cluster(self, all_dets):
        """
        Gộp các Detection (từ mọi camera) theo khoảng cách trên mặt bàn.
        Trả list các cụm; mỗi cụm là list[Detection] -> 1 vật.
        """
        clusters = []
        for d in all_dets:
            placed = False
            for cl in clusters:
                # so với tâm hiện tại của cụm
                cx = np.mean([e.table_xy[0] for e in cl])
                cy = np.mean([e.table_xy[1] for e in cl])
                if np.hypot(d.table_xy[0] - cx, d.table_xy[1] - cy) < self.merge_dist:
                    cl.append(d); placed = True; break
            if not placed:
                clusters.append([d])
        return clusters

    # --------------------------------------------------------
    def update(self, per_cam_detections):
        """
        per_cam_detections: dict {cam_name: [ (img_xy, barcode, bbox), ... ]}
        Trả list[TrackedObject] hiện hành.
        """
        # 1) quy mọi phát hiện về mặt bàn
        all_dets = []
        for cam_name, dets_raw in per_cam_detections.items():
            all_dets += self._project(cam_name, dets_raw)

        # 2) gộp chéo camera -> các cụm (mỗi cụm = 1 vật ở 1 vị trí)
        clusters = self._cluster(all_dets)

        # 3) gán cụm vào object đang track (theo vị trí gần nhất) hoặc tạo mới
        used_ids = set()
        now = time.time()
        for cl in clusters:
            cx = float(np.mean([e.table_xy[0] for e in cl]))
            cy = float(np.mean([e.table_xy[1] for e in cl]))

            # tìm object cũ gần nhất chưa dùng
            best_id, best_d = None, self.track_dist
            for oid, obj in self.objects.items():
                if oid in used_ids:
                    continue
                dd = np.hypot(obj.table_xy[0] - cx, obj.table_xy[1] - cy)
                if dd < best_d:
                    best_d, best_id = dd, oid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.objects[best_id] = TrackedObject(best_id, (cx, cy))

            self.objects[best_id].update(cl)
            used_ids.add(best_id)

        # 4) quên các object lâu không thấy
        for oid in [o for o, ob in self.objects.items()
                    if now - ob.last_update > self.forget_after]:
            del self.objects[oid]

        return list(self.objects.values())
