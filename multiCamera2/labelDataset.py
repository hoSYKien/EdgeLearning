# -*- coding: utf-8 -*-
"""
CÔNG CỤ DUYỆT & CHỈNH SỬA DATASET (YOLO format)

Tính năng:
    - Zoom in/out bằng con lăn chuột (zoom đúng tại vị trí con trỏ)
    - Kéo chuột PHẢI để pan (di chuyển khung nhìn)
    - Kéo chuột TRÁI trên vùng trống -> vẽ box mới (theo class hiện tại)
    - Click vào 1 box -> CHỌN box đó (viền dày hơn để nhận biết)
    - Kéo box đã chọn (giữ giữa box) -> DI CHUYỂN box
    - Kéo góc box đã chọn -> THAY ĐỔI KÍCH THƯỚC box
    - Phím số 0-9 -> đổi "class hiện tại" (dùng cho box mới) VÀ đổi class
      của box đang được chọn (nếu có)
    - Phím Delete hoặc X -> xoá box đang chọn
    - Phím C (2 kiểu dùng):
      * BẤM RỒI THẢ NGAY (không giữ, không click) -> nếu ĐÃ có box được ghi
        nhớ từ trước (ở ảnh nào đó trước đây) thì TỰ ĐỘNG dán hết các box
        đó vào ảnh hiện tại luôn (kiểu thêm vào, không xoá box đang có,
        bỏ qua box nào trùng/lệch nhỏ). Dùng cho các ảnh SAU khi đã biết
        chắc box nào cần lặp lại (VD box xe đậu yên).
      * GIỮ C (nhấn và giữ) -> hiện box "bóng mờ" đè lên ảnh hiện tại để
        CLICK CHỌN. Nếu chưa ghi nhớ box nào thì hiện toàn bộ box của ảnh
        gần nhất để chọn lần đầu; nếu đã ghi nhớ rồi thì chỉ hiện đúng các
        box đã ghi nhớ đó. Click vào 1 box mờ -> dán vào ảnh hiện tại VÀ
        ghi nhớ lại (nếu chưa có trong bộ nhớ) để dùng cho phím C sau này.
    - Nút RESET GHI NHỚ (hoặc phím V) -> xoá sạch bộ nhớ box đã ghi nhớ ở
      trên (VD khi xe đậu đó đã rời đi, không cần nhớ box của nó nữa, và
      muốn dùng GIỮ C để chọn/ghi nhớ đối tượng khác).
      Sau khi reset, giữ C sẽ quay lại hiện toàn bộ box của ảnh gần nhất
      như lúc chưa ghi nhớ gì, và bấm-thả nhanh C sẽ không dán gì cả cho
      tới khi bạn ghi nhớ box mới.
    - Phím T -> LOAD MODEL (YOLO - ultralytics, chỉ load 1 lần đầu tiên rồi
      dùng lại) và CHẠY LẠI model trên ẢNH HIỆN TẠI với NGƯỠNG CONFIDENCE
      THẤP (mặc định 0.2, xem MODEL_RERUN_CONF bên dưới) để bắt thêm các
      box mà lần auto-label ban đầu (ngưỡng cao hơn) đã bỏ sót. Các box
      model tìm được sẽ được THÊM VÀO danh sách box hiện tại (không xoá
      box đang có), bỏ qua box nào trùng/lệch nhỏ (IoU) so với box đã có
      sẵn - y hệt cách xử lý trùng lặp của phím C ở trên.
    - Nút APPROVE (hoặc phím A) -> lưu ảnh + label hiện tại vào thư mục APPROVE,
      xoá khỏi thư mục nguồn, chuyển ảnh tiếp theo
    - Nút REJECT (hoặc phím R) -> chuyển ảnh + label vào thư mục REJECT,
      xoá khỏi thư mục nguồn, chuyển ảnh tiếp theo
    - Phím SPACE / mũi tên phải -> xem tiếp ảnh sau (KHÔNG quyết định, ảnh vẫn ở nguồn)
    - Phím B / mũi tên trái -> quay lại ảnh trước (chỉ xem lại được nếu ảnh đó
      chưa bị approve/reject, vì ảnh sẽ bị CHUYỂN đi khỏi thư mục nguồn)
    - Q / ESC -> thoát

Cài thư viện:
    pip install opencv-python keyboard ultralytics

Chạy:
    python dataset_reviewer.py
"""

import os
import shutil
import ctypes
from ctypes import wintypes
import cv2
import numpy as np

HAS_KEYBOARD = False
try:
    import keyboard   # pip install keyboard - bắt phím toàn hệ thống
    HAS_KEYBOARD = True
except ImportError:
    pass

WINDOW_NAME = "Dataset Reviewer"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
SW_SHOW = 5


def bring_window_to_front():
    """Ép cửa sổ lấy lại focus bàn phím một cách CHẮC CHẮN, kể cả khi chuyển ảnh
    (không phải do người dùng vừa click) - Windows chặn SetForegroundWindow đơn giản
    trong trường hợp này (foreground lock), nên phải gắn tạm luồng bàn phím hiện tại
    vào cửa sổ đích (AttachThreadInput) rồi mới ép được."""
    try:
        hwnd = user32.FindWindowW(None, WINDOW_NAME)
        if not hwnd:
            return

        fg_hwnd = user32.GetForegroundWindow()
        target_thread = user32.GetWindowThreadProcessId(hwnd, None)
        current_thread = kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0

        if fg_thread and fg_thread != target_thread:
            user32.AttachThreadInput(target_thread, fg_thread, True)
        if current_thread != target_thread:
            user32.AttachThreadInput(target_thread, current_thread, True)

        user32.ShowWindow(hwnd, SW_SHOW)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)

        if fg_thread and fg_thread != target_thread:
            user32.AttachThreadInput(target_thread, fg_thread, False)
        if current_thread != target_thread:
            user32.AttachThreadInput(target_thread, current_thread, False)
    except Exception:
        pass

# ==================== CẤU HÌNH ====================
# Thư mục nguồn cần duyệt - PHẢI có cấu trúc images/ + labels/
SOURCE_DIR = r"D:\TongHop\RTC Technologi\RD\Tutorial\code2\multiCamera2\dataset_yolo"

# 2 thư mục đích - script tự tạo cấu trúc images/ + labels/ bên trong
APPROVE_DIR = r"D:\TongHop\RTC Technologi\RD\Tutorial\code2\multiCamera2\dataset_yolo\train"
REJECT_DIR = r"D:\TongHop\RTC Technologi\RD\Tutorial\code2\multiCamera2\dataset_yolo\val"

CLASS_NAMES = {0: "object"}
CLASS_COLORS = {0: (0, 200, 0)}  # BGR

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CANVAS_W, CANVAS_H = 1280, 800   # kích thước cửa sổ hiển thị
HANDLE_PX = 10                     # bán kính (px trên canvas) để bắt trúng góc box khi resize
MIN_BOX_PX = 4                     # kích thước tối thiểu (px ảnh gốc) để tính là 1 box hợp lệ

# ---- Cấu hình MODEL để dùng cho phím T (chạy lại model với ngưỡng thấp) ----
MODEL_PATH = r"D:\TongHop\RTC Technologi\G8\model\modelYolo\best.pt"  # SỬA đường dẫn tới file .pt của bạn
MODEL_RERUN_CONF = 0.1   # ngưỡng confidence THẤP dùng khi bấm T để bắt thêm box bị sót
# LƯU Ý: id class model trả về (0,1,2,...) phải khớp thứ tự với CLASS_NAMES ở trên,
# nếu model của bạn train với thứ tự class khác thì sửa lại MODEL_CLASS_ID_MAP bên dưới,
# vd MODEL_CLASS_ID_MAP = {0: 2, 1: 0, 2: 1}  (id model -> id trong CLASS_NAMES)
MODEL_CLASS_ID_MAP = None
# ==================================================

BTN_APPROVE = (20, CANVAS_H - 55, 180, CANVAS_H - 15)
BTN_REJECT = (200, CANVAS_H - 55, 360, CANVAS_H - 15)
BTN_PREV = (380, CANVAS_H - 55, 480, CANVAS_H - 15)
BTN_NEXT = (500, CANVAS_H - 55, 600, CANVAS_H - 15)
BTN_DELETE = (620, CANVAS_H - 55, 760, CANVAS_H - 15)
BTN_RESET_MEMORY = (620, CANVAS_H - 90, 760, CANVAS_H - 60)   # xoá bộ nhớ box đã ghi nhớ (copied_boxes_norm)
BTN_RERUN_MODEL = (780, CANVAS_H - 90, 940, CANVAS_H - 60)    # chạy lại model với ngưỡng thấp (phím T)
# Nút chọn class - tự sinh theo CLASS_NAMES, xếp hàng ngang bắt đầu từ x=780
BTN_CLASSES = {
    cid: (780 + i * 110, CANVAS_H - 55, 780 + i * 110 + 100, CANVAS_H - 15)
    for i, cid in enumerate(sorted(CLASS_NAMES))
}


def find_images(images_dir):
    return sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS))


def load_labels(label_path, img_w, img_h):
    """Đọc label .txt YOLO -> list box pixel [cls,x1,y1,x2,y2]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            boxes.append([cid, x1, y1, x2, y2])
    return boxes


def save_labels(label_path, boxes, img_w, img_h):
    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def boxes_to_normalized(boxes, img_w, img_h):
    """[cid,x1,y1,x2,y2] pixel -> [cid,xc,yc,bw,bh] tỉ lệ (0..1) - để lưu tạm và
    tái sử dụng được ở ảnh KHÁC kích thước (không phụ thuộc img_w/img_h gốc)."""
    return [
        [cid, (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]
        for cid, x1, y1, x2, y2 in boxes
    ]


def boxes_from_normalized(norm_boxes, img_w, img_h):
    """[cid,xc,yc,bw,bh] tỉ lệ -> [cid,x1,y1,x2,y2] pixel theo kích thước ảnh HIỆN TẠI,
    có kẹp lại trong biên ảnh phòng trường hợp ảnh trước/sau lệch kích thước."""
    result = []
    for cid, xc, yc, bw, bh in norm_boxes:
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        x1 = min(max(0, x1), img_w)
        y1 = min(max(0, y1), img_h)
        x2 = min(max(0, x2), img_w)
        y2 = min(max(0, y2), img_h)
        result.append([cid, x1, y1, x2, y2])
    return result


OVERLAP_IOU_THRESHOLD = 0.3  # box cũ có IoU >= ngưỡng này với 1 box đang có -> coi là trùng/lệch nhỏ, bỏ qua không dán


def box_iou(a, b):
    """IoU giữa 2 box pixel [cid,x1,y1,x2,y2] - không quan tâm class, chỉ so vị trí/kích thước."""
    _, ax1, ay1, ax2, ay2 = a
    _, bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_one_box(box, img_w, img_h):
    """Chuẩn hoá 1 box pixel [cid,x1,y1,x2,y2] -> [cid,xc,yc,bw,bh] tỉ lệ 0..1."""
    return boxes_to_normalized([box], img_w, img_h)[0]


def compute_ghost_source_norm(state):
    """Tính (snapshot) nguồn box để hiện 'bóng mờ' khi giữ C - gọi 1 LẦN khi mở ảnh mới.
    ƯU TIÊN box đã từng được ghi nhớ (copied_boxes_norm) - vì đây mới đúng là những box người
    dùng thật sự cần dùng lại (VD xe đậu yên). Chỉ khi CHƯA ghi nhớ box nào mới tạm dùng toàn bộ
    box của ảnh gần nhất để có cái mà chọn lần đầu. Trả về BẢN SAO (list mới) để không bị ảnh
    hưởng khi copied_boxes_norm được thêm box mới trong lúc đang xem ảnh này."""
    source = state.copied_boxes_norm if state.copied_boxes_norm else state.last_boxes_norm
    return list(source) if source else None


def reset_copied_memory(state):
    """Xoá sạch bộ nhớ box đã ghi nhớ (copied_boxes_norm) và tính lại ghost_source_norm ngay
    (để có hiệu lực tức thì trên ảnh đang xem, không cần đợi sang ảnh khác)."""
    state.copied_boxes_norm = []
    state.ghost_source_norm = compute_ghost_source_norm(state)


def paste_all_remembered(state, img_w, img_h):
    """Dán TOÀN BỘ box đã ghi nhớ (copied_boxes_norm) vào ảnh hiện tại - kiểu THÊM VÀO
    (không xoá box đang có), bỏ qua box nào trùng/lệch nhỏ so với box đã có sẵn."""
    if not state.copied_boxes_norm:
        return 0
    candidates = boxes_from_normalized(state.copied_boxes_norm, img_w, img_h)
    added = 0
    for cand in candidates:
        is_dup = any(box_iou(cand, existing) >= OVERLAP_IOU_THRESHOLD for existing in state.boxes)
        if not is_dup:
            state.boxes.append(cand)
            added += 1
    return added


# ==================== MODEL (chạy lại với ngưỡng thấp - phím T) ====================
_model_cache = {"model": None, "load_failed": False}


def get_model():
    """Load model YOLO (ultralytics) - CHỈ load 1 LẦN DUY NHẤT trong cả phiên chạy,
    những lần bấm T sau tái sử dụng luôn model đã load (đỡ mất thời gian chờ)."""
    if _model_cache["model"] is not None or _model_cache["load_failed"]:
        return _model_cache["model"]
    try:
        from ultralytics import YOLO
        print(f"⏳ Đang load model: {MODEL_PATH} ...")
        _model_cache["model"] = YOLO(MODEL_PATH)
        print("✅ Model đã sẵn sàng, có thể bấm T để chạy lại trên ảnh hiện tại.")
    except Exception as e:
        print(f"❌ Không load được model ({MODEL_PATH}): {e}")
        print("   Kiểm tra lại MODEL_PATH và đã cài 'pip install ultralytics' chưa.")
        _model_cache["load_failed"] = True
    return _model_cache["model"]


def rerun_model_on_image(state, img_path):
    """Phím T: chạy lại model trên ẢNH HIỆN TẠI với ngưỡng confidence THẤP (MODEL_RERUN_CONF)
    để bắt thêm các box mà lần auto-label ban đầu (thường chạy ngưỡng cao hơn) đã bỏ sót.
    Các box mới sẽ được THÊM VÀO danh sách box hiện tại (không xoá box đang có), bỏ qua box
    nào trùng/lệch nhỏ (IoU) so với box đã có sẵn - giống hệt cách xử lý của phím C."""
    model = get_model()
    if model is None:
        return 0

    print(f"⏳ Đang chạy lại model (conf >= {MODEL_RERUN_CONF}) trên: {os.path.basename(img_path)} ...")
    try:
        results = model.predict(source=img_path, conf=MODEL_RERUN_CONF, verbose=False)
    except Exception as e:
        print(f"❌ Lỗi khi chạy model: {e}")
        return 0

    added = 0
    if results:
        r = results[0]
        for b in r.boxes:
            cid = int(b.cls[0].item())
            if MODEL_CLASS_ID_MAP is not None:
                cid = MODEL_CLASS_ID_MAP.get(cid, cid)
            if cid not in CLASS_NAMES:
                continue  # bỏ qua class model biết mà file config (CLASS_NAMES) không dùng tới
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cand = [cid, x1, y1, x2, y2]
            is_dup = any(box_iou(cand, existing) >= OVERLAP_IOU_THRESHOLD for existing in state.boxes)
            if not is_dup:
                state.boxes.append(cand)
                added += 1

    print(f"✅ Đã thêm {added} box mới từ model (ngưỡng {MODEL_RERUN_CONF}).")
    return added
# =====================================================================================


class View:
    """Quản lý zoom (scale) + pan (offset) - quy đổi giữa toạ độ canvas <-> ảnh gốc."""

    def __init__(self, img_w, img_h):
        self.scale = min(CANVAS_W / img_w, CANVAS_H / img_h)
        self.offset_x = 0.0
        self.offset_y = 0.0

    def to_canvas(self, x, y):
        return (x - self.offset_x) * self.scale, (y - self.offset_y) * self.scale

    def to_orig(self, cx, cy):
        return cx / self.scale + self.offset_x, cy / self.scale + self.offset_y

    def zoom(self, cx, cy, factor):
        orig_x, orig_y = self.to_orig(cx, cy)
        self.scale = max(0.05, min(20.0, self.scale * factor))
        self.offset_x = orig_x - cx / self.scale
        self.offset_y = orig_y - cy / self.scale

    def pan(self, dx_canvas, dy_canvas):
        self.offset_x -= dx_canvas / self.scale
        self.offset_y -= dy_canvas / self.scale


class ReviewerState:
    def __init__(self):
        self.img = None
        self.img_w = 0
        self.img_h = 0
        self.boxes = []       # [cid, x1, y1, x2, y2] toạ độ pixel ảnh gốc
        self.view = None
        self.selected = -1
        self.current_class = 0

        self.mode = None      # None | "draw" | "move" | "resize" | "pan" | "click_approve" | "click_reject"
        self.mouse_canvas = (0, 0)
        self.drag_start_canvas = None
        self.drag_start_orig = None
        self.resize_handle = None   # "tl","tr","bl","br"
        self.box_orig_snapshot = None
        self.file_info = ""
        self.last_boxes_norm = None    # box (tỉ lệ 0..1) của ảnh GẦN NHẤT - chỉ dùng làm nguồn chọn LẦN ĐẦU khi chưa ghi nhớ box nào
        self.copied_boxes_norm = []    # box (tỉ lệ 0..1) ĐÃ TỪNG được chọn qua giữ C + click - nhớ xuyên suốt nhiều ảnh,
                                        # từ ảnh sau trở đi giữ C sẽ ưu tiên hiện đúng những box này thay vì cả ảnh cũ
        self.ghost_source_norm = None  # nguồn ghost đã CHỐT CỐ ĐỊNH cho ảnh đang xem - chỉ tính lại khi sang ảnh khác,
                                        # KHÔNG tính lại mỗi lần click (nếu không sẽ mất box còn lại để chọn tiếp)


def box_at_point(boxes, x, y):
    """Tìm box (theo thứ tự vẽ sau cùng ưu tiên) chứa điểm (x,y) toạ độ ảnh gốc."""
    for i in range(len(boxes) - 1, -1, -1):
        _, x1, y1, x2, y2 = boxes[i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return -1


def corner_handle_hit(view, box, cx, cy):
    """Trả về góc nào của box đang bị click gần đúng (trong bán kính HANDLE_PX px canvas)."""
    _, x1, y1, x2, y2 = box
    corners = {
        "tl": (x1, y1), "tr": (x2, y1),
        "bl": (x1, y2), "br": (x2, y2),
    }
    for name, (ox, oy) in corners.items():
        px, py = view.to_canvas(ox, oy)
        if abs(px - cx) <= HANDLE_PX and abs(py - cy) <= HANDLE_PX:
            return name
    return None


def render(state):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 40, dtype=np.uint8)
    view = state.view

    # Vẽ ảnh (crop vùng nhìn thấy theo scale/offset)
    x0, y0 = view.to_orig(0, 0)
    x1, y1 = view.to_orig(CANVAS_W, CANVAS_H)
    sx0, sy0 = max(0, int(x0)), max(0, int(y0))
    sx1, sy1 = min(state.img_w, int(np.ceil(x1))), min(state.img_h, int(np.ceil(y1)))

    if sx1 > sx0 and sy1 > sy0:
        crop = state.img[sy0:sy1, sx0:sx1]
        dst_w = int((sx1 - sx0) * view.scale)
        dst_h = int((sy1 - sy0) * view.scale)
        if dst_w > 0 and dst_h > 0:
            resized = cv2.resize(crop, (dst_w, dst_h))
            dst_x, dst_y = view.to_canvas(sx0, sy0)
            dst_x, dst_y = int(dst_x), int(dst_y)
            cx0, cy0 = max(0, dst_x), max(0, dst_y)
            cx1, cy1 = min(CANVAS_W, dst_x + dst_w), min(CANVAS_H, dst_y + dst_h)
            if cx1 > cx0 and cy1 > cy0:
                canvas[cy0:cy1, cx0:cx1] = resized[cy0 - dst_y:cy1 - dst_y, cx0 - dst_x:cx1 - dst_x]

    # Vẽ các box đã có (Viền siêu mảnh 1px, box đang chọn 2px)
    for i, (cid, bx1, by1, bx2, by2) in enumerate(state.boxes):
        p1 = tuple(map(int, view.to_canvas(bx1, by1)))
        p2 = tuple(map(int, view.to_canvas(bx2, by2)))
        color = CLASS_COLORS.get(cid, (0, 255, 255))
        thickness = 2 if i == state.selected else 1
        cv2.rectangle(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        label = CLASS_NAMES.get(cid, str(cid))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 3, p1[1] - 3)
        cv2.rectangle(canvas, (p1[0], ty - th - 2), (p1[0] + tw + 3, ty + baseline), (10, 10, 10), -1)
        cv2.putText(canvas, label, (p1[0] + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if i == state.selected:
            for hx, hy in [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]:
                hp = tuple(map(int, view.to_canvas(hx, hy)))
                cv2.circle(canvas, hp, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, hp, 3, color, 1, cv2.LINE_AA)

    # Preview box đang kéo vẽ
    if state.mode == "draw" and state.drag_start_orig is not None:
        ox0, oy0 = state.drag_start_orig
        ox1, oy1 = view.to_orig(*state.mouse_canvas)
        p1 = tuple(map(int, view.to_canvas(ox0, oy0)))
        p2 = tuple(map(int, view.to_canvas(ox1, oy1)))
        color = CLASS_COLORS.get(state.current_class, (0, 255, 255))
        cv2.rectangle(canvas, p1, p2, color, 1, lineType=cv2.LINE_AA)

    # Thanh trạng thái + nút bấm
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, CANVAS_H - 100), (CANVAS_W, CANVAS_H), (20, 20, 20), -1)
    canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

    def draw_button(rect, text, color):
        bx1, by1, bx2, by2 = rect
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(canvas, text, (bx1 + 12, by2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    draw_button(BTN_APPROVE, "APPROVE (A)", (0, 150, 0))
    draw_button(BTN_REJECT, "REJECT (R)", (0, 0, 200))
    draw_button(BTN_PREV, "< TRUOC", (90, 90, 90))
    draw_button(BTN_NEXT, "TIEP >", (90, 90, 90))
    draw_button(BTN_DELETE, "XOA BOX", (60, 60, 160))
    draw_button(BTN_RESET_MEMORY, "RESET GHI NHO (V)", (130, 70, 130))
    draw_button(BTN_RERUN_MODEL, "CHAY LAI MODEL (T)", (150, 110, 0))

    for cid, rect in BTN_CLASSES.items():
        color = CLASS_COLORS.get(cid, (120, 120, 120))
        is_current = (cid == state.current_class)
        bx1, by1, bx2, by2 = rect
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
        if is_current:
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        label = f"{cid}-{CLASS_NAMES.get(cid, cid)}"
        cv2.putText(canvas, label, (bx1 + 6, by2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    class_name = CLASS_NAMES.get(state.current_class, str(state.current_class))
    sel_name = "khong" if state.selected < 0 else CLASS_NAMES.get(state.boxes[state.selected][0], "?")
    info = (f"Class hien tai: {state.current_class}-{class_name} | So box: {len(state.boxes)} | "
            f"Dang chon: {sel_name} | Box da ghi nho: {len(state.copied_boxes_norm)}")
    cv2.putText(canvas, info, (20, CANVAS_H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tên ảnh + vị trí trong danh sách - vẽ trên canvas thay vì đổi tiêu đề cửa sổ
    # (đổi tiêu đề liên tục sẽ làm hỏng việc tìm cửa sổ để ép focus bàn phím)
    if state.file_info:
        cv2.putText(canvas, state.file_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, state.file_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # GIỮ C: hiện box "bóng mờ" đè lên - dùng nguồn đã CHỐT SẴN khi mở ảnh này (state.ghost_source_norm)
    if c_hold_state["held"] and state.ghost_source_norm:
        ghost_boxes = boxes_from_normalized(state.ghost_source_norm, state.img_w, state.img_h)
        ghost_layer = canvas.copy()
        for cid, gx1, gy1, gx2, gy2 in ghost_boxes:
            p1 = tuple(map(int, view.to_canvas(gx1, gy1)))
            p2 = tuple(map(int, view.to_canvas(gx2, gy2)))
            color = CLASS_COLORS.get(cid, (200, 200, 200))
            cv2.rectangle(ghost_layer, p1, p2, color, 1, lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(ghost_layer, 0.45, canvas, 0.55, 0)
        if state.copied_boxes_norm:
            hint = "DANG GIU C - BOX DA GHI NHO - CLICK DE DAN LAI SANG ANH NAY"
        else:
            hint = "DANG GIU C - BOX ANH TRUOC (CHUA GHI NHO GI) - CLICK 1 BOX DE GHI NHO"
        cv2.putText(canvas, hint, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, hint, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

    return canvas


def make_mouse_callback(state):
    def callback(event, x, y, flags, param):
        view = state.view
        state.mouse_canvas = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            bring_window_to_front()

            # Đang GIỮ C -> chế độ chọn box mờ (dùng nguồn đã chốt sẵn khi mở ảnh này), bỏ qua nút/vẽ/chọn khác
            if c_hold_state["held"]:
                if state.ghost_source_norm:
                    ghost_boxes = boxes_from_normalized(state.ghost_source_norm, state.img_w, state.img_h)
                    ox, oy = view.to_orig(x, y)
                    hit = box_at_point(ghost_boxes, ox, oy)
                    if hit >= 0:
                        cand = ghost_boxes[hit]
                        # Thêm vào ảnh hiện tại nếu chưa có box nào gần trùng
                        is_dup_in_image = any(box_iou(cand, existing) >= OVERLAP_IOU_THRESHOLD for existing in state.boxes)
                        if not is_dup_in_image:
                            state.boxes.append(cand)
                            state.selected = len(state.boxes) - 1
                        # GHI NHỚ box này để các ảnh SAU chỉ hiện đúng box đã chọn, không phải cả ảnh cũ nữa
                        cand_norm = normalize_one_box(cand, state.img_w, state.img_h)
                        remembered_pixel = boxes_from_normalized(state.copied_boxes_norm, state.img_w, state.img_h)
                        already_remembered = any(box_iou(cand, r) >= OVERLAP_IOU_THRESHOLD for r in remembered_pixel)
                        if not already_remembered:
                            state.copied_boxes_norm.append(cand_norm)
                return

            ax1, ay1, ax2, ay2 = BTN_APPROVE
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                state.mode = "click_approve"
                return
            rx1, ry1, rx2, ry2 = BTN_REJECT
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                state.mode = "click_reject"
                return
            px1, py1, px2, py2 = BTN_PREV
            if px1 <= x <= px2 and py1 <= y <= py2:
                state.mode = "click_prev"
                return
            nx1, ny1, nx2, ny2 = BTN_NEXT
            if nx1 <= x <= nx2 and ny1 <= y <= ny2:
                state.mode = "click_next"
                return
            dx1, dy1, dx2, dy2 = BTN_DELETE
            if dx1 <= x <= dx2 and dy1 <= y <= dy2:
                if state.selected >= 0:
                    del state.boxes[state.selected]
                    state.selected = -1
                return
            mx1, my1, mx2, my2 = BTN_RESET_MEMORY
            if mx1 <= x <= mx2 and my1 <= y <= my2:
                state.mode = "click_reset_memory"
                return
            tx1, ty1, tx2, ty2 = BTN_RERUN_MODEL
            if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                state.mode = "click_rerun_model"
                return
            for cid, (cx1, cy1, cx2, cy2) in BTN_CLASSES.items():
                if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                    state.current_class = cid
                    if state.selected >= 0:
                        state.boxes[state.selected][0] = cid
                    return

            ox, oy = view.to_orig(x, y)

            if state.selected >= 0:
                handle = corner_handle_hit(view, state.boxes[state.selected], x, y)
                if handle:
                    state.mode = "resize"
                    state.resize_handle = handle
                    return

            hit = box_at_point(state.boxes, ox, oy)
            if hit >= 0:
                state.selected = hit
                state.mode = "move"
                _, bx1, by1, bx2, by2 = state.boxes[hit]
                state.drag_start_orig = (ox, oy)
                state.box_orig_snapshot = (bx1, by1, bx2, by2)
                return

            state.selected = -1
            state.mode = "draw"
            state.drag_start_orig = (ox, oy)

        elif event == cv2.EVENT_RBUTTONDOWN:
            bring_window_to_front()
            state.mode = "pan"
            state.drag_start_canvas = (x, y)

        elif event == cv2.EVENT_MBUTTONDOWN:
            bring_window_to_front()
            state.mode = "pan"
            state.drag_start_canvas = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.mode == "pan" and state.drag_start_canvas is not None:
                lx, ly = state.drag_start_canvas
                view.pan(x - lx, y - ly)
                state.drag_start_canvas = (x, y)

            elif state.mode == "move" and state.drag_start_orig is not None:
                ox, oy = view.to_orig(x, y)
                dx = ox - state.drag_start_orig[0]
                dy = oy - state.drag_start_orig[1]
                bx1, by1, bx2, by2 = state.box_orig_snapshot
                bw, bh = bx2 - bx1, by2 - by1
                nx1 = min(max(0, bx1 + dx), state.img_w - bw)
                ny1 = min(max(0, by1 + dy), state.img_h - bh)
                cid = state.boxes[state.selected][0]
                state.boxes[state.selected] = [cid, nx1, ny1, nx1 + bw, ny1 + bh]

            elif state.mode == "resize" and state.selected >= 0:
                ox, oy = view.to_orig(x, y)
                cid, bx1, by1, bx2, by2 = state.boxes[state.selected]
                ox = min(max(0, ox), state.img_w)
                oy = min(max(0, oy), state.img_h)
                if state.resize_handle == "tl":
                    bx1, by1 = ox, oy
                elif state.resize_handle == "tr":
                    bx2, by1 = ox, oy
                elif state.resize_handle == "bl":
                    bx1, by2 = ox, oy
                elif state.resize_handle == "br":
                    bx2, by2 = ox, oy
                state.boxes[state.selected] = [cid, min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)]

        elif event == cv2.EVENT_LBUTTONUP:
            if state.mode == "draw" and state.drag_start_orig is not None:
                ox, oy = view.to_orig(x, y)
                x1o, y1o = state.drag_start_orig
                bx1, bx2 = sorted([x1o, ox])
                by1, by2 = sorted([y1o, oy])
                if (bx2 - bx1) >= MIN_BOX_PX and (by2 - by1) >= MIN_BOX_PX:
                    bx1 = min(max(0, bx1), state.img_w)
                    bx2 = min(max(0, bx2), state.img_w)
                    by1 = min(max(0, by1), state.img_h)
                    by2 = min(max(0, by2), state.img_h)
                    state.boxes.append([state.current_class, bx1, by1, bx2, by2])
                    state.selected = len(state.boxes) - 1
                else:
                    hit = box_at_point(state.boxes, ox, oy)
                    state.selected = hit
            state.mode = None
            state.drag_start_orig = None
            state.drag_start_canvas = None

        elif event == cv2.EVENT_RBUTTONUP:
            state.mode = None
            state.drag_start_canvas = None

        elif event == cv2.EVENT_MBUTTONUP:
            state.mode = None
            state.drag_start_canvas = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            bring_window_to_front()
            factor = 1.15 if flags > 0 else 1 / 1.15
            view.zoom(x, y, factor)

    return callback


def undo_decision(idx, decided_moves, images_dir, labels_dir):
    """Chuyển ảnh + label từ thư mục Approve/Reject về lại thư mục nguồn (hoàn tác)."""
    info = decided_moves.pop(idx, None)
    if info is None:
        return False
    filename, dest_root = info["filename"], info["dest_root"]
    name, _ = os.path.splitext(filename)

    src_img = os.path.join(dest_root, "images", filename)
    src_label = os.path.join(dest_root, "labels", name + ".txt")
    dst_img = os.path.join(images_dir, filename)
    dst_label = os.path.join(labels_dir, name + ".txt")

    if os.path.exists(src_img) and os.path.abspath(src_img) != os.path.abspath(dst_img):
        shutil.move(src_img, dst_img)
    if os.path.exists(src_label) and os.path.abspath(src_label) != os.path.abspath(dst_label):
        shutil.move(src_label, dst_label)
    return True


def ensure_dirs():
    for root in (APPROVE_DIR, REJECT_DIR):
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, "labels"), exist_ok=True)


# ==================== PHÍM TẮT TOÀN CỤC (không phụ thuộc focus cửa sổ) ====================
pending_action = {"value": None}
c_hold_state = {"held": False}      # True khi đang GIỮ phím C - dùng để hiện box mờ + chọn từng box
c_release_trigger = {"value": False}  # bật lên True mỗi khi THẢ phím C - main loop sẽ đọc cờ này để
                                       # tự động dán các box ĐÃ GHI NHỚ vào ảnh hiện tại (nếu có)


def _set_action(name):
    def handler(e):
        pending_action["value"] = name
    return handler


def _set_c_held(value):
    def handler(e):
        c_hold_state["held"] = value
        if not value:
            c_release_trigger["value"] = True   # vừa thả C ra -> báo cho main loop biết để paste nếu có bộ nhớ
    return handler


def register_global_hotkeys():
    if not HAS_KEYBOARD:
        return
    try:
        keyboard.on_press_key("a", _set_action("approve"))
        keyboard.on_press_key("r", _set_action("reject"))
        keyboard.on_press_key("space", _set_action("next"))
        keyboard.on_press_key("right", _set_action("next"))
        keyboard.on_press_key("b", _set_action("prev"))
        keyboard.on_press_key("left", _set_action("prev"))
        keyboard.on_press_key("delete", _set_action("delete"))
        keyboard.on_press_key("x", _set_action("delete"))
        keyboard.on_press_key("c", _set_c_held(True))
        keyboard.on_release_key("c", _set_c_held(False))
        keyboard.on_press_key("v", _set_action("reset_memory"))
        keyboard.on_press_key("t", _set_action("rerun_model"))
        keyboard.on_press_key("q", _set_action("quit"))
        keyboard.on_press_key("esc", _set_action("quit"))
        for d in "0123456789":
            keyboard.on_press_key(d, _set_action(f"class_{d}"))
    except Exception as e:
        print(f"[Hotkeys] Không thể đăng ký global hotkeys: {e}")
# ============================================================================================


def move_decision(filename, images_dir, labels_dir, dest_root, boxes, img_w, img_h):
    name, _ = os.path.splitext(filename)
    dst_img = os.path.join(dest_root, "images", filename)
    dst_label = os.path.join(dest_root, "labels", name + ".txt")

    save_labels(dst_label, boxes, img_w, img_h)  # ghi label theo box HIỆN TẠI (đã chỉnh sửa nếu có)

    src_img = os.path.join(images_dir, filename)
    if os.path.exists(src_img) and os.path.abspath(src_img) != os.path.abspath(dst_img):
        shutil.move(src_img, dst_img)

    src_label = os.path.join(labels_dir, name + ".txt")
    if os.path.exists(src_label) and os.path.abspath(src_label) != os.path.abspath(dst_label):
        os.remove(src_label)  # đã ghi bản mới vào dest rồi, chỉ xoá bản cũ ở nguồn nếu khác thư mục đích


def main():
    images_dir = os.path.join(SOURCE_DIR, "images")
    labels_dir = os.path.join(SOURCE_DIR, "labels")
    if not os.path.isdir(images_dir):
        print(f"❌ Không tìm thấy '{images_dir}'. SOURCE_DIR phải có cấu trúc images/ + labels/.")
        return

    ensure_dirs()
    register_global_hotkeys()

    files = find_images(images_dir)
    if not files:
        print("❌ Không còn ảnh nào để duyệt trong thư mục nguồn.")
        return

    print(f"🔍 {len(files)} ảnh cần duyệt.")
    print("Chuột: kéo trái=vẽ box | click box=chọn | kéo box đã chọn=di chuyển | kéo góc=resize")
    print("Chuột phải kéo=pan | lăn chuột=zoom")
    print("Phím: 0-9=đổi class | Delete/X=xoá box đã chọn | A=Approve | R=Reject")
    print("SPACE/-> = ảnh tiếp (không quyết định) | B/<- = ảnh trước | Q/ESC = thoát")
    print("Phím C: bấm-thả nhanh = tự dán các box ĐÃ GHI NHỚ vào ảnh này | GIỮ C = hiện box mờ để click chọn/ghi nhớ mới")
    print("Phím V (hoặc nút RESET GHI NHỚ) = xoá sạch bộ nhớ box đã ghi nhớ")
    print(f"Phím T (hoặc nút CHAY LAI MODEL) = load model + chạy lại trên ảnh hiện tại với ngưỡng {MODEL_RERUN_CONF}\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    state = ReviewerState()
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state))

    idx = 0
    decided_moves = {}     # idx -> {"filename":..., "dest_root":...} - để hoàn tác khi bấm quay lại

    while 0 <= idx < len(files):
        filename = files[idx]
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            # Ảnh không có ở nguồn - có thể do đã approve/reject trước đó trong phiên này.
            # Nếu vậy, TỰ ĐỘNG HOÀN TÁC (chuyển ảnh về lại nguồn) để xem/sửa lại được.
            if undo_decision(idx, decided_moves, images_dir, labels_dir):
                pass  # ảnh đã được chuyển về, vòng lặp sẽ đọc lại ở dưới
            else:
                idx += 1
                continue

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        state.img = img
        state.img_w, state.img_h = w, h
        state.view = View(w, h)
        state.boxes = load_labels(os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt"), w, h)
        state.selected = -1
        state.mode = None
        state.file_info = f"[{idx+1}/{len(files)}] {filename}"
        state.ghost_source_norm = compute_ghost_source_norm(state)  # chốt nguồn ghost cho riêng ảnh này

        decision = None  # "approve" | "reject" | "next" | "prev" | "quit"
        focus_retry = 8   # ép focus lặp lại vài vòng đầu, tránh trường hợp cửa sổ chưa kịp render xong

        while decision is None:
            canvas = render(state)
            cv2.imshow(WINDOW_NAME, canvas)
            if focus_retry > 0:
                bring_window_to_front()
                focus_retry -= 1
            key = cv2.waitKey(20) & 0xFF

            if not HAS_KEYBOARD:
                is_c_down = bool(user32.GetAsyncKeyState(ord('C')) & 0x8000)
                if is_c_down and not c_hold_state["held"]:
                    c_hold_state["held"] = True
                elif not is_c_down and c_hold_state["held"]:
                    c_hold_state["held"] = False
                    c_release_trigger["value"] = True

            # THẢ phím C ra -> nếu đã có box được ghi nhớ từ trước, tự động dán hết vào ảnh này luôn
            # (không cần giữ + click lại - chỉ cần bấm-thả C là paste, giống như đã "copy" sẵn từ trước)
            if c_release_trigger["value"]:
                c_release_trigger["value"] = False
                paste_all_remembered(state, w, h)
                continue

            # Ưu tiên xử lý phím tắt TOÀN CỤC (bắt được kể cả khi cửa sổ ảnh không có focus)
            action = pending_action["value"]
            if action:
                pending_action["value"] = None
                if action == "approve":
                    decision = "approve"
                elif action == "reject":
                    decision = "reject"
                elif action == "next":
                    decision = "next"
                elif action == "prev":
                    decision = "prev"
                elif action == "quit":
                    decision = "quit"
                elif action == "delete" and state.selected >= 0:
                    del state.boxes[state.selected]
                    state.selected = -1
                elif action == "reset_memory":
                    reset_copied_memory(state)
                elif action == "rerun_model":
                    # Hiện chữ báo đang chạy TRƯỚC khi gọi model (vì predict() sẽ block vài giây)
                    loading_canvas = canvas.copy()
                    msg = f"DANG CHAY LAI MODEL (conf>={MODEL_RERUN_CONF})... CHO VAI GIAY"
                    cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                    cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
                    cv2.imshow(WINDOW_NAME, loading_canvas)
                    cv2.waitKey(1)
                    rerun_model_on_image(state, img_path)
                elif action.startswith("class_"):
                    cid = int(action.split("_")[1])
                    state.current_class = cid
                    if state.selected >= 0:
                        state.boxes[state.selected][0] = cid
                continue  # đã xử lý xong action này, vẽ lại canvas ngay rồi mới đọc phím/chuột tiếp

            if state.mode == "click_approve":
                decision = "approve"
                state.mode = None
            elif state.mode == "click_reject":
                decision = "reject"
                state.mode = None
            elif state.mode == "click_next":
                decision = "next"
                state.mode = None
            elif state.mode == "click_prev":
                decision = "prev"
                state.mode = None
            elif state.mode == "click_reset_memory":
                reset_copied_memory(state)
                state.mode = None
            elif state.mode == "click_rerun_model":
                loading_canvas = canvas.copy()
                msg = f"DANG CHAY LAI MODEL (conf>={MODEL_RERUN_CONF})... CHO VAI GIAY"
                cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, loading_canvas)
                cv2.waitKey(1)
                rerun_model_on_image(state, img_path)
                state.mode = None
            elif key in (ord("a"), ord("A")):
                decision = "approve"
            elif key in (ord("r"), ord("R")):
                decision = "reject"
            elif key in (32, 83, 3):  # SPACE hoặc mũi tên phải
                decision = "next"
            elif key in (ord("b"), ord("B"), 81, 2):  # B hoặc mũi tên trái
                decision = "prev"
            elif key in (ord("q"), ord("Q"), 27):
                decision = "quit"
            elif key in (ord("x"), ord("X"), 8) and state.selected >= 0:
                del state.boxes[state.selected]
                state.selected = -1
            elif key in (ord("v"), ord("V")):
                reset_copied_memory(state)
            elif key in (ord("t"), ord("T")):
                loading_canvas = canvas.copy()
                msg = f"DANG CHAY LAI MODEL (conf>={MODEL_RERUN_CONF})... CHO VAI GIAY"
                cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(loading_canvas, msg, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, loading_canvas)
                cv2.waitKey(1)
                rerun_model_on_image(state, img_path)
            elif ord("0") <= key <= ord("9"):
                cid = key - ord("0")
                state.current_class = cid
                if state.selected >= 0:
                    state.boxes[state.selected][0] = cid

        if decision == "quit":
            break

        # Ghi nhớ box của ảnh này (nếu có) làm "box gần nhất" cho ảnh tiếp theo dùng khi giữ C
        # (áp dụng cho mọi hướng - kể cả khi bấm B quay lại - vì đây luôn là ảnh gần nhất vừa xem)
        if state.boxes:
            state.last_boxes_norm = boxes_to_normalized(state.boxes, w, h)

        if decision == "approve":
            move_decision(filename, images_dir, labels_dir, APPROVE_DIR, state.boxes, w, h)
            decided_moves[idx] = {"filename": filename, "dest_root": APPROVE_DIR}
            idx += 1
        elif decision == "reject":
            move_decision(filename, images_dir, labels_dir, REJECT_DIR, state.boxes, w, h)
            decided_moves[idx] = {"filename": filename, "dest_root": REJECT_DIR}
            idx += 1
        elif decision == "next":
            idx += 1
        elif decision == "prev":
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print("\n🎉 Kết thúc phiên duyệt.")


if __name__ == "__main__":
    main()