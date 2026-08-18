"""
CALIBRATE HOMOGRAPHY (ảnh -> mặt bàn) CHO TỪNG CAMERA
=====================================================
Chạy 1 lần cho MỖI camera. Ý tưởng:
  - Đặt các ĐIỂM MỐC lên mặt bàn mà bạn BIẾT toạ độ thật của chúng
    (đơn vị cm, tự chọn gốc toạ độ). Ví dụ dán 4 chấm tạo hình chữ nhật
    30x20 cm: (0,0), (30,0), (30,20), (0,20).
  - Trên ảnh camera, BẤM CHUỘT vào đúng 4 điểm đó theo THỨ TỰ.
  - Chương trình tính ma trận H (ảnh->bàn) và lưu vào homographies.json.

>>> Càng nhiều điểm (6-8) và trải rộng khắp view -> H càng chính xác.
>>> Bấm theo ĐÚNG thứ tự bạn khai trong TABLE_POINTS.

PHÍM:
  chuột trái : thêm 1 điểm ảnh (khớp lần lượt với TABLE_POINTS)
  u          : undo điểm cuối
  s          : tính H + lưu (khi đã đủ điểm)
  q/ESC      : bỏ qua camera này
"""

import json
import os
import numpy as np
import cv2

# ==== KHAI BÁO ĐIỂM MỐC THẬT TRÊN MẶT BÀN (cm) - CHỈNH THEO BÀN CỦA BẠN ====
# Thứ tự này phải khớp thứ tự bạn bấm chuột trên ảnh.
# Mốc của bạn: hình chữ nhật 25cm (ngang) x 20cm (dọc).
# Nếu thực tế NGANG=20, DỌC=25 thì đổi số cho đúng.
# Bấm theo thứ tự: trên-trái -> trên-phải -> dưới-phải -> dưới-trái.
TABLE_POINTS = [
    (0.0, 0.0),     # 1: trên-trái  (gốc toạ độ)
    (25.0, 0.0),    # 2: trên-phải
    (25.0, 20.0),   # 3: dưới-phải
    (0.0, 20.0),    # 4: dưới-trái
]

OUT_JSON = "homographies.json"


def calibrate_one(cam_name, get_frame):
    """
    cam_name : tên camera (khoá trong json).
    get_frame: callable trả về 1 ảnh BGR để calibrate.
    Trả H (3x3) hoặc None nếu bỏ qua.

    ĐIỀU KHIỂN:
      - LĂN chuột lên/xuống : ZOOM to/nhỏ quanh con trỏ (bấm chính xác).
      - GIỮ chuột PHẢI + kéo: PAN (di chuyển vùng nhìn khi đã zoom).
      - Chuột TRÁI           : đặt 1 điểm mốc (đúng toạ độ ảnh gốc).
      - u = undo | r = reset zoom | s = lưu | q/ESC = bỏ qua.
    """
    img_points = []

    win = (f"CALIBRATE [{cam_name}] - lan chuot=ZOOM, "
           f"chuot phai=PAN, trai=cham diem | u/r/s/q")
    frame = get_frame()
    if frame is None:
        print(f"[{cam_name}] không lấy được ảnh.")
        return None
    base = frame.copy()
    Himg, Wimg = base.shape[:2]

    # Cửa sổ hiển thị cố định
    VIEW_W, VIEW_H = 1280, 720

    # Trạng thái view: zoom + góc trên-trái (theo toạ độ ảnh gốc)
    view = {
        "zoom": min(VIEW_W / Wimg, VIEW_H / Himg),  # fit ban đầu
        "ox": 0.0, "oy": 0.0,     # toạ độ ảnh gốc tại góc trên-trái khung nhìn
        "panning": False, "pan_start": (0, 0), "o_start": (0, 0),
        "mouse": (0, 0),
    }
    view["ox"] = 0.0
    view["oy"] = 0.0

    def clamp_offsets():
        # không cho pan ra ngoài ảnh
        vis_w = VIEW_W / view["zoom"]
        vis_h = VIEW_H / view["zoom"]
        view["ox"] = max(0.0, min(view["ox"], max(0.0, Wimg - vis_w)))
        view["oy"] = max(0.0, min(view["oy"], max(0.0, Himg - vis_h)))

    def screen_to_image(sx, sy):
        """toạ độ trên cửa sổ -> toạ độ ảnh gốc."""
        return view["ox"] + sx / view["zoom"], view["oy"] + sy / view["zoom"]

    def image_to_screen(ix, iy):
        return (ix - view["ox"]) * view["zoom"], (iy - view["oy"]) * view["zoom"]

    def render():
        clamp_offsets()
        vis_w = int(VIEW_W / view["zoom"])
        vis_h = int(VIEW_H / view["zoom"])
        x0, y0 = int(view["ox"]), int(view["oy"])
        x1, y1 = min(Wimg, x0 + vis_w), min(Himg, y0 + vis_h)
        crop = base[y0:y1, x0:x1]
        disp = cv2.resize(crop, (VIEW_W, VIEW_H), interpolation=cv2.INTER_NEAREST)

        # vẽ điểm đã đặt
        for i, (ix, iy) in enumerate(img_points):
            sx, sy = image_to_screen(ix, iy)
            if -20 <= sx <= VIEW_W + 20 and -20 <= sy <= VIEW_H + 20:
                cv2.drawMarker(disp, (int(sx), int(sy)), (0, 255, 0),
                               cv2.MARKER_CROSS, 22, 2)
                cv2.circle(disp, (int(sx), int(sy)), 3, (0, 0, 255), -1)
                cv2.putText(disp, str(i + 1), (int(sx) + 10, int(sy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # con trỏ + toạ độ ảnh gốc (giúp bấm chuẩn)
        mx, my = view["mouse"]
        ix, iy = screen_to_image(mx, my)
        cv2.line(disp, (mx, 0), (mx, VIEW_H), (80, 80, 80), 1)
        cv2.line(disp, (0, my), (VIEW_W, my), (80, 80, 80), 1)

        nxt = len(img_points)
        if nxt < len(TABLE_POINTS):
            tp = TABLE_POINTS[nxt]
            msg = f"Cham diem {nxt+1}: ban {tp} cm | zoom x{view['zoom']:.1f}"
            col = (0, 255, 255)
        else:
            msg = f"Du diem. 's'=luu | zoom x{view['zoom']:.1f}"
            col = (0, 255, 0)
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(disp, f"img({ix:.0f},{iy:.0f})", (10, VIEW_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow(win, disp)

    def on_mouse(ev, x, y, flags, param):
        view["mouse"] = (x, y)

        if ev == cv2.EVENT_MOUSEWHEEL:
            # zoom quanh con trỏ
            ix, iy = screen_to_image(x, y)
            up = flags > 0
            factor = 1.25 if up else 1 / 1.25
            new_zoom = view["zoom"] * factor
            new_zoom = max(min(VIEW_W / Wimg, VIEW_H / Himg) * 0.9,
                           min(new_zoom, 40.0))
            view["zoom"] = new_zoom
            # giữ điểm dưới con trỏ đứng yên
            view["ox"] = ix - x / view["zoom"]
            view["oy"] = iy - y / view["zoom"]
            render()

        elif ev == cv2.EVENT_RBUTTONDOWN:
            view["panning"] = True
            view["pan_start"] = (x, y)
            view["o_start"] = (view["ox"], view["oy"])

        elif ev == cv2.EVENT_MOUSEMOVE:
            if view["panning"]:
                dx = (x - view["pan_start"][0]) / view["zoom"]
                dy = (y - view["pan_start"][1]) / view["zoom"]
                view["ox"] = view["o_start"][0] - dx
                view["oy"] = view["o_start"][1] - dy
            render()

        elif ev == cv2.EVENT_RBUTTONUP:
            view["panning"] = False

        elif ev == cv2.EVENT_LBUTTONDOWN:
            if len(img_points) < len(TABLE_POINTS):
                ix, iy = screen_to_image(x, y)
                img_points.append((ix, iy))
                render()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    render()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            cv2.destroyWindow(win)
            return None
        if k == ord('u') and img_points:
            img_points.pop(); render()
        if k == ord('r'):   # reset zoom về fit
            view["zoom"] = min(VIEW_W / Wimg, VIEW_H / Himg)
            view["ox"] = 0.0; view["oy"] = 0.0
            render()
        if k == ord('s') and len(img_points) >= 4:
            src = np.array(img_points, np.float32)
            dst = np.array(TABLE_POINTS[:len(img_points)], np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
            cv2.destroyWindow(win)
            return H


def _fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    _fit.scale = s
    if s < 1.0:
        return cv2.resize(img, (int(w * s), int(h * s)))
    return img
_fit.scale = 1.0


def save_homographies(hdict, path=OUT_JSON):
    data = {name: H.tolist() for name, H in hdict.items() if H is not None}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Đã lưu {len(data)} homography -> {path}")


def load_homographies(path=OUT_JSON):
    with open(path) as f:
        data = json.load(f)
    return {name: np.array(v, np.float64) for name, v in data.items()}


if __name__ == "__main__":
    # Ví dụ calibrate với ảnh tĩnh test (thay bằng camera thật khi chạy).
    from cameras import FakeCamera
    import numpy as np

    def dummy():
        img = np.full((720, 1280, 3), 60, np.uint8)
        for (x, y) in [(300, 200), (900, 220), (920, 560), (280, 540)]:
            cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
        return img

    cam = FakeCamera("demo", dummy)
    H = calibrate_one("demo", cam.read)
    if H is not None:
        save_homographies({"demo": H})