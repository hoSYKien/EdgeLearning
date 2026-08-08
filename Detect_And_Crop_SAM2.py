"""
KIẾN TRÚC: Detect + Crop sản phẩm bằng SAM2 (Segment Anything Model 2).
KHÔNG cần train, KHÔNG cần chỉnh threshold màu tay.

2 NHÓM CHẾ ĐỘ (đổi ở DETECTION_MODE bên dưới):
- "template_guided" : (MỚI - KHUYẾN NGHỊ) bạn khoanh tay 1 lần vùng sản
                  phẩm trên ảnh đầu tiên của thư mục, script dùng đối sánh
                  đặc trưng ảnh (ORB) để tự tìm vị trí vật tương tự ở các
                  ảnh còn lại, rồi đưa vị trí đó vào SAM2 để cắt chính xác.
                  Không cần giả định vị trí, chịu được vật xoay/lệch góc.
- "everything"  : quét lưới điểm khắp ảnh, KHÔNG cần biết trước vật nằm ở
                  đâu, nhưng có thể vẫn bắt sai/sót nếu vật khó phân biệt
                  với nền.
- "box_point"/"box"/"point" : nhanh hơn nữa nhưng GIẢ ĐỊNH vật nằm gần giữa
                  khung hình - sai nếu vật lệch hẳn khỏi giữa ảnh.

BỘ LỌC CHỌN MASK (áp dụng cho mọi chế độ, không phụ thuộc vị trí vật):
- Lọc theo diện tích (MIN/MAX_MASK_AREA_RATIO).
- Lọc "chạm viền ảnh" (BORDER_TOUCH_MAX_RATIO): nếu camera chụp có margin
  quanh sản phẩm thì NỀN luôn chạm mép ảnh, còn SẢN PHẨM hầu như không chạm
  - nhờ đó tự động loại được mask nền dù sản phẩm nằm ở bất kỳ đâu trong
  khung, không cần giả định vị trí trung tâm.

Ý TƯỞNG CHUNG:
- SAM2 trả về mask -> lọc theo diện tích -> chọn mask phù hợp nhất -> suy ra
  bounding box -> crop + đệm padding.

YÊU CẦU CÀI ĐẶT (chạy 1 lần):
    pip install ultralytics opencv-python numpy
Lần chạy đầu tiên, ultralytics sẽ TỰ ĐỘNG TẢI weight "sam2.1_b.pt" (khoảng
150-300MB tùy bản) về thư mục làm việc - cần internet lần đầu, sau đó dùng
offline được.

CÁC BẢN SAM2 (đổi ở SAM_MODEL_NAME nếu muốn):
    sam2.1_t.pt  -> nhỏ nhất, nhanh nhất, độ chính xác thấp hơn
    sam2.1_s.pt  -> nhỏ, cân bằng tốc độ
    sam2.1_b.pt  -> (khuyến nghị) cân bằng tốc độ/độ chính xác, hợp GPU thường
    sam2.1_l.pt  -> lớn nhất, chính xác nhất, chậm nhất - hợp khi có GPU khỏe

Cách chạy:
    python detect_and_crop_sam2.py
"""

import os

import cv2
import numpy as np

# ==========================================================================
# CONFIG
# ==========================================================================

MODE = "preview"   # "preview" | "batch"

# --- Model ---
SAM_MODEL_NAME = "sam2.1_b.pt"
DEVICE = "cuda"     # "cuda" nếu có GPU, đổi thành "cpu" nếu không có / lỗi

# --- Chế độ detect ---
# "template_guided" : (MỚI - KHUYẾN NGHỊ nếu everything mode vẫn bắt kém)
#                Bạn khoanh tay 1 lần vùng sản phẩm trên ẢNH ĐẦU TIÊN của
#                thư mục (1 cửa sổ hiện lên, kéo chuột chọn vùng, ENTER xác
#                nhận). Script dùng vùng đó làm MẪU THAM CHIẾU, rồi với mỗi
#                ảnh sau, tự tìm vị trí vật tương tự bằng đối sánh đặc trưng
#                ảnh (ORB feature matching - chịu được xoay/lệch vị trí),
#                sau đó đưa vị trí tìm được vào SAM2 để cắt chính xác. Nếu
#                không tìm thấy vị trí đáng tin cậy ở 1 ảnh nào đó, tự động
#                lùi về TEMPLATE_FALLBACK_MODE cho ảnh đó.
# "everything" : (TỔNG QUÁT NHẤT nhưng có thể vẫn bắt sai/sót nếu vật khó
#                phân biệt với nền) quét lưới điểm khắp ảnh, KHÔNG cần biết
#                trước vật nằm ở đâu.
# "box_point"  : (NHANH, nhưng GIẢ ĐỊNH vật nằm gần giữa khung) kết hợp khung
#                + 1 điểm foreground ở giữa.
# "box"        : chỉ dùng khung, không có điểm định hướng - dễ nhận nhầm nền.
# "point"      : chỉ dùng 1 điểm giữa ảnh - dễ lệch vào chi tiết nhỏ.
DETECTION_MODE = "template_guided"
# "template_guided" | "everything" | "box_point" | "box" | "point"

# Chỉ dùng cho DETECTION_MODE = "template_guided":
TEMPLATE_FALLBACK_MODE = "everything"   # dùng khi không định vị được bằng template
TEMPLATE_MIN_GOOD_MATCHES = 8   # số điểm đặc trưng khớp tối thiểu để tin là tìm đúng vị trí
TEMPLATE_BOX_PADDING_RATIO = 0.15   # nới thêm khung quanh vùng định vị được (15%) trước khi đưa vào SAM2

# Chỉ dùng cho DETECTION_MODE = "everything": số điểm quét mỗi cạnh (tổng số
# điểm = POINTS_STRIDE^2). Mặc định gốc của SAM2 là 32 (1024 điểm - khá
# chậm). Giảm xuống để nhanh hơn, đổi lại có thể bỏ sót vật rất nhỏ/mảnh.
#   8  -> 64 điểm   (rất nhanh, hợp vật to chiếm nhiều diện tích khung)
#   16 -> 256 điểm  (khuyến nghị - cân bằng tốc độ/độ phủ)
#   32 -> 1024 điểm (mặc định gốc SAM2, chậm nhất, phủ dày nhất)
POINTS_STRIDE = 16
CROP_N_LAYERS = 0   # >0 sẽ quét thêm ở các vùng crop nhỏ hơn - CHẬM HƠN NHIỀU, chỉ bật nếu vật rất nhỏ

# Chỉ dùng cho DETECTION_MODE = "box"/"box_point": tỉ lệ viền bỏ ra mỗi cạnh
# so với kích thước ảnh. 0.05 nghĩa là khung bao phủ 90% ảnh ở giữa.
BOX_MARGIN_RATIO = 0.05

# --- Dùng cho MODE = "preview" ---
# Có thể là ĐƯỜNG DẪN 1 FILE ẢNH, hoặc ĐƯỜNG DẪN 1 THƯ MỤC chứa nhiều ảnh
# (ví dụ: D:\TongHop\RTC Technologi\RD\Tutorial\dataset\train\cans) - nếu là
# thư mục, script sẽ duyệt lần lượt từng ảnh, hiện preview, nhấn phím bất kỳ
# để qua ảnh tiếp theo, nhấn ESC để dừng giữa chừng.
PREVIEW_IMAGE_PATH = r"D:\TongHop\RTC Technologi\RD\Tutorial\Edge Learning\Goi_Hut_Am\Goi hut am\train\type_1"
PREVIEW_DISPLAY_MAX_WIDTH = 700   # chỉ để hiển thị, không ảnh hưởng crop thật

# --- Dùng cho MODE = "batch" ---
BATCH_INPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset\train"
BATCH_OUTPUT_DIR = r"D:\TongHop\RTC Technologi\RD\detect_color\data2\dataset_cropped\train"

# --- Tham số lọc/chọn mask ---
MIN_MASK_AREA_RATIO = 0.02   # bỏ mask quá nhỏ (< 2% diện tích ảnh) - lọc nhiễu
MAX_MASK_AREA_RATIO = 0.95   # bỏ mask quá lớn (gần như che hết ảnh = nhận nhầm nền)

# Bộ lọc TỔNG QUÁT (không cần biết vật nằm đâu trong khung): nếu camera chụp
# có margin quanh sản phẩm, NỀN sẽ luôn chạm tới mép ảnh, còn SẢN PHẨM hầu
# như không chạm mép. Loại các mask có tỉ lệ pixel-viền-bị-chiếm quá cao -
# đây chính là cách chặn lỗi "chọn nhầm nền" đã gặp trước đó, không phụ
# thuộc việc vật nằm giữa hay lệch góc.
# Giảm xuống nếu ảnh của bạn có sản phẩm chạm/gần mép khung hình.
BORDER_TOUCH_MAX_RATIO = 0.3

# "largest": trong các mask còn lại sau khi lọc, chọn mask diện tích lớn nhất
# "most_central": chọn mask có tâm gần tâm ảnh nhất (chỉ hợp nếu sản phẩm
#                 luôn ở giữa khung - nếu vật có thể lệch góc, dùng "largest")
SELECT_STRATEGY = "largest"   # "largest" | "most_central"
CROP_PADDING = 15

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_model = None  # cache model SAM (dùng cho box/box_point/point mode), chỉ load 1 lần
_everything_predictor = None  # cache predictor cấp thấp (dùng riêng cho everything mode)


def get_model():
    global _model
    if _model is None:
        import torch
        from ultralytics import SAM

        cuda_ok = torch.cuda.is_available()
        print(f"torch.cuda.is_available() = {cuda_ok}")
        if DEVICE == "cuda" and not cuda_ok:
            print("  CẢNH BÁO: DEVICE='cuda' nhưng máy không thấy GPU/CUDA khả dụng "
                  "(có thể bạn cài bản torch CPU-only). Sẽ CHẠY RẤT CHẬM trên CPU.")

        print(f"Đang tải/khởi tạo model {SAM_MODEL_NAME} (lần đầu có thể mất thời gian tải weight + "
              f"load lên GPU, có thể tới vài phút, xin đợi)...")
        _model = SAM(SAM_MODEL_NAME)
        print("Đã khởi tạo model xong.")
    return _model


def get_everything_predictor():
    """
    Khởi tạo trực tiếp đối tượng Predictor cấp thấp (SAM2Predictor cho model
    sam2.x, hoặc Predictor thường cho SAM1/mobile_sam) - CẦN THIẾT để truyền
    được points_stride/crop_n_layers, vì wrapper cấp cao SAM(...) không chấp
    nhận 2 tham số này (chúng thuộc hàm generate() nội bộ, không phải cấu
    hình cfg hợp lệ của lệnh predict cấp cao).
    """
    global _everything_predictor
    if _everything_predictor is None:
        import torch
        from ultralytics.models.sam import Predictor as SAM1Predictor

        is_sam2 = "sam2" in SAM_MODEL_NAME
        if is_sam2:
            from ultralytics.models.sam import SAM2Predictor
            PredictorClass = SAM2Predictor
        else:
            PredictorClass = SAM1Predictor

        cuda_ok = torch.cuda.is_available()
        if DEVICE == "cuda" and not cuda_ok:
            print("  CẢNH BÁO: DEVICE='cuda' nhưng máy không thấy GPU/CUDA khả dụng. "
                  "Sẽ CHẠY RẤT CHẬM trên CPU.")

        print(f"Đang khởi tạo predictor cho {SAM_MODEL_NAME} (everything mode)...")
        overrides = dict(conf=0.25, task="segment", mode="predict", model=SAM_MODEL_NAME, device=DEVICE)
        _everything_predictor = PredictorClass(overrides=overrides)
        try:
            _everything_predictor.get_model()
        except Exception:
            pass  # 1 số bản ultralytics tự load model ở lần gọi đầu, bỏ qua nếu không có hàm này
        print("Đã khởi tạo predictor xong.")
    return _everything_predictor


def run_sam_everything(img_bgr):
    """
    Chạy SAM2 ở chế độ 'segment everything' - quét lưới POINTS_STRIDE^2 điểm
    khắp ảnh để tìm MỌI vật thể, không cần biết trước vị trí. Trả về list
    các mask nhị phân (0/255).
    """
    predictor = get_everything_predictor()
    n_points = POINTS_STRIDE * POINTS_STRIDE
    print(f"  [everything mode] Đang quét lưới {POINTS_STRIDE}x{POINTS_STRIDE} = {n_points} điểm khắp ảnh...")
    results = predictor(source=img_bgr, points_stride=POINTS_STRIDE, crop_n_layers=CROP_N_LAYERS)
    print("  Suy luận xong.")

    masks = []
    r = results[0]
    if r.masks is None:
        return masks

    H, W = img_bgr.shape[:2]
    for m in r.masks.data:  # tensor (N, h, w) giá trị 0/1
        mask_np = m.cpu().numpy().astype(np.uint8) * 255
        if mask_np.shape[:2] != (H, W):
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_np)
    return masks


def run_sam_box(img_bgr, box_xyxy=None):
    """
    Chạy SAM2 với 1 khung (bounding box) prompt. NHANH nhưng có rủi ro: nếu
    khung quá lỏng (phủ gần hết ảnh) và nền có texture đồng nhất/chiếm diện
    tích lớn hơn vật thể, SAM2 có thể NHẬN NHẦM NỀN là vật thể chính. Nên ưu
    tiên dùng 'box_point' thay vì mode này.
    """
    model = get_model()
    H, W = img_bgr.shape[:2]
    if box_xyxy is None:
        mx = int(W * BOX_MARGIN_RATIO)
        my = int(H * BOX_MARGIN_RATIO)
        box_xyxy = [mx, my, W - mx, H - my]

    print(f"  [box mode] Đang prompt SAM2 với khung {box_xyxy}...")
    results = model(img_bgr, bboxes=[box_xyxy], device=DEVICE, verbose=True)
    print("  Suy luận xong.")

    masks = []
    r = results[0]
    if r.masks is None:
        return masks

    for m in r.masks.data:
        mask_np = m.cpu().numpy().astype(np.uint8) * 255
        if mask_np.shape[:2] != (H, W):
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_np)
    return masks


def run_sam_box_point(img_bgr, box_xyxy=None, point_xy=None):
    """
    (KHUYẾN NGHỊ - ỔN ĐỊNH NHẤT) Kết hợp khung (box) + 1 điểm foreground.
    Box giới hạn vùng tìm kiếm, điểm foreground CHỈ RÕ "vật thể chính nằm ở
    đây" - loại bỏ khả năng SAM2 chọn nhầm nền (như vấn đề gặp phải ở
    box-only mode khi nền chiếm diện tích lớn/đồng nhất hơn vật thể).
    """
    model = get_model()
    H, W = img_bgr.shape[:2]
    if box_xyxy is None:
        mx = int(W * BOX_MARGIN_RATIO)
        my = int(H * BOX_MARGIN_RATIO)
        box_xyxy = [mx, my, W - mx, H - my]
    if point_xy is None:
        point_xy = [W // 2, H // 2]

    print(f"  [box_point mode] Đang prompt SAM2 với khung {box_xyxy} + điểm foreground {point_xy}...")
    results = model(img_bgr, bboxes=[box_xyxy], points=[point_xy], labels=[1], device=DEVICE, verbose=True)
    print("  Suy luận xong.")

    masks = []
    r = results[0]
    if r.masks is None:
        return masks

    for m in r.masks.data:
        mask_np = m.cpu().numpy().astype(np.uint8) * 255
        if mask_np.shape[:2] != (H, W):
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_np)
    return masks


def run_sam_point(img_bgr, point_xy=None):
    """
    Chạy SAM2 với 1 điểm prompt duy nhất (mặc định: giữa ảnh). NHANH vì chỉ
    xử lý 1 điểm thay vì quét lưới ~1024 điểm như 'everything mode'.
    LƯU Ý: nếu điểm rơi vào chi tiết nhỏ/lỗ hõm của vật, có thể segment nhầm
    chi tiết đó thay vì cả vật - dùng 'box_point' mode để tránh lỗi này.
    Trả về list các mask ứng viên SAM2 sinh ra cho điểm đó (thường 1-3 mask
    ở các mức độ chi tiết khác nhau).
    """
    model = get_model()
    H, W = img_bgr.shape[:2]
    if point_xy is None:
        point_xy = [W // 2, H // 2]

    print(f"  [point mode] Đang prompt SAM2 tại điểm {point_xy}...")
    results = model(img_bgr, points=[point_xy], labels=[1], device=DEVICE, verbose=True)
    print("  Suy luận xong.")

    masks = []
    r = results[0]
    if r.masks is None:
        return masks

    for m in r.masks.data:
        mask_np = m.cpu().numpy().astype(np.uint8) * 255
        if mask_np.shape[:2] != (H, W):
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_np)
    return masks


def run_sam_detect(img_bgr):
    """Điều phối theo DETECTION_MODE."""
    if DETECTION_MODE == "template_guided":
        return run_sam_template_guided(img_bgr)
    elif DETECTION_MODE == "box_point":
        return run_sam_box_point(img_bgr)
    elif DETECTION_MODE == "box":
        return run_sam_box(img_bgr)
    elif DETECTION_MODE == "point":
        return run_sam_point(img_bgr)
    elif DETECTION_MODE == "everything":
        return run_sam_everything(img_bgr)
    else:
        raise ValueError("DETECTION_MODE phải là 'template_guided', 'box_point', 'box', 'point' hoặc 'everything'")


# ==========================================================================
# TEMPLATE-GUIDED MODE: chọn tay 1 lần trên ảnh đầu tiên, sau đó tự định vị
# vật tương tự ở các ảnh còn lại bằng đối sánh đặc trưng ảnh (ORB).
# ==========================================================================

_orb = cv2.ORB_create(nfeatures=2000)
_ref_kp = None
_ref_des = None
_ref_wh = None   # (width, height) của vùng mẫu đã chọn


def select_reference_roi(image_path):
    """
    Hiện ảnh, để người dùng KÉO CHUỘT chọn vùng sản phẩm mẫu, nhấn ENTER/SPACE
    để xác nhận (hoặc 'c' để hủy). Trả về ảnh đã crop (vùng mẫu) hoặc None
    nếu người dùng hủy.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Lỗi: không đọc được ảnh {image_path}")
        return None

    H, W = img.shape[:2]
    disp = _resize_for_display(img, PREVIEW_DISPLAY_MAX_WIDTH)
    scale = disp.shape[1] / W

    win = "Keo chuot chon vung SAN PHAM MAU - ENTER de xac nhan, C de huy"
    print("\n=== CHỌN VÙNG SẢN PHẨM MẪU ===")
    print("Kéo chuột để khoanh vùng sản phẩm trên ảnh vừa hiện lên.")
    print("Nhấn ENTER hoặc SPACE để xác nhận, nhấn 'c' để hủy chọn.\n")
    r = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, w, h = r
    if w == 0 or h == 0:
        print("  Không có vùng nào được chọn - bỏ qua template.")
        return None

    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
    template_crop = img[y:y + h, x:x + w]
    print(f"  Đã chọn vùng mẫu: x={x}, y={y}, w={w}, h={h}")
    return template_crop


def set_reference_template(template_crop):
    """Tính đặc trưng ORB cho vùng mẫu và lưu lại (dùng chung cho cả folder)."""
    global _ref_kp, _ref_des, _ref_wh
    gray = cv2.cvtColor(template_crop, cv2.COLOR_BGR2GRAY)
    _ref_kp, _ref_des = _orb.detectAndCompute(gray, None)
    _ref_wh = (template_crop.shape[1], template_crop.shape[0])
    n_kp = len(_ref_kp) if _ref_kp else 0
    print(f"  Đã trích {n_kp} điểm đặc trưng từ vùng mẫu.")
    if n_kp < TEMPLATE_MIN_GOOD_MATCHES:
        print("  CẢNH BÁO: vùng mẫu quá ít chi tiết/đặc trưng (bề mặt trơn, ít chữ/hoa văn) "
              "- đối sánh có thể không đáng tin cậy. Nên chọn vùng có nhiều chi tiết hơn.")


def has_reference_template():
    return _ref_des is not None and _ref_kp is not None and len(_ref_kp) > 0


def locate_by_template(img_bgr):
    """
    Tìm vùng giống với template tham chiếu trong ảnh, bằng ORB feature
    matching + homography (chịu được xoay/lệch tỉ lệ tốt hơn template
    matching thông thường). Trả về bbox [x1,y1,x2,y2] hoặc None nếu không
    đủ tin cậy.
    """
    if not has_reference_template():
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp2, des2 = _orb.detectAndCompute(gray, None)
    if des2 is None or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(_ref_des, des2, k=2)
    except cv2.error:
        return None

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < TEMPLATE_MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([_ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask_inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None
    n_inliers = int(mask_inliers.sum()) if mask_inliers is not None else 0
    if n_inliers < TEMPLATE_MIN_GOOD_MATCHES:
        return None

    rw, rh = _ref_wh
    corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(corners, M)
    xs = dst_corners[:, 0, 0]
    ys = dst_corners[:, 0, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    H, W = img_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    pad_x, pad_y = bw * TEMPLATE_BOX_PADDING_RATIO, bh * TEMPLATE_BOX_PADDING_RATIO
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(W, x2 + pad_x), min(H, y2 + pad_y)

    return [int(x1), int(y1), int(x2), int(y2)]


def run_sam_template_guided(img_bgr):
    """
    Dùng vị trí tìm được qua template (nếu có) làm box+point prompt cho
    SAM2. Nếu không định vị được (template chưa được chọn, hoặc đối sánh
    thất bại ở ảnh này), tự động lùi về TEMPLATE_FALLBACK_MODE.
    """
    located_bbox = locate_by_template(img_bgr)

    if located_bbox is not None:
        x1, y1, x2, y2 = located_bbox
        point_xy = [(x1 + x2) // 2, (y1 + y2) // 2]
        print(f"  [template_guided] Định vị được vùng nghi là sản phẩm: {located_bbox}")
        return run_sam_box_point(img_bgr, box_xyxy=located_bbox, point_xy=point_xy)

    print(f"  [template_guided] Không định vị được bằng template (không đủ điểm khớp tin cậy) "
          f"- lùi về chế độ '{TEMPLATE_FALLBACK_MODE}' cho ảnh này.")
    if TEMPLATE_FALLBACK_MODE == "box_point":
        return run_sam_box_point(img_bgr)
    elif TEMPLATE_FALLBACK_MODE == "box":
        return run_sam_box(img_bgr)
    elif TEMPLATE_FALLBACK_MODE == "point":
        return run_sam_point(img_bgr)
    else:
        return run_sam_everything(img_bgr)


def _border_touch_ratio(mask):
    """
    Tính tỉ lệ pixel trên viền ảnh (4 cạnh, dày 2px) thuộc về mask.
    NỀN chụp có margin luôn chạm mép ảnh nhiều -> tỉ lệ cao.
    SẢN PHẨM (được chụp có khoảng trống xung quanh) hầu như không chạm mép
    -> tỉ lệ thấp. Không phụ thuộc vật nằm ở đâu trong khung.
    """
    border = np.zeros_like(mask, dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    border_pixels = border.sum()
    if border_pixels == 0:
        return 0.0
    touched = np.count_nonzero((mask > 0) & border)
    return touched / border_pixels


def select_best_mask(masks, img_shape):
    """
    Lọc theo diện tích + tỉ lệ chạm viền, rồi chọn 1 mask theo SELECT_STRATEGY.
    Trả về (mask, bbox) hoặc (None, None).
    """
    H, W = img_shape[:2]
    img_area = H * W
    cx_img, cy_img = W / 2.0, H / 2.0

    all_candidates = []
    for mask in masks:
        area = int(np.count_nonzero(mask))
        ratio = area / img_area
        if ratio < MIN_MASK_AREA_RATIO or ratio > MAX_MASK_AREA_RATIO:
            continue
        border_ratio = _border_touch_ratio(mask)
        ys, xs = np.where(mask > 0)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()
        cx, cy = x + w / 2.0, y + h / 2.0
        dist_center = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        all_candidates.append({
            "mask": mask, "bbox": (x, y, w, h), "area": area,
            "dist_center": dist_center, "border_ratio": border_ratio,
        })

    if not all_candidates:
        return None, None

    # Ưu tiên loại các mask giống nền (chạm viền nhiều) - tổng quát, không
    # cần biết vật nằm giữa hay lệch góc.
    candidates = [c for c in all_candidates if c["border_ratio"] <= BORDER_TOUCH_MAX_RATIO]
    if not candidates:
        # Không còn candidate nào "sạch viền" - dùng tạm candidate chạm viền
        # ít nhất trong số ban đầu, còn hơn không có gì.
        candidates = sorted(all_candidates, key=lambda c: c["border_ratio"])[:1]

    if SELECT_STRATEGY == "most_central":
        best = min(candidates, key=lambda c: c["dist_center"])
    else:  # "largest"
        best = max(candidates, key=lambda c: c["area"])

    return best["mask"], best["bbox"]


def crop_with_padding(img_bgr, bbox, padding):
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    return img_bgr[y1:y2, x1:x2]


def _resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def show_image(window_name, img):
    cv2.imshow(window_name, _resize_for_display(img, PREVIEW_DISPLAY_MAX_WIDTH))


def draw_all_masks_overlay(img_bgr, masks):
    """Vẽ tất cả các mask SAM2 tìm được lên ảnh (mỗi mask 1 màu ngẫu nhiên) để xem trực quan."""
    overlay = img_bgr.copy()
    rng = np.random.default_rng(42)
    for mask in masks:
        color = rng.integers(60, 255, size=3).tolist()
        colored = np.zeros_like(img_bgr)
        colored[:] = color
        m = mask.astype(bool)
        overlay[m] = cv2.addWeighted(img_bgr, 0.4, colored, 0.6, 0)[m]
    return overlay


_LAST_KEY = None  # lưu phím vừa nhấn trong preview, dùng để phát hiện ESC khi duyệt thư mục


def process_single_image(image_path, output_dir=None, save_debug_steps=False, show_preview=False):
    """Xử lý 1 ảnh bằng SAM2, trả về ảnh đã crop hoặc None nếu thất bại."""
    global _LAST_KEY
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Lỗi: không đọc được ảnh {image_path}")
        return None

    masks = run_sam_detect(img)
    print(f"  SAM2 tìm thấy {len(masks)} vùng/vật thể trong ảnh.")

    best_mask, bbox = select_best_mask(masks, img.shape)
    if bbox is None:
        print(f"  Cảnh báo: không có mask nào đạt tiêu chí diện tích trong {os.path.basename(image_path)}")
        if show_preview and masks:
            show_image("Tat ca mask SAM2 tim duoc", draw_all_masks_overlay(img, masks))
            print("  Nhấn phím bất kỳ để qua ảnh tiếp theo (ESC để dừng)...")
            _LAST_KEY = cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    cropped = crop_with_padding(img, bbox, CROP_PADDING)

    bbox_img = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    if show_preview:
        show_image("1 - Tat ca mask SAM2 tim duoc", draw_all_masks_overlay(img, masks))
        show_image("2 - Mask duoc chon", best_mask)
        show_image("3 - Bounding box", bbox_img)
        show_image("4 - Cropped", cropped)
        print("  Nhấn phím bất kỳ trên cửa sổ ảnh để qua ảnh tiếp theo (ESC để dừng)...")
        _LAST_KEY = cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_debug_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_1_all_masks.png"), draw_all_masks_overlay(img, masks))
        cv2.imwrite(os.path.join(output_dir, f"{base}_2_mask.png"), best_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base}_3_bbox.png"), bbox_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_4_cropped.png"), cropped)

    return cropped


def ensure_reference_template(first_image_path):
    """Nếu DETECTION_MODE='template_guided' và chưa có template, mở cửa sổ cho người dùng chọn."""
    if DETECTION_MODE != "template_guided":
        return
    if has_reference_template():
        return
    template_crop = select_reference_roi(first_image_path)
    if template_crop is not None:
        set_reference_template(template_crop)
    else:
        print(f"  Không có template - các ảnh sẽ dùng luôn chế độ dự phòng '{TEMPLATE_FALLBACK_MODE}'.")


def run_preview():
    """
    Nếu PREVIEW_IMAGE_PATH là 1 file: xử lý + hiện preview file đó.
    Nếu PREVIEW_IMAGE_PATH là 1 thư mục: duyệt lần lượt từng ảnh trong thư
    mục (không đệ quy vào thư mục con), hiện preview từng ảnh, nhấn phím bất
    kỳ để qua ảnh tiếp theo, nhấn ESC để dừng giữa chừng.
    Nếu DETECTION_MODE='template_guided': trước khi duyệt, sẽ hiện ảnh ĐẦU
    TIÊN để bạn khoanh tay vùng sản phẩm mẫu 1 lần duy nhất.
    """
    path = PREVIEW_IMAGE_PATH

    if os.path.isdir(path):
        image_files = sorted([
            f for f in os.listdir(path) if f.lower().endswith(IMG_EXTENSIONS)
        ])
        if not image_files:
            print(f"Không tìm thấy ảnh nào trong thư mục: {path}")
            return

        print(f"Tìm thấy {len(image_files)} ảnh trong thư mục: {path}")

        first_fpath = os.path.join(path, image_files[0])
        ensure_reference_template(first_fpath)

        print("Duyệt lần lượt từng ảnh - nhấn phím bất kỳ để qua ảnh tiếp theo, ESC để dừng.\n")

        for idx, fname in enumerate(image_files, start=1):
            fpath = os.path.join(path, fname)
            print(f"[{idx}/{len(image_files)}] {fname}")
            cropped = process_single_image(fpath, show_preview=True)
            if cropped is not None:
                print("  -> Thành công.")
            else:
                print("  -> Thất bại (xem cảnh báo phía trên).")

            if _LAST_KEY == 27:  # phím ESC
                print("\nĐã nhấn ESC - dừng duyệt.")
                break

        print("\nHoàn tất duyệt thư mục preview. (Chỉ hiển thị, không lưu file nào)")

    elif os.path.isfile(path):
        ensure_reference_template(path)
        print(f"Đang xử lý thử ảnh: {path}")
        cropped = process_single_image(path, show_preview=True)
        if cropped is not None:
            print("Thành công! (Chỉ hiển thị lên màn hình, không lưu file nào)")
        else:
            print("Thất bại - xem cảnh báo phía trên. Thử: giảm MIN_MASK_AREA_RATIO, "
                  "đổi SELECT_STRATEGY, giảm BORDER_TOUCH_MAX_RATIO, hoặc đổi bản model (sam2.1_l.pt để chính xác hơn).")
    else:
        print(f"Không tìm thấy file hoặc thư mục: {path}")


def run_batch():
    if not os.path.isdir(BATCH_INPUT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {BATCH_INPUT_DIR}")

    class_names = sorted([
        d for d in os.listdir(BATCH_INPUT_DIR)
        if os.path.isdir(os.path.join(BATCH_INPUT_DIR, d))
    ])
    print(f"Các class tìm thấy: {class_names}")

    # Nếu dùng template_guided: chọn template 1 lần từ ảnh đầu tiên tìm được
    if DETECTION_MODE == "template_guided" and class_names:
        first_dir = os.path.join(BATCH_INPUT_DIR, class_names[0])
        first_files = [f for f in os.listdir(first_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        if first_files:
            ensure_reference_template(os.path.join(first_dir, first_files[0]))

    total_success, total_fail = 0, 0

    for cls in class_names:
        in_dir = os.path.join(BATCH_INPUT_DIR, cls)
        out_dir = os.path.join(BATCH_OUTPUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(in_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        print(f"\n[{cls}] Đang xử lý {len(image_files)} ảnh...")

        for fname in image_files:
            in_path = os.path.join(in_dir, fname)
            cropped = process_single_image(in_path)

            if cropped is not None:
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, cropped)
                total_success += 1
            else:
                total_fail += 1

    print(f"\n=== Hoàn tất ===")
    print(f"Thành công: {total_success} ảnh")
    print(f"Thất bại (không detect được): {total_fail} ảnh")
    print(f"Kết quả lưu tại: {BATCH_OUTPUT_DIR}")
    if total_fail > 0:
        print("Với các ảnh thất bại, thử giảm MIN_MASK_AREA_RATIO, đổi SELECT_STRATEGY, "
              "hoặc giảm BORDER_TOUCH_MAX_RATIO.")


if __name__ == "__main__":
    if MODE == "preview":
        run_preview()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError("MODE phải là 'preview' hoặc 'batch'")