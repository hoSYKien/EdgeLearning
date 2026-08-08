"""
Tự động nhận diện vật thể + xoay thẳng + cắt ROI kích thước cố định +
ÁP MASK HÌNH HỌC CỐ ĐỊNH để xóa nền thành đen tuyệt đối.

TRIẾT LÝ BẢN NÀY: con hàng là hàng cố định, ảnh output luôn cùng kích
thước và cùng hướng -> KHÔNG dò biên vật thể nữa (dò kiểu gì cũng trượt
vỏ USB Type-C vì mặt trên nó tối gần bằng nền). Chỉ dùng HÌNH HỌC: một
mask cố định gồm
    - thân board: chữ nhật bo góc
    - khối USB: chữ nhật nhô ra phía PHẢI
    - vết khuyết: khoét đen ở cạnh TRÁI
Mọi toạ độ khai báo theo TỈ LỆ (0.0 - 1.0) của khung ảnh crop nên không
phụ thuộc độ phân giải.

THỨ TỰ XỬ LÝ (quan trọng):
    Lượt 1: dò contour + đo ROI (như bản gốc)
    Lượt 2: xoay + cắt ROI, nới padding ĐỐI XỨNG 2 bên (lúc này CHƯA biết
            USB nằm trái hay phải nên không thể áp mask hình học)
    Lượt 3: đồng bộ hướng bằng SIFT -> sau bước này USB LUÔN ở bên phải
    Lượt 4: áp mask hình học cố định lên toàn bộ ảnh crop

CÁCH CHỈNH TOẠ ĐỘ MASK:
    1. Đặt CHE_DO_XEM_THU = True rồi chạy. Ảnh có vẽ viền mask sẽ nằm ở
       thư mục con "_xem_thu", ảnh crop thật KHÔNG bị ghi đè.
    2. Nhìn viền, chỉnh các hằng số trong khối THAM SỐ MASK HÌNH HỌC.
    3. Ưng ý thì đặt lại CHE_DO_XEM_THU = False và chạy lần cuối.

Cách dùng:
    Sửa đường dẫn ở các biến bên dưới rồi chạy:
        python 01_tao_template_v4.py
"""

import os
import cv2
import numpy as np

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_THU_MUC_ANH_MAU = r"D:\TongHop\RTC Technologi\PCB\captured_raw"        # thư mục chứa ảnh cần xử lý
DUONG_DAN_THU_MUC_CROP_MAU = r"D:\TongHop\RTC Technologi\PCB\captured_raw\crop"   # nơi lưu ảnh ROI đã cắt

# Ảnh CHUẨN CỐ ĐỊNH để xác định hướng - DÙNG CHUNG CHO MỌI LẦN CHẠY.
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\captured_crop\capture_20260727_105717_001_roi.png"
# =====================================================================

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")

NGUONG_DIEN_TICH_MIN = 0.6
NGUONG_DIEN_TICH_MAX = 1.6

# Nới khung cắt ĐỐI XỨNG 2 bên (pixel ảnh gốc). Phải đủ rộng để chứa trọn
# đầu USB nhô ra. Cụt USB -> tăng số này lên.
PAD_ROI_NGANG = 300
PAD_ROI_DOC = 60

# =================== THAM SỐ MASK HÌNH HỌC (lượt 4) ===================
AP_MASK_HINH_HOC = True
CHE_DO_XEM_THU = False   # True = chỉ xuất ảnh xem thử, KHÔNG ghi đè ảnh crop

# CÁC SỐ DƯỚI ĐÂY ĐÃ ĐO TRỰC TIẾP TRÊN ẢNH RAW capture_20260730_111713_001.png
# (board bbox 2587x1657 px, padding 300/60 -> khung crop 3187x1777 px).
# Chỉ chỉnh khi đổi ống kính / khoảng cách chụp / loại board.

# --- Thân board: chữ nhật bo góc (tỉ lệ bề rộng W và chiều cao H ảnh crop)
BOARD_X1 = 0.0948    # mép trái board   (302/3187)
BOARD_X2 = 0.9049    # mép phải board   (2884/3187)
BOARD_Y1 = 0.0360    # mép trên board   (64/1777)
BOARD_Y2 = 0.9657    # mép dưới board   (1716/1777)
BOARD_BO_GOC = 0.0141  # bán kính bo góc, tỉ lệ CHIỀU CAO (25/1777)

# --- Khối USB Type-C nhô ra bên PHẢI (chồng lấn vào thân board cho kín mối)
USB_X1 = 0.8472      # (2700/3187) nằm trong thân board
USB_X2 = 0.9576      # mép ngoài cùng vỏ USB (3052/3187)
USB_Y1 = 0.2746      # mép trên vỏ USB (488/1777)
USB_Y2 = 0.6989      # mép dưới vỏ USB (1242/1777)
USB_BO_GOC = 0.0113  # (20/1777)

# --- Vết khuyết (tab bẻ) khoét ĐEN ở cạnh TRÁI. Đặt KHUYET_X2 = 0 để tắt.
KHUYET_X2 = 0.1403   # khoét từ mép trái ảnh tới đây (447/3187)
KHUYET_Y1 = 0.1435   # (255/1777)
KHUYET_Y2 = 0.8694   # (1545/1777)
KHUYET_BO_GOC = 0.0535  # (95/1777)

# Nới/co toàn bộ mask vài pixel: >0 giãn ra (an toàn hơn nếu board xê dịch
# giữa các ảnh, đổi lại lọt vài px nền), <0 co vào (chắc chắn sạch nền,
# đổi lại ăn bớt mép board). 0 = đúng số đo.
NOI_MASK = 0

# Cắt sát: sau khi áp mask, cắt bỏ phần nền thừa quanh mask, chỉ chừa lại
# LE_TRONG pixel viền đen. Mask cố định nên bbox cũng cố định -> mọi ảnh
# vẫn CÙNG KÍCH THƯỚC. Đặt CAT_SAT_MASK = False để giữ nguyên khung rộng.
CAT_SAT_MASK = True
LE_TRONG = 15

# Làm mượt biên mask vài pixel cho đỡ răng cưa (0 = tắt, phải là số lẻ).
LAM_MUOT_BIEN = 5
# =====================================================================

# Xác định trái/phải bằng HÌNH HỌC: phía có USB bị contour khoét lõm rất
# sâu (vì vỏ kim loại không lọt ngưỡng Saturation), phía kia chỉ có vết
# khuyết nông. Tỉ lệ độ lõm sâu/nông phải lớn hơn ngưỡng này thì mới tin;
# dưới ngưỡng sẽ chuyển sang đối chiếu SIFT với ảnh chuẩn.
NGUONG_TI_LE_LOM = 2.0

SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000
MIN_INLIERS_HUONG = 15
# =====================================================================


def tim_contour_tu_dong(img, target_w=1200):
    """Threshold Saturation + lọc thành phần liên thông lớn nhất + lấp lỗ.
    (GIỮ NGUYÊN bản gốc: chỉ cần bắt được thân board để xoay thẳng và
    canh khung, phần USB thiếu đã có mask hình học ở lượt 4 lo.)"""
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    def k(size):  # kernel size quy đổi theo tỉ lệ, làm tròn về số lẻ, tối thiểu 3
        return max(3, round(size * scale) // 2 * 2 + 1)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    blur = cv2.medianBlur(s, k(15))
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k(7), k(7)), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k(41), k(41)), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return (contour.astype(np.float32) / scale).astype(np.int32)  # quy về ảnh gốc


def _do_do_lom(contour_aligned):
    """Đo độ lõm ăn vào từ mép trái và mép phải của contour (đã xoay thẳng).
    Phía lắp USB lõm rất sâu vì contour (ngưỡng Saturation) không bắt được
    vỏ kim loại; phía kia chỉ có vết khuyết tab nông hơn nhiều."""
    x, y, w, h = cv2.boundingRect(contour_aligned)
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [contour_aligned - [x, y]], -1, 255, cv2.FILLED)
    m = m > 0
    trai, phai = [], []
    for r in range(h):
        cot = np.where(m[r])[0]
        if len(cot):
            trai.append(cot.min())
            phai.append(cot.max())
    if not trai:
        return 0.0, 0.0
    trai, phai = np.array(trai), np.array(phai)
    return float(np.median(trai) - trai.min()), float(phai.max() - np.median(phai))


def tinh_m_align_va_roi(contour):
    """Làm thẳng vật thể quanh tâm minAreaRect, đo khung ROI trong khung
    toạ độ đã thẳng, ép quy ước ngang > cao, ÉP USB VỀ BÊN PHẢI bằng hình
    học, rồi nới ĐỐI XỨNG 2 bên.

    Trả về (M_align, roi, ti_le_lom). ti_le_lom là mức tin cậy của việc
    xác định hướng: càng lớn càng chắc, <= NGUONG_TI_LE_LOM thì nên nhờ
    SIFT quyết lại ở lượt 3."""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    M_align = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
    x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    if h > w:
        M_xoay_them = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 90, 1.0)
        M_align_3x3 = np.vstack([M_align, [0, 0, 1]])
        M_xoay_them_3x3 = np.vstack([M_xoay_them, [0, 0, 1]])
        M_align = (M_xoay_them_3x3 @ M_align_3x3)[:2, :]
        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    # --- ép USB về bên PHẢI (phía lõm sâu hơn) ---
    lom_trai, lom_phai = _do_do_lom(contour_aligned.astype(np.int32))
    if lom_trai > lom_phai:
        M_180 = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 180, 1.0)
        M_align = (np.vstack([M_180, [0, 0, 1]]) @ np.vstack([M_align, [0, 0, 1]]))[:2, :]
        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))
    ti_le_lom = max(lom_trai, lom_phai) / (min(lom_trai, lom_phai) + 1e-6)

    x -= PAD_ROI_NGANG
    y -= PAD_ROI_DOC
    w += 2 * PAD_ROI_NGANG
    h += 2 * PAD_ROI_DOC
    return M_align, (x, y, w, h), ti_le_lom


def cat_an_toan(img, x, y, w, h):
    """Cắt (x, y, w, h) kể cả khi tràn mép ảnh; phần tràn lấp ĐEN."""
    H, W = img.shape[:2]
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 > x1 and y2 > y1:
        out[y1 - y:y2 - y, x1 - x:x2 - x] = img[y1:y2, x1:x2]
    return out


def _chu_nhat_bo_goc(mask, x1, y1, x2, y2, r, mau):
    """Tô chữ nhật bo 4 góc (r = bán kính bo, pixel)."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(mask, (x1, y1), (x2, y2), mau, -1)
        return
    cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), mau, -1)
    cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), mau, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(mask, (cx, cy), r, mau, -1)


def tao_mask_hinh_hoc(W: int, H: int):
    """Dựng mask cố định cho khung ảnh crop W x H (USB nằm bên PHẢI)."""
    mask = np.zeros((H, W), np.uint8)

    # 1. Thân board (chữ nhật bo góc)
    _chu_nhat_bo_goc(mask,
                     int(BOARD_X1 * W), int(BOARD_Y1 * H),
                     int(BOARD_X2 * W), int(BOARD_Y2 * H),
                     int(BOARD_BO_GOC * H), 255)

    # 2. Khối USB nhô ra bên phải
    _chu_nhat_bo_goc(mask,
                     int(USB_X1 * W), int(USB_Y1 * H),
                     int(USB_X2 * W), int(USB_Y2 * H),
                     int(USB_BO_GOC * H), 255)

    # 3. Khoét vết khuyết cạnh trái (tô ĐEN đè lên)
    if KHUYET_X2 > 0:
        _chu_nhat_bo_goc(mask,
                         0, int(KHUYET_Y1 * H),
                         int(KHUYET_X2 * W), int(KHUYET_Y2 * H),
                         int(KHUYET_BO_GOC * H), 0)

    if NOI_MASK > 0:
        mask = cv2.dilate(mask, np.ones((2 * NOI_MASK + 1,) * 2, np.uint8))
    elif NOI_MASK < 0:
        mask = cv2.erode(mask, np.ones((2 * -NOI_MASK + 1,) * 2, np.uint8))

    if LAM_MUOT_BIEN and LAM_MUOT_BIEN >= 3:
        mask = cv2.medianBlur(mask, LAM_MUOT_BIEN | 1)
    return mask


def trich_sift_anh(img, target_w=SIFT_TARGET_WIDTH, mask_full=None, nfeatures=SIFT_NFEATURES):
    """SIFT trên ảnh đã thu nhỏ (nhanh hơn), quy đổi toạ độ về ảnh gốc."""
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask_small = None
    if mask_full is not None:
        mask_small = cv2.resize(mask_full, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray, mask_small)
    for k in kp:
        k.pt = (k.pt[0] / scale, k.pt[1] / scale)
    return kp, des


def uoc_luong_M_sift(kp1, des1, kp2, des2, min_inliers=MIN_INLIERS_HUONG):
    """Ước lượng M (ảnh1 -> ảnh2) bằng SIFT + RANSAC. None nếu không đủ tin cậy."""
    if des1 is None or des2 is None:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
    if M is None or inliers is None or inliers.sum() < min_inliers:
        return None, len(good)
    return M, int(inliers.sum())


def luot_1_do_kich_thuoc(thu_muc_anh: str):
    """Quét toàn bộ ảnh, tìm contour + m_align + roi cho từng ảnh."""
    danh_sach_anh = sorted(
        f for f in os.listdir(thu_muc_anh) if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
    )
    if not danh_sach_anh:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong thư mục: {thu_muc_anh}")

    print(f"[1/4] Dò contour + kích thước ROI cho {len(danh_sach_anh)} ảnh...\n")

    du_lieu = {}
    dien_tich_list = []
    for ten_file in danh_sach_anh:
        img = cv2.imread(os.path.join(thu_muc_anh, ten_file))
        if img is None:
            print(f"  [{ten_file}] Không đọc được ảnh, bỏ qua.")
            continue
        contour = tim_contour_tu_dong(img)
        if contour is None:
            print(f"  [{ten_file}] Không tìm thấy vật thể, bỏ qua.")
            continue
        m_align, roi, ti_le_lom = tinh_m_align_va_roi(contour)
        dien_tich = cv2.contourArea(contour)
        du_lieu[ten_file] = {
            "m_align": m_align, "roi": roi, "contour": contour,
            "dien_tich": dien_tich, "img_size": (img.shape[1], img.shape[0]),
            "ti_le_lom": ti_le_lom,
        }
        dien_tich_list.append(dien_tich)
        chac = "chắc" if ti_le_lom > NGUONG_TI_LE_LOM else "NGỜ - để SIFT quyết lại"
        print(f"  [{ten_file}] ROI {roi[2]}x{roi[3]}, diện tích {dien_tich:.0f}, "
              f"hướng {chac} (tỉ lệ lõm {ti_le_lom:.1f})")

    if not du_lieu:
        raise RuntimeError("Không xử lý được ảnh nào.")

    dien_tich_trung_vi = float(np.median(dien_tich_list))
    canonical_w = int(round(np.median([d["roi"][2] for d in du_lieu.values()])))
    canonical_h = int(round(np.median([d["roi"][3] for d in du_lieu.values()])))

    print(f"\nDiện tích trung vị: {dien_tich_trung_vi:.0f}")
    print(f"Kích thước ROI chuẩn chung (trung vị): {canonical_w}x{canonical_h}\n")

    return du_lieu, dien_tich_trung_vi, canonical_w, canonical_h


def luot_2_cat_anh(thu_muc_anh: str, du_lieu: dict, dien_tich_trung_vi: float,
                    canonical_w: int, canonical_h: int, thu_muc_crop: str):
    os.makedirs(thu_muc_crop, exist_ok=True)
    print(f"[2/4] Xoay + cắt ROI cho {len(du_lieu)} ảnh...\n")

    thanh_cong, that_bai, file_crop = [], [], []
    for ten_file, d in du_lieu.items():
        print(f"[{ten_file}]")
        try:
            ti_le = d["dien_tich"] / dien_tich_trung_vi
            if not (NGUONG_DIEN_TICH_MIN < ti_le < NGUONG_DIEN_TICH_MAX):
                raise RuntimeError(
                    f"Diện tích vật thể lệch bất thường so với trung vị cả loạt "
                    f"(tỉ lệ {ti_le:.2f}). Có thể bị che khuất hoặc nhận diện sai."
                )

            img = cv2.imread(os.path.join(thu_muc_anh, ten_file))
            x, y, w, h = d["roi"]
            warped = cv2.warpAffine(img, d["m_align"], d["img_size"])
            crop = cat_an_toan(warped, x, y, w, h)
            if crop.shape[1] != canonical_w or crop.shape[0] != canonical_h:
                crop = cv2.resize(crop, (canonical_w, canonical_h))

            ten_goc, _ = os.path.splitext(ten_file)
            duong_dan_crop = os.path.join(thu_muc_crop, f"{ten_goc}_roi.png")
            cv2.imwrite(duong_dan_crop, crop)

            print(f"  -> Đã lưu: {duong_dan_crop}")
            thanh_cong.append(ten_file)
            file_crop.append(f"{ten_goc}_roi.png")
        except Exception as e:
            print(f"  -> LỖI: {e}")
            that_bai.append(ten_file)
        print()

    print("=" * 50)
    print(f"Hoàn tất cắt ảnh: {len(thanh_cong)}/{len(du_lieu)} ảnh thành công.")
    if that_bai:
        print(f"Các ảnh bị lỗi ({len(that_bai)}): {', '.join(that_bai)}")
        print("  LƯU Ý: ảnh lỗi KHÔNG được ghi đè, nên file crop cũ (nếu có) "
              "vẫn nằm lại trong thư mục - lượt 4 sẽ bỏ qua chúng.")
    print()
    return file_crop


def luot_3_dong_bo_huong(thu_muc_anh: str, du_lieu: dict, thu_muc_crop: str,
                          duong_dan_anh_chuan_co_dinh: str = None):
    """Xác định hướng ĐÚNG bằng SIFT trên ẢNH GỐC ĐẦY ĐỦ giữa mỗi ảnh và 1
    ảnh chuẩn -> lệch ~180° thì lật, lệch ~0° thì giữ nguyên.
    SAU BƯỚC NÀY mọi ảnh crop mới cùng hướng -> lượt 4 mới áp mask được."""
    danh_sach = [f for f in du_lieu
                 if du_lieu[f]["ti_le_lom"] <= NGUONG_TI_LE_LOM]
    if not danh_sach:
        print("[3/4] Mọi ảnh đã xác định hướng chắc chắn bằng hình học "
              "-> bỏ qua bước đối chiếu SIFT.\n")
        return
    print(f"[3/4] {len(danh_sach)} ảnh có hướng chưa chắc -> đối chiếu SIFT.")

    pad = 60

    if duong_dan_anh_chuan_co_dinh:
        print(f"[3/4] Đồng bộ hướng trái/phải bằng SIFT (ảnh gốc), "
              f"dùng ẢNH CHUẨN CỐ ĐỊNH: {duong_dan_anh_chuan_co_dinh}\n")
        img_chuan = cv2.imread(duong_dan_anh_chuan_co_dinh)
        if img_chuan is None:
            raise FileNotFoundError(
                f"Không đọc được ảnh chuẩn cố định: {duong_dan_anh_chuan_co_dinh}")
        contour_chuan = tim_contour_tu_dong(img_chuan)
        if contour_chuan is None:
            raise RuntimeError(
                f"Không tìm thấy vật thể trong ảnh chuẩn cố định: {duong_dan_anh_chuan_co_dinh}")
        m_align_chuan, _, _ = tinh_m_align_va_roi(contour_chuan)
        ten_file_chuan = None
    else:
        ten_file_chuan = sorted(danh_sach)[0]
        print(f"[3/4] Đồng bộ hướng trái/phải bằng SIFT (ảnh gốc), "
              f"dùng ảnh chuẩn (ảnh đầu tiên trong thư mục): {ten_file_chuan}\n")
        img_chuan = cv2.imread(os.path.join(thu_muc_anh, ten_file_chuan))
        contour_chuan = du_lieu[ten_file_chuan]["contour"]
        m_align_chuan = du_lieu[ten_file_chuan]["m_align"]

    x0, y0, w0, h0 = cv2.boundingRect(contour_chuan)
    mask_chuan = np.zeros(img_chuan.shape[:2], np.uint8)
    mask_chuan[max(0, y0 - pad):y0 + h0 + pad, max(0, x0 - pad):x0 + w0 + pad] = 255
    kp_chuan, des_chuan = trich_sift_anh(img_chuan, mask_full=mask_chuan)

    R_chuan_inv = cv2.invertAffineTransform(m_align_chuan)[:, :2]
    dir_chuan = R_chuan_inv @ np.array([1.0, 0.0])

    so_lat = 0
    so_khong_chac = 0
    for ten_file in danh_sach:
        if ten_file == ten_file_chuan:
            continue
        d = du_lieu[ten_file]
        img = cv2.imread(os.path.join(thu_muc_anh, ten_file))
        xg, yg, wg, hg = cv2.boundingRect(d["contour"])
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, yg - pad):yg + hg + pad, max(0, xg - pad):xg + wg + pad] = 255
        kp, des = trich_sift_anh(img, mask_full=mask)

        M, so_inlier = uoc_luong_M_sift(kp, des, kp_chuan, des_chuan)
        if M is None:
            print(f"  [{ten_file}] Không đủ tin cậy để xác định hướng "
                  f"({so_inlier} match) - GIỮ NGUYÊN theo mặc định.")
            so_khong_chac += 1
            continue

        R_inv = cv2.invertAffineTransform(d["m_align"])[:, :2]
        dir_nay = R_inv @ np.array([1.0, 0.0])
        dir_nay_trong_he_chuan = M[:, :2] @ dir_nay

        goc_do = np.degrees(np.arccos(
            np.clip(np.dot(dir_nay_trong_he_chuan, dir_chuan) /
                    (np.linalg.norm(dir_nay_trong_he_chuan) * np.linalg.norm(dir_chuan) + 1e-9),
                    -1, 1)
        ))

        if goc_do > 90:
            duong_dan_crop = os.path.join(thu_muc_crop, f"{os.path.splitext(ten_file)[0]}_roi.png")
            crop = cv2.imread(duong_dan_crop)
            if crop is not None:
                cv2.imwrite(duong_dan_crop, cv2.rotate(crop, cv2.ROTATE_180))
            print(f"  [{ten_file}] LẬT 180° (lệch góc {goc_do:.0f}°, {so_inlier} inlier)")
            so_lat += 1
        else:
            print(f"  [{ten_file}] giữ nguyên (lệch góc {goc_do:.0f}°, {so_inlier} inlier)")

    tong_so_anh_xet = len(danh_sach) - (1 if ten_file_chuan else 0)
    print(f"\nĐã lật {so_lat}/{tong_so_anh_xet} ảnh để đồng bộ hướng "
          f"({so_khong_chac} ảnh không đủ tin cậy, giữ nguyên mặc định).\n")


def luot_4_ap_mask_hinh_hoc(thu_muc_crop: str, file_crop: list = None):
    """Áp mask hình học CỐ ĐỊNH lên các ảnh crop vừa tạo Ở LẦN CHẠY NÀY.

    file_crop = danh sách tên file lượt 2 vừa ghi. File crop cũ còn sót lại
    trong thư mục (từ lần chạy trước, hoặc của ảnh bị lỗi lần này) sẽ được
    liệt kê ra và BỎ QUA, vì kích thước/hướng của chúng có thể khác chuẩn.

    Mask dựng theo TỈ LỆ nên khớp với mọi kích thước ảnh - không bỏ qua ảnh
    chỉ vì lệch vài pixel so với ảnh đầu tiên."""
    co_trong_thu_muc = sorted(f for f in os.listdir(thu_muc_crop)
                              if f.lower().endswith("_roi.png"))
    if file_crop is None:
        danh_sach = co_trong_thu_muc
    else:
        hop_le = set(file_crop)
        danh_sach = [f for f in co_trong_thu_muc if f in hop_le]
        thua = [f for f in co_trong_thu_muc if f not in hop_le]
        if thua:
            print(f"  CẢNH BÁO: {len(thua)} file crop cũ còn sót, KHÔNG áp mask "
                  f"(nên xoá đi để khỏi lẫn vào tập training):")
            for f in thua:
                print(f"    - {f}")

    if not danh_sach:
        print("[4/4] Không có ảnh crop nào để áp mask.")
        return

    mau = cv2.imread(os.path.join(thu_muc_crop, danh_sach[0]))
    H, W = mau.shape[:2]
    mask = tao_mask_hinh_hoc(W, H)

    if CHE_DO_XEM_THU:
        thu_muc_xem = os.path.join(thu_muc_crop, "_xem_thu")
        os.makedirs(thu_muc_xem, exist_ok=True)
        print(f"[4/4] CHẾ ĐỘ XEM THỬ - xuất {len(danh_sach)} ảnh có viền mask "
              f"vào: {thu_muc_xem}\n      (ảnh crop thật KHÔNG bị ghi đè)\n")
        for ten_file in danh_sach:
            img = cv2.imread(os.path.join(thu_muc_crop, ten_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            m = mask if (h, w) == (H, W) else tao_mask_hinh_hoc(w, h)
            xem = img.copy()
            vien, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(xem, vien, -1, (0, 0, 255), max(2, w // 500))
            if CAT_SAT_MASK:
                bx, by, bw, bh = cv2.boundingRect(m)
                cv2.rectangle(xem, (max(0, bx - LE_TRONG), max(0, by - LE_TRONG)),
                              (min(w - 1, bx + bw + LE_TRONG),
                               min(h - 1, by + bh + LE_TRONG)),
                              (0, 255, 255), max(2, w // 500))
            cv2.imwrite(os.path.join(thu_muc_xem, ten_file), xem)
        print("  -> Xong. Chỉnh toạ độ trong khối THAM SỐ MASK HÌNH HỌC, "
              "ưng ý thì đặt CHE_DO_XEM_THU = False rồi chạy lại.")
        return

    print(f"[4/4] Áp mask hình học cho {len(danh_sach)} ảnh (khung chuẩn {W}x{H})...")
    if CAT_SAT_MASK:
        bx, by, bw, bh = cv2.boundingRect(mask)
        print(f"      cắt sát mask -> kích thước cuối "
              f"{min(W, bx + bw + LE_TRONG) - max(0, bx - LE_TRONG)}x"
              f"{min(H, by + bh + LE_TRONG) - max(0, by - LE_TRONG)}")
    print()

    so_ok, so_bo_qua, kich_thuoc = 0, 0, {}
    for ten_file in danh_sach:
        duong_dan = os.path.join(thu_muc_crop, ten_file)
        img = cv2.imread(duong_dan)
        if img is None:
            so_bo_qua += 1
            continue
        h, w = img.shape[:2]
        m = mask if (h, w) == (H, W) else tao_mask_hinh_hoc(w, h)
        if (h, w) != (H, W):
            print(f"  [{ten_file}] kích thước {w}x{h} lệch chuẩn - dựng mask "
                  f"riêng theo tỉ lệ.")
        img[m == 0] = 0
        if CAT_SAT_MASK:
            bx, by, bw, bh = cv2.boundingRect(m)
            img = img[max(0, by - LE_TRONG):min(h, by + bh + LE_TRONG),
                      max(0, bx - LE_TRONG):min(w, bx + bw + LE_TRONG)]
        cv2.imwrite(duong_dan, img)
        kich_thuoc[(img.shape[1], img.shape[0])] = \
            kich_thuoc.get((img.shape[1], img.shape[0]), 0) + 1
        so_ok += 1

    print(f"\n  -> Đã áp mask cho {so_ok} ảnh ({so_bo_qua} ảnh bỏ qua).")
    if len(kich_thuoc) > 1:
        print("  CẢNH BÁO: ảnh output KHÔNG cùng kích thước:")
        for kt, n in sorted(kich_thuoc.items()):
            print(f"    {kt[0]}x{kt[1]}: {n} ảnh")


def tao_bo_template(thu_muc_anh_mau: str, thu_muc_crop_mau: str, duong_dan_anh_chuan: str = None):
    du_lieu, dien_tich_trung_vi, canonical_w, canonical_h = luot_1_do_kich_thuoc(thu_muc_anh_mau)
    file_crop = luot_2_cat_anh(thu_muc_anh_mau, du_lieu, dien_tich_trung_vi,
                               canonical_w, canonical_h, thu_muc_crop_mau)
    luot_3_dong_bo_huong(thu_muc_anh_mau, du_lieu, thu_muc_crop_mau, duong_dan_anh_chuan)
    if AP_MASK_HINH_HOC:
        luot_4_ap_mask_hinh_hoc(thu_muc_crop_mau, file_crop)


if __name__ == "__main__":
    tao_bo_template(DUONG_DAN_THU_MUC_ANH_MAU, DUONG_DAN_THU_MUC_CROP_MAU,
                     DUONG_DAN_ANH_CHUAN_CO_DINH)