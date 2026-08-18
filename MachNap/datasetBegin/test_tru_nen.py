r"""
test_tru_nen.py - Học nền từ 1 thư mục ảnh nền, rồi chạy detect trên thư mục
ảnh có vật. HOÀN TOÀN KHÔNG CẦN CAMERA - đọc ảnh từ đĩa.

Rút gọn từ pipeline băng tải: giữ nguyên [1] color diff, [2] khử bóng,
[2.5] hysteresis, [2.7] lấp thủng + convex hull; BỎ tracking (ảnh tĩnh trên
đĩa, mỗi ảnh 1 frame nên tracking vô nghĩa) và BỎ watershed (bật lại được
bằng TACH_VAT_CHAM_NHAU nếu ảnh có nhiều vật dính nhau).

    python test_tru_nen.py
        -> học nền, chạy hết thư mục vật, hiện từng ảnh (phím bất kỳ = ảnh kế,
           's' = lưu ảnh kết quả, 'q' = thoát)

    python test_tru_nen.py --luu-het
        -> không hiện cửa sổ, xử lý hết và lưu kết quả vào THU_MUC_KET_QUA

Nền học bằng TRUNG VỊ nên 6 ảnh nền lệch nhẹ sáng/góc vẫn ra nền sạch.
"""

import os
import sys
import glob

import cv2
import numpy as np

# ====================== CHỈNH Ở ĐÂY ======================
THU_MUC_NEN = r"D:\TongHop\RTC Technologi\PCB\BackGround"       # 6 ảnh nền trống
THU_MUC_VAT = r"C:\Users\LAP273\MVS\Data"                       # ảnh có vật
THU_MUC_KET_QUA = r"D:\TongHop\RTC Technologi\PCB\ket_qua_detect"

# Thu nhỏ ảnh về bề rộng này để xử lý cho nhanh (contour tự phóng về gốc).
BE_RONG_XU_LY = 1900

# Hysteresis 2 ngưỡng. Vật khó tách khỏi nền -> giảm cả hai.
THRESH_STRONG = 10
THRESH_WEAK = 5


# Khử bóng: bóng làm TỐI nền nhưng KHÔNG đổi sắc màu.
CHROMA_TOL = 0.04
SHADOW_DIM = (0.4, 0.95)
SHADOW_MAX_DIFF = 70

# Blob nhỏ hơn ngần này (tỉ lệ diện tích ảnh đã thu nhỏ) thì bỏ.
TI_LE_DIEN_TICH_MIN = 0.003
SOLIDITY_MIN = 0.85
PADDING = 10                 # nới bbox ra vài px cho dễ nhìn

# Chỉ lấy 1 vật lớn nhất (True) hay mọi vật đạt ngưỡng (False).
CHI_LAY_LON_NHAT = False

# Xoay vật cho nằm NGANG rồi cắt sát (minAreaRect) thay vì cắt bbox thẳng.
# True  -> vật được xoay thẳng, cắt ôm sát viền, ít nền thừa nhất
# False -> cắt bbox thẳng đứng như cũ (nhanh hơn, nhưng vật nghiêng thừa nền)
XOAY_VAT_NAM_NGANG = True
PAD_XOAY = 8            # chừa vài px quanh vật sau khi xoay cho khỏi gặm mép

# Bôi ĐEN mọi pixel NGOÀI viền vật (dùng chính mask). Sau khi xoay+cắt vẫn
# còn 4 góc nền lọt vào khung chữ nhật -> bật cái này cho nền đen tuyệt đối,
# model không học nhầm nền. Chỉ giữ đúng phần board.
XOA_NEN_NGOAI_MASK = True
CO_MASK = 2            # co viền mask vào trong vài px cho ăn hết mép sáng của
                       # board (0 = giữ nguyên viền, tăng nếu còn rìa nền)

# Tách vật chạm nhau bằng watershed. Cần scipy + skimage.
TACH_VAT_CHAM_NHAU = False
SPLIT_MIN_AREA = 10000       # blob nhỏ hơn mức này không bao giờ tách
MIN_PEAK_DISTANCE = 60
NECK_RATIO = 0.6

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
# =========================================================


def liet_ke(thu_muc):
    if not os.path.isdir(thu_muc):
        raise SystemExit(f"Không có thư mục: {thu_muc}")
    ra = []
    for e in VALID_EXT:
        ra += glob.glob(os.path.join(thu_muc, "*" + e))
        ra += glob.glob(os.path.join(thu_muc, "*" + e.upper()))
    return sorted(set(ra))


def thu_nho(img, be_rong=BE_RONG_XU_LY):
    h, w = img.shape[:2]
    ti_le = min(1.0, be_rong / w)
    if ti_le >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(w * ti_le), int(h * ti_le))), ti_le


# ====================================================================
# HỌC NỀN
# ====================================================================

def hoc_nen(thu_muc_nen):
    """Trung vị của các ảnh nền (đã thu nhỏ + làm mờ). Trung vị để 1-2 ảnh
    lệch sáng/góc không kéo nền đi."""
    ten = liet_ke(thu_muc_nen)
    if not ten:
        raise SystemExit(f"Không có ảnh nền nào trong: {thu_muc_nen}")
    print(f"Học nền từ {len(ten)} ảnh trong {thu_muc_nen}")

    xep, kich_thuoc = [], None
    for t in ten:
        img = cv2.imread(t)
        if img is None:
            print(f"  bỏ qua (không đọc được): {os.path.basename(t)}")
            continue
        nho, _ = thu_nho(img)
        if kich_thuoc is None:
            kich_thuoc = nho.shape
        elif nho.shape != kich_thuoc:
            print(f"  bỏ qua (khác kích thước): {os.path.basename(t)}")
            continue
        xep.append(cv2.GaussianBlur(nho, (7, 7), 0).astype(np.float32))
        print(f"  + {os.path.basename(t)}")
    if not xep:
        raise SystemExit("Không dùng được ảnh nền nào.")

    nen = np.median(np.stack(xep), axis=0).astype(np.float32)

    # tự kiểm tra: độ lệch của TỪNG ảnh nền so với nền trung vị
    print("Độ lệch từng ảnh nền so với nền trung vị:")
    for t, x in zip(ten, xep):
        print(f"  {os.path.basename(t):40s} {np.mean(cv2.absdiff(x, nen)):.1f}")
    return nen


# ====================================================================
# TẠO MASK BẰNG TRỪ NỀN (lõi thuật toán, bỏ tracking)
# ====================================================================

class TruNen:
    def __init__(self, nen):
        self.nen = nen.astype(np.float32)
        self.nen_u8 = nen.astype(np.uint8)
        b, g, r = cv2.split(self.nen)
        self.nen_sum = b + g + r + 3.0
        self.nen_sum_3d = cv2.merge([self.nen_sum] * 3)
        self.nen_plus_1 = self.nen + 1.0
        self.dim_thap = SHADOW_DIM[0] * self.nen_sum
        self.dim_cao = SHADOW_DIM[1] * self.nen_sum

    def _khu_bong(self, diff, roi_f, y1, y2, x1, x2):
        rb, rg, rr = cv2.split(roi_f)
        S_roi = rb + rg + rr + 3.0
        S_roi_3d = cv2.merge([S_roi] * 3)
        term_roi = (roi_f + 1.0) * self.nen_sum_3d[y1:y2, x1:x2]
        term_nen = self.nen_plus_1[y1:y2, x1:x2] * S_roi_3d
        bc, gc, rc = cv2.split(cv2.absdiff(term_roi, term_nen))
        lech_mau = cv2.max(cv2.max(bc, gc), rc)
        cung_mau = lech_mau < (CHROMA_TOL * self.nen_sum[y1:y2, x1:x2]) * S_roi
        toi_vua = ((S_roi > self.dim_thap[y1:y2, x1:x2]) &
                   (S_roi < self.dim_cao[y1:y2, x1:x2]))
        diff[cung_mau & toi_vua & (diff < SHADOW_MAX_DIFF)] = 0
        return diff

    def tao_mask(self, img_nho):
        """img_nho: ảnh đã thu nhỏ. Trả về mask uint8 hoặc None."""
        H, W = img_nho.shape[:2]
        blur = cv2.GaussianBlur(img_nho, (7, 7), 0)

        # [1] diff nhanh trên uint8 để khoanh vùng, tránh toán float cả ảnh
        b, g, r = cv2.split(cv2.absdiff(blur, self.nen_u8))
        diff_u8 = cv2.max(cv2.max(b, g), r)
        if cv2.minMaxLoc(diff_u8)[1] < THRESH_STRONG - 1:
            return None
        xw, yw, ww, hw = cv2.boundingRect((diff_u8 > THRESH_WEAK).astype(np.uint8))
        if ww == 0 or hw == 0:
            return None
        pad = 5
        y1, y2 = max(0, yw - pad), min(H, yw + hw + pad)
        x1, x2 = max(0, xw - pad), min(W, xw + ww + pad)

        # [2] diff float + khử bóng trên vùng crop
        roi_f = blur[y1:y2, x1:x2].astype(np.float32)
        b, g, r = cv2.split(cv2.absdiff(roi_f, self.nen[y1:y2, x1:x2]))
        diff = cv2.max(cv2.max(b, g), r)
        diff = self._khu_bong(diff, roi_f, y1, y2, x1, x2)

        # [2.5] hysteresis
        strong = (diff > THRESH_STRONG).astype(np.uint8)
        weak = (diff > THRESH_WEAK).astype(np.uint8)
        if cv2.countNonZero(strong) == 0:
            return None
        n_lbl, lbl = cv2.connectedComponents(weak, connectivity=8)
        nhan = np.unique(lbl[strong > 0])
        nhan = nhan[nhan != 0]
        giu = np.zeros(n_lbl, bool)
        giu[nhan] = True
        mask_crop = giu[lbl].astype(np.uint8) * 255

        # [2.7] morphology + lấp lỗ kín
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, k)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k)
        mask_crop = _lap_lo(mask_crop)

        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = mask_crop
        return mask


def _lap_lo(mask):
    h, w = mask.shape
    dem = np.zeros((h + 2, w + 2), np.uint8)
    dem[1:-1, 1:-1] = mask
    ff = dem.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(dem, cv2.bitwise_not(ff))[1:-1, 1:-1]


# ====================================================================
# XOAY VẬT NẰM NGANG + CẮT SÁT (minAreaRect)
# ====================================================================

def cat_xoay_nam_ngang(img_goc, contour_goc, mask_goc=None):
    """Xoay vật cho nằm ngang rồi cắt ôm sát viền.

    Dùng minAreaRect: hình chữ nhật XOAY nhỏ nhất bao contour -> diện tích
    vật/diện tích khung gần bằng nhau nhất. Luôn ép CHIỀU DÀI > CHIỀU CAO
    nên vật luôn nằm ngang.

    Nếu XOA_NEN_NGOAI_MASK và có mask_goc: bôi ĐEN mọi pixel ngoài viền vật
    (4 góc nền còn lọt vào khung chữ nhật) -> nền đen tuyệt đối.

    Trả về (ảnh vật đã xoay-cắt, góc đã xoay). Toạ độ contour/mask phải là
    của ẢNH GỐC."""
    (cx, cy), (w, h), goc = cv2.minAreaRect(contour_goc)
    if w < h:                      # ép cạnh dài nằm ngang
        w, h = h, w
        goc += 90.0

    M = cv2.getRotationMatrix2D((cx, cy), goc, 1.0)
    xoay = cv2.warpAffine(img_goc, M, (img_goc.shape[1], img_goc.shape[0]))

    w2 = int(round(w)) + 2 * PAD_XOAY
    h2 = int(round(h)) + 2 * PAD_XOAY
    crop = cv2.getRectSubPix(xoay, (w2, h2), (cx, cy))

    if XOA_NEN_NGOAI_MASK and mask_goc is not None:
        m = mask_goc
        if CO_MASK > 0:            # co viền vào trong cho ăn hết mép sáng
            k = np.ones((2 * CO_MASK + 1,) * 2, np.uint8)
            m = cv2.erode(m, k)
        # mask đi qua ĐÚNG phép xoay + cắt như ảnh
        m_xoay = cv2.warpAffine(m, M, (img_goc.shape[1], img_goc.shape[0]),
                                flags=cv2.INTER_NEAREST)
        m_crop = cv2.getRectSubPix(m_xoay, (w2, h2), (cx, cy))
        crop[m_crop < 128] = 0     # ngoài viền vật -> đen

    return crop, goc


def contour_goc_lon_nhat(mask_nho, ti_le):
    """Contour vật LỚN NHẤT, phóng toạ độ về ảnh gốc. None nếu không có."""
    contours, _ = cv2.findContours(mask_nho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    H, W = mask_nho.shape
    dt_min = TI_LE_DIEN_TICH_MIN * H * W
    hop_le = [c for c in contours if dt_min <= cv2.contourArea(c) <= H * W * 0.95]
    if not hop_le:
        return None
    c = max(hop_le, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    dth = cv2.contourArea(hull)
    if dth > 0 and cv2.contourArea(c) / dth >= SOLIDITY_MIN:
        c = hull
    return (c.astype(np.float32) / ti_le).astype(np.int32)


# ====================================================================
# TÁCH VẬT CHẠM NHAU (tùy chọn, từ code băng tải)
# ====================================================================

def _has_neck(dist, p1, p2):
    ys = np.linspace(p1[0], p2[0], 50).astype(int)
    xs = np.linspace(p1[1], p2[1], 50).astype(int)
    peak = min(dist[p1[0], p1[1]], dist[p2[0], p2[1]])
    return dist[ys, xs].min() < peak * NECK_RATIO


def _watershed_split(blob_mask, min_area):
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    dist = ndimage.distance_transform_edt(blob_mask)
    coords = peak_local_max(dist, min_distance=MIN_PEAK_DISTANCE,
                            labels=blob_mask.astype(bool))
    if len(coords) <= 1:
        return [blob_mask]
    kept = []
    for c in coords:
        if not any(not _has_neck(dist, tuple(c), tuple(k)) for k in kept):
            kept.append(tuple(c))
    if len(kept) <= 1:
        return [blob_mask]
    markers = np.zeros(dist.shape, np.int32)
    for i, (y, x) in enumerate(kept):
        markers[y, x] = i + 1
    labels = watershed(-dist, markers, mask=blob_mask.astype(bool))
    subs = [((labels == lb) * 255).astype(np.uint8) for lb in range(1, labels.max() + 1)]
    return [s for s in subs if cv2.countNonZero(s) >= min_area] or [blob_mask]


# ====================================================================
# MASK -> DANH SÁCH BBOX (toạ độ ảnh GỐC)
# ====================================================================

def lay_bboxes(mask, ti_le):
    """mask ở ảnh thu nhỏ -> list bbox ở ảnh gốc."""
    H, W = mask.shape
    dt_min = TI_LE_DIEN_TICH_MIN * H * W
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chon = []
    for c in contours:
        dt = cv2.contourArea(c)
        if dt < dt_min or dt > H * W * 0.95:
            continue
        hull = cv2.convexHull(c)
        dt_hull = cv2.contourArea(hull)
        if dt_hull > 0 and dt / dt_hull >= SOLIDITY_MIN:
            c, dt = hull, dt_hull
        chon.append((c, dt))

    if not chon:
        return []
    if CHI_LAY_LON_NHAT:
        chon = [max(chon, key=lambda cd: cd[1])]

    bboxes = []
    for c, dt in chon:
        cons = [c]
        if TACH_VAT_CHAM_NHAU and dt >= SPLIT_MIN_AREA:
            hull = cv2.convexHull(c)
            dth = cv2.contourArea(hull)
            if dth > 0 and dt / dth < 0.92:      # nghi ngờ nhiều vật dính
                x, y, bw, bh = cv2.boundingRect(c)
                m = np.zeros((bh, bw), np.uint8)
                cc = c.copy()
                cc[:, 0, 0] -= x
                cc[:, 0, 1] -= y
                cv2.drawContours(m, [cc], -1, 255, -1)
                cons = []
                for sub in _watershed_split(m, dt_min):
                    sc, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for s in sc:
                        s[:, 0, 0] += x
                        s[:, 0, 1] += y
                        cons.append(s)
        for cc in cons:
            x, y, bw, bh = cv2.boundingRect(cc)
            bboxes.append([int(max(0, x - PADDING) / ti_le),
                           int(max(0, y - PADDING) / ti_le),
                           int(min(W, x + bw + PADDING) / ti_le),
                           int(min(H, y + bh + PADDING) / ti_le)])
    return bboxes


# ====================================================================
def ve_ket_qua(img, mask_nho, bboxes, ti_le, ten):
    """Ảnh gốc + bbox + mask phủ mờ, thu nhỏ để hiển thị."""
    disp = img.copy()
    # phủ mask (phóng về gốc) cho thấy vùng nào bị coi là vật
    mask_goc = cv2.resize(mask_nho, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    xanh = np.zeros_like(disp)
    xanh[mask_goc > 0] = (0, 180, 0)
    disp = cv2.addWeighted(disp, 1.0, xanh, 0.25, 0)
    for b in bboxes:
        cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 3)
    cv2.putText(disp, f"{ten}  |  {len(bboxes)} vat", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    t = min(1.0, 1300 / max(disp.shape[:2]))
    return cv2.resize(disp, None, fx=t, fy=t) if t < 1.0 else disp


def main():
    luu_het = "--luu-het" in sys.argv

    nen = hoc_nen(THU_MUC_NEN)
    print(f"Nền: {nen.shape[1]}x{nen.shape[0]} (đã thu nhỏ)\n")
    bo = TruNen(nen)

    ten_vat = liet_ke(THU_MUC_VAT)
    if not ten_vat:
        raise SystemExit(f"Không có ảnh vật nào trong: {THU_MUC_VAT}")
    print(f"Chạy detect trên {len(ten_vat)} ảnh trong {THU_MUC_VAT}")
    if luu_het:
        os.makedirs(THU_MUC_KET_QUA, exist_ok=True)

    co_vat = 0
    for i, t in enumerate(ten_vat, 1):
        img = cv2.imread(t)
        if img is None:
            print(f"[{i}/{len(ten_vat)}] bỏ qua (không đọc được): {os.path.basename(t)}")
            continue
        nho, ti_le = thu_nho(img)
        if nho.shape[:2] != nen.shape[:2]:
            print(f"[{i}/{len(ten_vat)}] {os.path.basename(t)}: kích thước "
                  f"{nho.shape[1]}x{nho.shape[0]} khác nền {nen.shape[1]}x{nen.shape[0]} "
                  "-> nền và ảnh vật phải cùng độ phân giải camera.")
            continue

        mask = bo.tao_mask(nho)
        if mask is None:
            mask = np.zeros(nho.shape[:2], np.uint8)
        bboxes = lay_bboxes(mask, ti_le)
        if bboxes:
            co_vat += 1

        # --- xoay vật nằm ngang + cắt sát (chỉ với vật lớn nhất) ---
        anh_xoay = None
        if XOAY_VAT_NAM_NGANG:
            c_goc = contour_goc_lon_nhat(mask, ti_le)
            if c_goc is not None:
                mask_goc = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                anh_xoay, goc_xoay = cat_xoay_nam_ngang(img, c_goc, mask_goc)
                dt_vat = cv2.contourArea(c_goc)
                dt_khung = anh_xoay.shape[0] * anh_xoay.shape[1]
                lap_day = dt_vat / dt_khung if dt_khung else 0
                # so với bbox thẳng để thấy cải thiện
                x, y, bw, bh = cv2.boundingRect(c_goc)
                lap_day_thang = dt_vat / (bw * bh) if bw * bh else 0
                print(f"[{i}/{len(ten_vat)}] {os.path.basename(t):40s} -> "
                      f"{len(bboxes)} vật | xoay {goc_xoay:+.1f}° | "
                      f"lấp đầy {lap_day:.0%} (bbox thẳng chỉ {lap_day_thang:.0%})")
            else:
                print(f"[{i}/{len(ten_vat)}] {os.path.basename(t):40s} -> {len(bboxes)} vật")
        else:
            print(f"[{i}/{len(ten_vat)}] {os.path.basename(t):40s} -> {len(bboxes)} vật")

        khung = ve_ket_qua(img, mask, bboxes, ti_le, os.path.basename(t))
        if luu_het:
            cv2.imwrite(os.path.join(THU_MUC_KET_QUA, f"detect_{os.path.basename(t)}"),
                        khung)
            if anh_xoay is not None:
                cv2.imwrite(os.path.join(THU_MUC_KET_QUA, f"xoay_{os.path.basename(t)}"),
                            anh_xoay)
            continue

        cv2.imshow("Detect (phim bat ky=ke tiep, s=luu, q=thoat)", khung)
        if anh_xoay is not None:
            t2 = min(1.0, 700 / max(anh_xoay.shape[:2]))
            xem_xoay = cv2.resize(anh_xoay, None, fx=t2, fy=t2) if t2 < 1.0 else anh_xoay
            cv2.imshow("Vat da xoay nam ngang + cat sat", xem_xoay)
        phim = cv2.waitKey(0) & 0xFF
        if phim == ord('q'):
            break
        if phim == ord('s'):
            os.makedirs(THU_MUC_KET_QUA, exist_ok=True)
            cv2.imwrite(os.path.join(THU_MUC_KET_QUA, f"detect_{os.path.basename(t)}"), khung)
            if anh_xoay is not None:
                cv2.imwrite(os.path.join(THU_MUC_KET_QUA, f"xoay_{os.path.basename(t)}"), anh_xoay)
            print(f"    đã lưu vào {THU_MUC_KET_QUA}")

    cv2.destroyAllWindows()
    print(f"\nXong. {co_vat}/{len(ten_vat)} ảnh tìm thấy vật.")
    if luu_het:
        print(f"Kết quả đã lưu ở: {THU_MUC_KET_QUA}")


if __name__ == "__main__":
    main()