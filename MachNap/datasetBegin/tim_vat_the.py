r"""
tim_vat_the.py - Tìm vật thể bằng TRỪ NỀN (thay cho threshold Saturation).

Lấy lõi thuật toán từ pipeline băng tải, BỎ phần tracking (ở trạm chụp vật
đứng yên, mỗi lần chỉ có 1 frame nên tracking không xác nhận được gì) và
BỎ watershed (mỗi ảnh chỉ có 1 con hàng).

Các bước giữ nguyên:
    [1] diff màu, lấy max theo 3 kênh
    [2] khử bóng: cùng màu + tối vừa phải + lệch nhỏ  -> coi là bóng
    [3] hysteresis 2 ngưỡng (strong/weak) chống loang
    [4] morphology + lấp lỗ kín + convex hull khi blob đã gần lồi
    [5] lấy blob LỚN NHẤT (thay cho tracking)

Ưu điểm so với threshold Saturation: bắt được vật MỌI MÀU, không phụ thuộc
board xanh hay nền sáng, sang dự án mới không phải chỉnh gì ngoài học lại nền.

Chạy riêng để soi:
    python tim_vat_the.py anh_chup.png      -> hiện mask + contour
"""

import os
import sys

import cv2
import numpy as np

import cauhinh as cf


# ==================== NỀN ====================

def hoc_nen_tu_frames(frames):
    """TRUNG VỊ của N frame nền trống -> ảnh nền float32.

    Cố ý dùng trung vị chứ không phải trung bình: lỡ có 1-2 frame dính tay
    người, dính vật lọt vào khung, hay 1 frame bị nhiễu sáng thì trung vị
    vẫn cho ra nền sạch, còn trung bình sẽ bị kéo lệch vĩnh viễn."""
    if not frames:
        raise ValueError("Không có frame nào để học nền")
    xu_ly = [cv2.GaussianBlur(f, (7, 7), 0).astype(np.float32) for f in frames]
    return np.median(np.stack(xu_ly), axis=0).astype(np.float32)


def luu_nen(bg, duong_dan=None):
    duong_dan = duong_dan or cf.FILE_NEN
    os.makedirs(os.path.dirname(duong_dan) or ".", exist_ok=True)
    np.save(duong_dan, bg.astype(np.float32))
    cv2.imwrite(os.path.splitext(duong_dan)[0] + ".png", bg.astype(np.uint8))


def nap_nen(duong_dan=None):
    duong_dan = duong_dan or cf.FILE_NEN
    if not os.path.isfile(duong_dan):
        raise FileNotFoundError(
            f"Chưa có ảnh nền: {duong_dan}\n"
            "-> Chạy  python hoc_nen.py  (dọn trống bàn/đồ gá trước).")
    return np.load(duong_dan).astype(np.float32)


def do_lech_nen(img, bg):
    """Độ lệch trung bình giữa ảnh hiện tại và nền - để biết nền còn dùng
    được không (đèn đổi, camera bị đụng, đồ gá xê dịch...)."""
    cur = cv2.GaussianBlur(img, (7, 7), 0).astype(np.float32)
    if cur.shape != bg.shape:
        return float("inf")
    return float(np.mean(np.abs(cur - bg)))


# ==================== TÌM VẬT ====================

def _le(w, co_ban, chuan=1000):
    """Kích thước kernel co giãn theo độ phân giải ảnh (ảnh 3000px cần
    kernel to hơn ảnh 500px), luôn trả số lẻ."""
    return max(3, int(round(co_ban * w / chuan))) | 1


class TruNen:
    """Tạo mask vật thể từ 1 ảnh, so với ảnh nền đã học."""

    def __init__(self, bg):
        self.bg = bg.astype(np.float32)
        self.bg_uint8 = bg.astype(np.uint8)
        self.mau = (self.bg.ndim == 3)      # ảnh màu hay mono

        if self.mau:
            b, g, r = cv2.split(self.bg)
            self.bg_sum = b + g + r + 3.0
            self.bg_sum_3d = cv2.merge([self.bg_sum] * 3)
            self.bg_plus_1 = self.bg + 1.0
            self.dim_thap = cf.SHADOW_DIM[0] * self.bg_sum
            self.dim_cao = cf.SHADOW_DIM[1] * self.bg_sum

    # ---------- [2] khử bóng ----------
    def _khu_bong(self, diff, roi_f, y1, y2, x1, x2):
        """Bóng đổ = cùng màu với nền + tối đi vừa phải + lệch không lớn."""
        if not self.mau:
            return diff                     # ảnh mono: không tách được bóng
        rb, rg, rr = cv2.split(roi_f)
        S_roi = rb + rg + rr + 3.0
        S_roi_3d = cv2.merge([S_roi] * 3)

        term_roi = (roi_f + 1.0) * self.bg_sum_3d[y1:y2, x1:x2]
        term_bg = self.bg_plus_1[y1:y2, x1:x2] * S_roi_3d
        bc, gc, rc = cv2.split(cv2.absdiff(term_roi, term_bg))
        lech_mau = cv2.max(cv2.max(bc, gc), rc)

        cung_mau = lech_mau < (cf.CHROMA_TOL * self.bg_sum[y1:y2, x1:x2]) * S_roi
        toi_vua = ((S_roi > self.dim_thap[y1:y2, x1:x2]) &
                   (S_roi < self.dim_cao[y1:y2, x1:x2]))
        diff[cung_mau & toi_vua & (diff < cf.SHADOW_MAX_DIFF)] = 0
        return diff

    def tao_mask(self, img):
        """Trả về mask uint8 (0/255) của vật thể, hoặc None nếu không thấy."""
        H, W = img.shape[:2]
        dien_tich_min = cf.TI_LE_DIEN_TICH_MIN * H * W

        # [1] diff nhanh trên uint8 để thoát sớm
        blur = cv2.GaussianBlur(img, (_le(W, 7),) * 2, 0)
        diff_u8 = cv2.absdiff(blur, self.bg_uint8)
        if diff_u8.ndim == 3:
            b, g, r = cv2.split(diff_u8)
            diff_u8 = cv2.max(cv2.max(b, g), r)
        if cv2.minMaxLoc(diff_u8)[1] < cf.THRESH_STRONG - 1:
            return None

        # khoanh vùng weak để toán float chỉ chạy trên crop nhỏ
        weak_full = (diff_u8 > cf.THRESH_WEAK).astype(np.uint8)
        xw, yw, ww, hw = cv2.boundingRect(weak_full)
        if ww == 0 or hw == 0:
            return None
        pad = _le(W, 5)
        y1, y2 = max(0, yw - pad), min(H, yw + hw + pad)
        x1, x2 = max(0, xw - pad), min(W, xw + ww + pad)

        # [2] diff float + khử bóng trên crop
        roi_f = blur[y1:y2, x1:x2].astype(np.float32)
        diff = cv2.absdiff(roi_f, self.bg[y1:y2, x1:x2])
        if diff.ndim == 3:
            b, g, r = cv2.split(diff)
            diff = cv2.max(cv2.max(b, g), r)
        diff = self._khu_bong(diff, roi_f, y1, y2, x1, x2)

        # [3] hysteresis: giữ vùng weak nào có chạm seed strong
        strong = (diff > cf.THRESH_STRONG).astype(np.uint8)
        weak = (diff > cf.THRESH_WEAK).astype(np.uint8)
        if cv2.countNonZero(strong) == 0:
            return None
        n_lbl, lbl = cv2.connectedComponents(weak, connectivity=8)
        nhan_strong = np.unique(lbl[strong > 0])
        nhan_strong = nhan_strong[nhan_strong != 0]
        bang = np.zeros(n_lbl, dtype=bool)
        bang[nhan_strong] = True
        mask_crop = bang[lbl].astype(np.uint8) * 255

        # [4] morphology + lấp lỗ kín (đệm 1px cho flood-fill an toàn)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_le(W, 5),) * 2)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, k)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k)
        mask_crop = _lap_lo(mask_crop)

        # [5] lấy blob LỚN NHẤT (thay cho tracking của bản băng tải)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        dien_tich = cv2.contourArea(c)
        if dien_tich < dien_tich_min or dien_tich > H * W * 0.95:
            return None

        # convex hull chỉ khi blob đã gần lồi (lấp khuyết góc, không nuốt lỗ to)
        hull = cv2.convexHull(c)
        dt_hull = cv2.contourArea(hull)
        if dt_hull > 0 and dien_tich / dt_hull >= cf.SOLIDITY_MIN:
            c = hull

        c[:, 0, 0] += x1
        c[:, 0, 1] += y1
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
        return mask


def _lap_lo(mask):
    """Flood-fill từ ngoài vào; chỗ không tràn tới là lỗ kín -> tô đầy."""
    h, w = mask.shape
    dem = np.zeros((h + 2, w + 2), np.uint8)
    dem[1:-1, 1:-1] = mask
    ff = dem.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(dem, cv2.bitwise_not(ff))[1:-1, 1:-1]


# ==================== BACKEND CŨ (threshold Saturation) ====================

def mask_saturation(img, target_w=1200):
    """Cách cũ: threshold kênh Saturation. Chỉ hợp với board màu trên nền
    khác màu rõ. Giữ lại để đối chiếu."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        return None
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    def k(size):
        return max(3, round(size * scale) // 2 * 2 + 1)

    _, s, _ = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    _, m = cv2.threshold(cv2.medianBlur(s, k(15)), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k(7), k(7)), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    lon = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = np.where(lbl == lon, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k(41), k(41)), np.uint8))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


# ==================== API DÙNG CHUNG ====================

_bo_tru_nen = None


def lay_bo_tru_nen():
    global _bo_tru_nen
    if _bo_tru_nen is None:
        _bo_tru_nen = TruNen(nap_nen())
    return _bo_tru_nen


def tim_mask(img):
    """Mask vật thể theo backend đã chọn trong cauhinh.KIEU_TIM_VAT."""
    if cf.KIEU_TIM_VAT == "tru_nen":
        return lay_bo_tru_nen().tao_mask(img)
    if cf.KIEU_TIM_VAT == "saturation":
        return mask_saturation(img)
    raise ValueError('KIEU_TIM_VAT phải là "tru_nen" hoặc "saturation"')


def tim_contour(img):
    """Contour ngoài của vật thể (định dạng như cv2.findContours) hoặc None."""
    mask = tim_mask(img)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python tim_vat_the.py anh_chup.png")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Không đọc được ảnh:", sys.argv[1])
        raise SystemExit(1)

    print(f"Ảnh {img.shape[1]}x{img.shape[0]} | backend: {cf.KIEU_TIM_VAT}")
    if cf.KIEU_TIM_VAT == "tru_nen":
        bg = nap_nen()
        print(f"Nền: {cf.FILE_NEN} ({bg.shape[1]}x{bg.shape[0]})")
        if bg.shape[:2] != img.shape[:2]:
            print("*** Nền và ảnh KHÁC kích thước -> học lại nền.")
            raise SystemExit(1)

    mask = tim_mask(img)
    if mask is None:
        print("KHÔNG tìm thấy vật thể.")
        raise SystemExit(1)

    dt = cv2.countNonZero(mask)
    x, y, w, h = cv2.boundingRect(mask)
    print(f"Diện tích: {dt} px ({dt / (img.shape[0] * img.shape[1]):.1%} ảnh)")
    print(f"Bounding box: x{x} y{y} {w}x{h}")

    xem = img.copy()
    xem[mask == 0] = xem[mask == 0] // 3          # tối phần nền cho dễ nhìn
    cv2.drawContours(xem, cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)[0],
                     -1, (0, 255, 255), 3)
    ti_le = min(1.0, 1200 / max(xem.shape[:2]))
    cv2.imshow("Vat the tim duoc - nhan phim bat ky de dong",
               cv2.resize(xem, None, fx=ti_le, fy=ti_le))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
