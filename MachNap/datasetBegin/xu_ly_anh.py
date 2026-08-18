r"""
xu_ly_anh.py - Cắt vật thể ra khỏi ảnh chụp: tìm contour -> làm thẳng ->
ép USB về bên PHẢI -> cắt ROI có padding -> áp mask hình học (nền đen) ->
cắt sát mask.

DÙNG CHUNG cho cả script gán nhãn lẫn script chạy thật -> chỉ sửa 1 chỗ,
không sợ dataset lệch hệ toạ độ so với lúc kiểm tra.

Chạy riêng để soi kết quả crop 1 ảnh:
    python xu_ly_anh.py anh_chup.png
    python xu_ly_anh.py            (chỉ in kích thước output + xem mask)
"""

import os
import sys

import cv2
import numpy as np

import cauhinh as cf


# ==================== TÌM VẬT THỂ ====================

def tim_contour_tu_dong(img, target_w=1200):
    """Threshold Saturation + lấy thành phần liên thông lớn nhất + lấp lỗ."""
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    def k(size):
        return max(3, round(size * scale) // 2 * 2 + 1)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    blur = cv2.medianBlur(s, k(15))
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k(7), k(7)), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    lon_nhat = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == lon_nhat, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k(41), k(41)), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return (contour.astype(np.float32) / scale).astype(np.int32)


def _do_do_lom(contour_aligned):
    """Độ lõm ăn vào từ mép trái / mép phải (contour đã xoay thẳng).
    Phía lắp USB lõm sâu vì contour không bắt được vỏ kim loại."""
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
    """Trả về (M_align, roi, ti_le_lom); đã ép USB về bên PHẢI."""
    (cx, cy), _, angle = cv2.minAreaRect(contour)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ca = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M)
    x, y, w, h = cv2.boundingRect(ca.astype(np.int32))

    if h > w:      # dựng đứng -> xoay 90 cho nằm ngang
        M90 = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 90, 1.0)
        M = (np.vstack([M90, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2, :]
        ca = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M)
        x, y, w, h = cv2.boundingRect(ca.astype(np.int32))

    lom_trai, lom_phai = _do_do_lom(ca.astype(np.int32))
    if lom_trai > lom_phai:    # USB đang ở bên trái -> lật 180
        M180 = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 180, 1.0)
        M = (np.vstack([M180, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2, :]
        ca = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M)
        x, y, w, h = cv2.boundingRect(ca.astype(np.int32))
    ti_le_lom = max(lom_trai, lom_phai) / (min(lom_trai, lom_phai) + 1e-6)

    x -= cf.PAD_ROI_NGANG
    y -= cf.PAD_ROI_DOC
    w += 2 * cf.PAD_ROI_NGANG
    h += 2 * cf.PAD_ROI_DOC
    return M, (x, y, w, h), ti_le_lom


def cat_an_toan(img, x, y, w, h):
    """Cắt kể cả khi tràn mép ảnh; phần tràn lấp ĐEN."""
    H, W = img.shape[:2]
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 > x1 and y2 > y1:
        out[y1 - y:y2 - y, x1 - x:x2 - x] = img[y1:y2, x1:x2]
    return out


# ==================== MASK HÌNH HỌC ====================

def _chu_nhat_bo_goc(mask, x1, y1, x2, y2, r, mau):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(mask, (x1, y1), (x2, y2), mau, -1)
        return
    cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), mau, -1)
    cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), mau, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(mask, (cx, cy), r, mau, -1)


def tao_mask_hinh_hoc(W, H):
    """Mask cố định cho khung crop W x H (USB nằm bên PHẢI)."""
    mask = np.zeros((H, W), np.uint8)
    _chu_nhat_bo_goc(mask, int(cf.BOARD_X1 * W), int(cf.BOARD_Y1 * H),
                     int(cf.BOARD_X2 * W), int(cf.BOARD_Y2 * H),
                     int(cf.BOARD_BO_GOC * H), 255)
    _chu_nhat_bo_goc(mask, int(cf.USB_X1 * W), int(cf.USB_Y1 * H),
                     int(cf.USB_X2 * W), int(cf.USB_Y2 * H),
                     int(cf.USB_BO_GOC * H), 255)
    if cf.KHUYET_X2 > 0:
        _chu_nhat_bo_goc(mask, 0, int(cf.KHUYET_Y1 * H),
                         int(cf.KHUYET_X2 * W), int(cf.KHUYET_Y2 * H),
                         int(cf.KHUYET_BO_GOC * H), 0)
    if cf.NOI_MASK > 0:
        mask = cv2.dilate(mask, np.ones((2 * cf.NOI_MASK + 1,) * 2, np.uint8))
    elif cf.NOI_MASK < 0:
        mask = cv2.erode(mask, np.ones((2 * -cf.NOI_MASK + 1,) * 2, np.uint8))
    if cf.LAM_MUOT_BIEN and cf.LAM_MUOT_BIEN >= 3:
        mask = cv2.medianBlur(mask, cf.LAM_MUOT_BIEN | 1)
    return mask


def kich_thuoc_output_chuan():
    """Kích thước ảnh crop cuối - suy ra từ mask, không cần chụp thử."""
    if not cf.AP_MASK_HINH_HOC or not cf.CAT_SAT_MASK:
        return cf.KHUNG_CHUAN_W, cf.KHUNG_CHUAN_H
    mask = tao_mask_hinh_hoc(cf.KHUNG_CHUAN_W, cf.KHUNG_CHUAN_H)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return (min(cf.KHUNG_CHUAN_W, bx + bw + cf.LE_TRONG) - max(0, bx - cf.LE_TRONG),
            min(cf.KHUNG_CHUAN_H, by + bh + cf.LE_TRONG) - max(0, by - cf.LE_TRONG))


# ==================== SIFT (nhánh dự phòng xác định hướng) ====================

def trich_sift_anh(img, target_w=None, mask_full=None, nfeatures=None):
    target_w = target_w or cf.SIFT_TARGET_WIDTH
    nfeatures = nfeatures or cf.SIFT_NFEATURES
    h, w = img.shape[:2]
    scale = min(1.0, target_w / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask_small = None
    if mask_full is not None:
        mask_small = cv2.resize(mask_full, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    kp, des = cv2.SIFT_create(nfeatures=nfeatures).detectAndCompute(gray, mask_small)
    for kpt in kp:
        kpt.pt = (kpt.pt[0] / scale, kpt.pt[1] / scale)
    return kp, des


def uoc_luong_M_sift(kp1, des1, kp2, des2, min_inliers=None):
    min_inliers = min_inliers or cf.MIN_INLIERS_HUONG
    if des1 is None or des2 is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
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


def chuan_bi_anh_chuan(duong_dan=None):
    """Dữ liệu SIFT của ảnh chuẩn - CHỈ dùng cho nhánh dự phòng."""
    duong_dan = duong_dan or cf.DUONG_DAN_ANH_CHUAN_CO_DINH
    img = cv2.imread(duong_dan)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh chuẩn: {duong_dan}")
    contour = tim_contour_tu_dong(img)
    if contour is None:
        raise RuntimeError(f"Không tìm thấy vật thể trong ảnh chuẩn: {duong_dan}")
    m_align, _, ti_le_lom = tinh_m_align_va_roi(contour)

    xg, yg, wg, hg = cv2.boundingRect(contour)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[max(0, yg - cf.PAD_MASK_SIFT):yg + hg + cf.PAD_MASK_SIFT,
         max(0, xg - cf.PAD_MASK_SIFT):xg + wg + cf.PAD_MASK_SIFT] = 255
    kp, des = trich_sift_anh(img, mask_full=mask)

    dir_chuan = cv2.invertAffineTransform(m_align)[:, :2] @ np.array([1.0, 0.0])
    return {"kp_chuan": kp, "des_chuan": des, "dir_chuan": dir_chuan,
            "ti_le_lom": ti_le_lom}


# ==================== HÀM CHÍNH ====================

def crop_va_dong_bo_huong(img, anh_chuan_info, tra_mask=False):
    """Trả về (ảnh crop đã xoá nền, thông báo) hoặc (None, lý do lỗi).

    tra_mask=True -> trả (crop, thong_bao, mask_align) với mask_align là mask
    vật thể đã đi qua đúng phép xoay + cắt như ảnh, dùng làm nguyên liệu cho
    tao_mask_chuan.py."""
    import tim_vat_the as tv

    contour = tv.tim_contour(img)
    if contour is None:
        loi = ("Không tìm thấy vật thể trong ảnh."
               + (" (trừ nền: vật quá giống nền, hoặc nền cũ rồi -> chạy "
                  "hoc_nen.py)" if cf.KIEU_TIM_VAT == "tru_nen" else ""))
        return (None, loi, None) if tra_mask else (None, loi)

    m_align, (x, y, w, h), ti_le_lom = tinh_m_align_va_roi(contour)
    warped = cv2.warpAffine(img, m_align, (img.shape[1], img.shape[0]))
    crop = cat_an_toan(warped, x, y, w, h)

    # mask vật thể đi qua ĐÚNG phép biến đổi như ảnh
    mask_vat = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask_vat, [contour], -1, 255, cv2.FILLED)
    mask_align = cat_an_toan(
        cv2.warpAffine(mask_vat, m_align, (img.shape[1], img.shape[0]),
                       flags=cv2.INTER_NEAREST), x, y, w, h)

    if crop.shape[1] != cf.KHUNG_CHUAN_W or crop.shape[0] != cf.KHUNG_CHUAN_H:
        crop = cv2.resize(crop, (cf.KHUNG_CHUAN_W, cf.KHUNG_CHUAN_H))
        mask_align = cv2.resize(mask_align, (cf.KHUNG_CHUAN_W, cf.KHUNG_CHUAN_H),
                                interpolation=cv2.INTER_NEAREST)

    if ti_le_lom > cf.NGUONG_TI_LE_LOM:
        thong_bao = f"Hướng xác định bằng hình học (tỉ lệ lõm {ti_le_lom:.1f})."
    else:
        xg, yg, wg, hg = cv2.boundingRect(contour)
        mask_sift = np.zeros(img.shape[:2], np.uint8)
        mask_sift[max(0, yg - cf.PAD_MASK_SIFT):yg + hg + cf.PAD_MASK_SIFT,
                  max(0, xg - cf.PAD_MASK_SIFT):xg + wg + cf.PAD_MASK_SIFT] = 255
        kp, des = trich_sift_anh(img, mask_full=mask_sift)
        M, so_inlier = uoc_luong_M_sift(kp, des, anh_chuan_info["kp_chuan"],
                                        anh_chuan_info["des_chuan"])
        if M is None:
            thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) và SIFT cũng "
                         f"không đủ tin cậy ({so_inlier} match) - SOI BẰNG MẮT.")
        else:
            dir_nay = cv2.invertAffineTransform(m_align)[:, :2] @ np.array([1.0, 0.0])
            v = M[:, :2] @ dir_nay
            dc = anh_chuan_info["dir_chuan"]
            goc = np.degrees(np.arccos(np.clip(
                np.dot(v, dc) / (np.linalg.norm(v) * np.linalg.norm(dc) + 1e-9), -1, 1)))
            if goc > 90:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                mask_align = cv2.rotate(mask_align, cv2.ROTATE_180)
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT LẬT "
                             f"180° (lệch {goc:.0f}°, {so_inlier} inlier).")
            else:
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT giữ "
                             f"nguyên (lệch {goc:.0f}°, {so_inlier} inlier).")

    mask = lay_mask_xoa_nen(crop.shape[1], crop.shape[0], mask_align)
    if mask is not None:
        crop[mask == 0] = 0
        if cf.CAT_SAT_MASK:
            bx, by, bw, bh = cv2.boundingRect(mask)
            y1c = max(0, by - cf.LE_TRONG)
            y2c = min(crop.shape[0], by + bh + cf.LE_TRONG)
            x1c = max(0, bx - cf.LE_TRONG)
            x2c = min(crop.shape[1], bx + bw + cf.LE_TRONG)
            crop = crop[y1c:y2c, x1c:x2c]
            mask_align = mask_align[y1c:y2c, x1c:x2c]

    return (crop, thong_bao, mask_align) if tra_mask else (crop, thong_bao)


def lay_mask_xoa_nen(W, H, mask_align=None):
    """Mask dùng để xoá nền, theo cf.NGUON_MASK. None = không xoá."""
    if cf.NGUON_MASK == "tat" or not cf.AP_MASK_HINH_HOC:
        return None
    if cf.NGUON_MASK == "hinh_hoc":
        return tao_mask_hinh_hoc(W, H)
    if cf.NGUON_MASK == "tuc_thi":
        return mask_align
    if cf.NGUON_MASK == "file":
        m = cv2.imread(cf.FILE_MASK_CHUAN, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(
                f"Chưa có mask chuẩn: {cf.FILE_MASK_CHUAN}\n"
                "-> Chạy  python tao_mask_chuan.py  (cần vài chục ảnh OK).")
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m
    raise ValueError(f'NGUON_MASK không hợp lệ: {cf.NGUON_MASK}')


def thu_nho_de_xem(img, max_dim=900):
    h, w = img.shape[:2]
    ti_le = min(1.0, max_dim / max(h, w))
    return img if ti_le >= 1.0 else cv2.resize(img, (int(w * ti_le), int(h * ti_le)))


if __name__ == "__main__":
    print(f"Khung chuẩn {cf.KHUNG_CHUAN_W}x{cf.KHUNG_CHUAN_H} -> ảnh crop cuối: "
          f"{kich_thuoc_output_chuan()[0]}x{kich_thuoc_output_chuan()[1]}")

    if len(sys.argv) < 2:
        print("\nXem thử mask hình học. Nhấn phím bất kỳ để đóng.")
        print("Muốn crop thử 1 ảnh: python xu_ly_anh.py anh_chup.png")
        mask = tao_mask_hinh_hoc(cf.KHUNG_CHUAN_W, cf.KHUNG_CHUAN_H)
        cv2.imshow("Mask hinh hoc", thu_nho_de_xem(mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()

    duong_dan = sys.argv[1]
    img = cv2.imread(duong_dan)
    if img is None:
        print(f"Không đọc được ảnh: {duong_dan}")
        sys.exit(1)
    print(f"Ảnh vào: {img.shape[1]}x{img.shape[0]}")

    info = chuan_bi_anh_chuan()
    print(f"Ảnh chuẩn có tỉ lệ lõm {info['ti_le_lom']:.1f}")

    crop, thong_bao = crop_va_dong_bo_huong(img, info)
    if crop is None:
        print("CROP LỖI:", thong_bao)
        sys.exit(1)
    print(thong_bao)
    print(f"Ảnh crop: {crop.shape[1]}x{crop.shape[0]}")

    ra = os.path.splitext(duong_dan)[0] + "_thu_crop.png"
    cv2.imwrite(ra, crop)
    print(f"Đã lưu: {ra}")
    cv2.imshow("Anh da crop - nhan phim bat ky de dong", thu_nho_de_xem(crop))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
