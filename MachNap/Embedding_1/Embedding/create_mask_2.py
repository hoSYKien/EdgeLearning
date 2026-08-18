
# """
# Pipeline: Robust Background Subtraction → MobileSAM (vật chuyển động trên băng tải)
#   [1]   Color diff per-channel max
#   [2]   Loại bóng (3 điều kiện)
#   [2.5] Hysteresis 2 ngưỡng (chống loang)
#   [2.7] Lấp thủng: flood-fill lỗ kín + convex hull khuyết góc (có kiểm soát)
#   [3]   Xác nhận mức BLOB bằng tracking → tạo STABLE MASK HOÀN CHỈNH
#   [4]   (CUỐI CÙNG) Watershed trên stable mask → bbox
#   [5]   Bbox → MobileSAM segment (nguyên bản)
# """

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import time
import get_data_irayple

points = np.array([
    [10, 10],
    [670, 10],
    [670, 490],
    [10, 490]
], dtype=np.int32)

roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(points)

# Hysteresis 2 ngưỡng
THRESH_STRONG = 20
THRESH_WEAK = 10

CHROMA_TOL = 0.04
SHADOW_DIM = (0.35, 0.95)
SHADOW_MAX_DIFF = 70

MIN_AREA = 1000
PADDING = 15

# Lấp thủng
SOLIDITY_MIN = 0.85

# Xác nhận blob bằng tracking
N_CONFIRM = 2
MAX_MOVE = 180
MAX_MISSED = 2

# Watershed (bước cuối)
SPLIT_MIN_AREA = 500   # blob nhỏ hơn mức này không bao giờ tách
MIN_PEAK_DISTANCE = 60
BORDER_GRAD_MIN = 45

# Học nền
WARMUP_FRAMES = 20
BG_FRAMES = 30
BG_MAX_RESIDUAL = 8.0


# ================================

def get_roi(frame):
    roi = frame[
        roi_y:roi_y + roi_h,
        roi_x:roi_x + roi_w
    ]

    shifted_points = points.copy()
    shifted_points[:, 0] -= roi_x
    shifted_points[:, 1] -= roi_y

    mask = np.zeros((roi_h, roi_w), np.uint8)
    cv2.fillPoly(mask, [shifted_points], 255)

    return cv2.bitwise_and(roi, roi, mask=mask)

def get_resized_frame(camera):
    img = camera.current_frame
    if img is not None:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 6, h // 6))
    return img

def learn_background(camera, num_frames=BG_FRAMES, warmup_frames=WARMUP_FRAMES):
    print(f"Chờ camera ổn định ({warmup_frames} frames)...")
    skipped = 0
    while skipped < warmup_frames:
        img = get_resized_frame(camera)
        if img is None:
            continue
        skipped += 1
        disp = img.copy()
        cv2.putText(disp, f"Camera warming up... {skipped}/{warmup_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.imshow("Detection", disp)
        cv2.waitKey(1)

    print("Học nền... giữ ROI trống")
    frames = []
    while len(frames) < num_frames:
        img = get_resized_frame(camera)
        if img is None:
            continue
        frames.append(cv2.GaussianBlur(get_roi(img), (7, 7), 0).astype(np.float32))
        disp = img.copy()
        cv2.polylines(disp, [points], True, (0, 255, 0), 2)
        cv2.putText(disp, f"Learning BG... {len(frames)}/{num_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detection", disp)
        cv2.waitKey(1)
    print("Xong!")
    return np.mean(frames, axis=0)


def learn_background_verified(camera):
    warmup = WARMUP_FRAMES
    while True:
        bg = learn_background(camera, warmup_frames=warmup)
        img = None
        while img is None:
            img = get_resized_frame(camera)
        cur = cv2.GaussianBlur(get_roi(img), (7, 7), 0).astype(np.float32)
        residual = np.mean(np.abs(cur - bg))
        print(f"Độ lệch nền vs frame hiện tại: {residual:.1f}")
        if residual < BG_MAX_RESIDUAL:
            return bg
        print("Nền chưa ổn định, học lại...")
        warmup = 30


def fill_holes(mask):
    """Flood-fill từ góc → vùng không tràn tới = lỗ kín → tô đầy."""
    h, w = mask.shape
    ff = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def solidify(mask, solidity_min=SOLIDITY_MIN):
    """Convex hull lấp khuyết góc/hông — chỉ khi blob đã gần lồi."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and area / hull_area >= solidity_min:
            cv2.drawContours(out, [hull], -1, 255, -1)
        else:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


NECK_RATIO = 0.6   # bề dày tại cổ < 60% bề dày 2 đỉnh → có eo thắt thật
def has_neck(dist, p1, p2):
    """Kiểm tra đường nối 2 đỉnh có đi qua chỗ thắt (mỏng hơn hẳn) không.
    p1, p2: (y, x). Trả về True nếu có eo thắt thật."""
    n_samples = 50
    ys = np.linspace(p1[0], p2[0], n_samples).astype(int)
    xs = np.linspace(p1[1], p2[1], n_samples).astype(int)
    profile = dist[ys, xs]                    # bề dày dọc đường nối

    peak_thickness = min(dist[p1[0], p1[1]], dist[p2[0], p2[1]])
    neck_thickness = profile.min()

    return neck_thickness < peak_thickness * NECK_RATIO


def watershed_split(blob_mask, roi_bgr):
    """CHỈ tách khi có eo thắt hình học thật sự.
    Vật dài/to không có cổ → không bao giờ bị cắt, bất kể bao nhiêu đỉnh."""
    dist = ndimage.distance_transform_edt(blob_mask)
    coords = peak_local_max(dist, min_distance=MIN_PEAK_DISTANCE,
                            labels=blob_mask.astype(bool))
    if len(coords) <= 1:
        return [blob_mask]

    # === KIỂM TRA EO THẮT: chỉ giữ các đỉnh thật sự bị ngăn cách bởi cổ ===
    # Gom đỉnh: 2 đỉnh không có cổ giữa chúng = cùng 1 vật → giữ 1 đại diện
    kept = []
    for c in coords:
        merged = False
        for k in kept:
            if not has_neck(dist, tuple(c), tuple(k)):
                merged = True          # không có cổ → cùng vật với đỉnh đã giữ
                break
        if not merged:
            kept.append(tuple(c))

    if len(kept) <= 1:
        return [blob_mask]             # mọi đỉnh cùng 1 vật → không tách

    # === Có >= 2 đỉnh bị cổ ngăn cách → tách bằng watershed trên distance ===
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(kept):
        markers[y, x] = i + 1

    labels = watershed(-dist, markers, mask=blob_mask.astype(bool))

    sub_masks = []
    for lb in range(1, labels.max() + 1):
        sub = ((labels == lb) * 255).astype(np.uint8)
        if cv2.countNonZero(sub) >= MIN_AREA:
            sub_masks.append(sub)

    return sub_masks if sub_masks else [blob_mask]

class Detector:
    """Bước [1]→[3]: tạo STABLE MASK hoàn chỉnh. KHÔNG watershed ở đây."""

    def __init__(self, roi_shape, bg):
        self.bg = bg.astype(np.float32)
        self.bg_uint8 = bg.astype(np.uint8)
        self.tracks = []
        
        # Tối ưu hóa: tính sẵn chroma và mean của background một lần duy nhất bằng các hàm OpenCV
        bg_b, bg_g, bg_r = cv2.split(self.bg)
        self.bg_sum = bg_b + bg_g + bg_r + 3.0
        self.bg_sum_3d = cv2.merge([self.bg_sum, self.bg_sum, self.bg_sum])
        self.bg_plus_1 = self.bg + 1.0
        
        # Tính sẵn các ngưỡng độ sáng cho shadow removal
        self.shadow_dim_low = SHADOW_DIM[0] * self.bg_sum
        self.shadow_dim_high = SHADOW_DIM[1] * self.bg_sum

    def _decay_tracks(self, roi_shape):
        """Frame không có vật: già hoá track (missed++) & trả mask rỗng."""
        kept = []
        for t in self.tracks:
            t['missed'] += 1
            if t['missed'] <= MAX_MISSED:
                kept.append(t)
        self.tracks = kept
        return np.zeros((roi_shape[0], roi_shape[1]), dtype=np.uint8)

    def build_mask(self, roi):
        # [1] Kiểm tra nhanh ở dạng uint8 để thoát cực sớm nếu không có vật thể
        roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)
        diff_c_uint8 = cv2.absdiff(roi_blur, self.bg_uint8)
        b_u, g_u, r_u = cv2.split(diff_c_uint8)
        diff_uint8 = cv2.max(cv2.max(b_u, g_u), r_u)
        
        # Chỉ chạy tiếp nếu có ít nhất 1 pixel biến động đủ lớn (THRESH_STRONG - 1 để bù sai số làm tròn)
        if cv2.minMaxLoc(diff_uint8)[1] < THRESH_STRONG - 1:
            return self._decay_tracks(roi.shape)

        # [2] Xác định vùng hoạt động (weak) ngay trên diff uint8 rẻ, chạy full ROI.
        #     Nhờ vậy toàn bộ toán float khử bóng đắt đỏ chỉ chạy trên vùng crop nhỏ.
        weak_full = (diff_uint8 > THRESH_WEAK).astype(np.uint8)
        x_w, y_w, w_w, h_w = cv2.boundingRect(weak_full)
        if w_w == 0 or h_w == 0:
            return self._decay_tracks(roi.shape)

        # Thêm padding an toàn để tránh mất biên khi chạy Morphology
        pad = 5
        y1 = max(0, y_w - pad)
        y2 = min(roi.shape[0], y_w + h_w + pad)
        x1 = max(0, x_w - pad)
        x2 = min(roi.shape[1], x_w + w_w + pad)

        # [3] Diff float + khử bóng CHỈ trên vùng crop (dùng các mảng nền cắt tương ứng)
        roi_f = roi_blur[y1:y2, x1:x2].astype(np.float32)
        diff_c = cv2.absdiff(roi_f, self.bg[y1:y2, x1:x2])
        b_f, g_f, r_f = cv2.split(diff_c)
        diff = cv2.max(cv2.max(b_f, g_f), r_f)

        # S_roi phải là TỔNG ĐỘ SÁNG của pixel hiện tại (roi_b+roi_g+roi_r),
        # KHÔNG phải tổng độ lệch |roi-bg| — nếu không mô hình bóng sẽ sai hoàn toàn.
        rb, rg, rr = cv2.split(roi_f)
        S_roi = rb + rg + rr + 3.0
        S_roi_3d = cv2.merge([S_roi, S_roi, S_roi])

        # so khớp màu sắc: | (I_i + 1) * S_bg - (BG_i + 1) * S_roi | < CHROMA_TOL * S_roi * S_bg
        term_roi = (roi_f + 1.0) * self.bg_sum_3d[y1:y2, x1:x2]
        term_bg = self.bg_plus_1[y1:y2, x1:x2] * S_roi_3d

        chroma_diff = cv2.absdiff(term_roi, term_bg)
        bc, gc, rc = cv2.split(chroma_diff)
        max_diff_term = cv2.max(cv2.max(bc, gc), rc)

        threshold_map = (CHROMA_TOL * self.bg_sum[y1:y2, x1:x2]) * S_roi
        same_color = max_diff_term < threshold_map

        # kiểm tra tỉ lệ độ sáng: SHADOW_DIM[0] * S_bg < S_roi < SHADOW_DIM[1] * S_bg
        dim_moderate = ((S_roi > self.shadow_dim_low[y1:y2, x1:x2]) &
                        (S_roi < self.shadow_dim_high[y1:y2, x1:x2]))

        # xóa bỏ bóng đổ
        diff[same_color & dim_moderate & (diff < SHADOW_MAX_DIFF)] = 0

        # [4] Hysteresis trên vùng crop
        strong_crop = (diff > THRESH_STRONG).astype(np.uint8)
        weak_crop = (diff > THRESH_WEAK).astype(np.uint8)

        # Sau khi khử bóng, nếu không còn seed strong nào thì coi như không có vật
        if cv2.countNonZero(strong_crop) == 0:
            return self._decay_tracks(roi.shape)

        n_lbl, lbl_crop = cv2.connectedComponents(weak_crop, connectivity=8)
        strong_labels = np.unique(lbl_crop[strong_crop > 0])
        strong_labels = strong_labels[strong_labels != 0]
        
        # Ánh xạ nhãn nhanh
        map_arr = np.zeros(n_lbl, dtype=bool)
        map_arr[strong_labels] = True
        mask_crop = map_arr[lbl_crop].astype(np.uint8) * 255

        # [5] Morphology trên vùng Crop (nhỏ hơn hàng chục lần so với ảnh gốc)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, k)
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k)

        # [6] Lấp lỗ kín trên vùng Crop (an toàn tuyệt đối nhờ đệm 1px)
        h_c, w_c = mask_crop.shape
        padded = np.zeros((h_c + 2, w_c + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = mask_crop
        ff = padded.copy()
        ff_mask = np.zeros((h_c + 4, w_c + 4), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        filled_padded = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
        mask_crop = filled_padded[1:-1, 1:-1]

        # [7] Tìm contours trên vùng Crop & chuyển đổi tọa độ về ảnh gốc (Chỉ chạy findContours 1 lần!)
        h_full, w_full = roi.shape[:2]
        raw_blobs = []
        
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0 and area / hull_area >= SOLIDITY_MIN:
                chosen_c = hull
                chosen_area = hull_area
            else:
                chosen_c = c
                chosen_area = area
                
            if chosen_area < MIN_AREA or chosen_area > h_full * w_full * 0.85:
                continue
                
            # Dịch chuyển tọa độ contour từ vùng crop về ROI đầy đủ
            chosen_c[:, 0, 0] += x1
            chosen_c[:, 0, 1] += y1
            
            x, y, bw, bh = cv2.boundingRect(chosen_c)
            raw_blobs.append((chosen_c, (x + bw / 2, y + bh / 2)))

        # [8] Tracking & Vẽ lên stable mask đầy đủ
        for t in self.tracks:
            t['matched'] = False
        new_tracks = []
        stable = np.zeros((h_full, w_full), dtype=np.uint8)

        for c, center in raw_blobs:
            best_t, best_d = None, MAX_MOVE
            for t in self.tracks:
                if t['matched']:
                    continue
                d = np.hypot(center[0] - t['center'][0], center[1] - t['center'][1])
                if d < best_d:
                    best_d, best_t = d, t
            if best_t is not None:
                best_t['matched'] = True
                best_t['center'] = center
                best_t['age'] += 1
                best_t['missed'] = 0
                new_tracks.append(best_t)
                if best_t['age'] >= N_CONFIRM:
                    cv2.drawContours(stable, [c], -1, 255, -1)
            else:
                new_tracks.append({'center': center, 'age': 1, 'missed': 0,
                                   'matched': True})

        for t in self.tracks:
            if not t['matched']:
                t['missed'] += 1
                if t['missed'] <= MAX_MISSED:
                    new_tracks.append(t)
        self.tracks = new_tracks

        return stable   # mask hoàn chỉnh, chưa tách


def extract_bboxes(stable_mask, roi):
    """[4] BƯỚC CUỐI: watershed trên stable mask hoàn chỉnh → bbox.
    Blob nhỏ hơn SPLIT_MIN_AREA giữ nguyên; chỉ blob to bất thường mới tách."""
    h, w = stable_mask.shape
    bboxes = []
    contours, _ = cv2.findContours(stable_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        if area < SPLIT_MIN_AREA:
            # Lấy trực tiếp bbox từ contour hiện tại, không cần vẽ lại và tìm contour lần 2
            x, y, bw, bh = cv2.boundingRect(c)
            bboxes.append([max(0, x-PADDING), max(0, y-PADDING),
                           min(w, x+bw+PADDING), min(h, y+bh+PADDING)])
        else:
            # Tối ưu hóa: Kiểm tra độ lồi (solidity). Nếu lồi (như hộp đơn lẻ) thì không cần chạy Watershed
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            
            if solidity >= 0.92:
                # Vật thể đơn lẻ dạng lồi -> Bỏ qua hoàn toàn Watershed để tiết kiệm CPU
                x, y, bw, bh = cv2.boundingRect(c)
                bboxes.append([max(0, x-PADDING), max(0, y-PADDING),
                               min(w, x+bw+PADDING), min(h, y+bh+PADDING)])
            else:
                # Nghi ngờ nhiều vật thể chạm nhau -> Chỉ chạy Watershed trên vùng ảnh crop nhỏ của riêng blob này
                x, y, bw, bh = cv2.boundingRect(c)
                
                # Tạo mask nhỏ vừa khít với blob
                blob_mask_crop = np.zeros((bh, bw), dtype=np.uint8)
                c_shifted = c.copy()
                c_shifted[:, 0, 0] -= x
                c_shifted[:, 0, 1] -= y
                cv2.drawContours(blob_mask_crop, [c_shifted], -1, 255, -1)
                
                roi_crop = roi[y:y+bh, x:x+bw]
                
                # Chạy watershed trên mask nhỏ
                subs_crop = watershed_split(blob_mask_crop, roi_crop)
                
                for sub_crop in subs_crop:
                    sub_contours, _ = cv2.findContours(sub_crop, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    for sc in sub_contours:
                        if cv2.contourArea(sc) < MIN_AREA:
                            continue
                        x_s, y_s, bw_s, bh_s = cv2.boundingRect(sc)
                        abs_x = x + x_s
                        abs_y = y + y_s
                        bboxes.append([max(0, abs_x-PADDING), max(0, abs_y-PADDING),
                                       min(w, abs_x+bw_s+PADDING), min(h, abs_y+bh_s+PADDING)])
    return bboxes


if __name__ == "__main__":
    camera = get_data_irayple.CameraIndustrial(index=0)
    if not camera.open():
        print("Failed to open Irayple camera.")
        exit(1)
    
    camera.start()

    # model = SAM(PATH_MOBILE_SAM)
    # model = FastSAM(PAHT_FASTSAM)

    bg = learn_background_verified(camera)
    detector = Detector((roi_h, roi_w), bg)

    try:
        while True:
            img = get_resized_frame(camera)
            if img is None:
                continue

            roi = get_roi(img)
            start_cr_mask = time.perf_counter()
            # Bước 1-3: tạo stable mask hoàn chỉnh
            stable_mask = detector.build_mask(roi)

            # Bước 4 (cuối): watershed trên mask hoàn chỉnh → bbox
            bboxes = extract_bboxes(stable_mask, roi)
            end_cr_mask = time.perf_counter()
            # print(f"Time creat mask: {(end_cr_mask - start_cr_mask):.3f}")
            # print(len(bboxes))`   `````````````
            disp = img.copy()

            for b in bboxes:
                cv2.rectangle(disp, (b[0]+roi_x, b[1]+roi_y), (b[2]+roi_x, b[3]+roi_y),
                              (0, 255, 255), 2)

            cv2.polylines(disp, [points], True, (0, 255, 0), 2)
            cv2.putText(disp, f"Objects: {len(bboxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Detection", disp)
            cv2.imshow("Mask", stable_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                bg = learn_background_verified(camera)
                detector = Detector((roi_h, roi_w), bg)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
