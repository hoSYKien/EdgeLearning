"""
Chụp ảnh trực tiếp từ camera Hikrobot (nhấn 'c' để chụp, 'q' để thoát).

Mỗi lần nhấn 'c':
    1. Lưu ảnh gốc (chưa crop) vào THU_MUC_ANH_GOC
    2. Tự động cắt ROI vật thể (tìm contour, làm thẳng, cắt về kích thước
       chuẩn) + đồng bộ hướng trái/phải bằng SIFT -> hiển thị ("Anh da crop")
    3. Đưa THẲNG ảnh đã crop (không cắt nhỏ thêm) vào model PatchCore
       (anomalib) để tính điểm bất thường (anomaly score) + vẽ heatmap
       vùng model cho là bất thường -> hiển thị ("Ket qua PatchCore")

Khác với bản dùng MobileNet classifier + Grad-CAM (chỉ chạy trên 1 vùng
ROI con), bản này chạy PatchCore trên TOÀN BỘ ảnh đã crop/xoay.
"""

import sys
import os
import time
import numpy as np
import cv2
import torch

# ============================================================
# 1. Trỏ tới thư mục chứa DLL runtime của MVS
# ============================================================
dll_dir = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
os.add_dll_directory(dll_dir)

# ============================================================
# 2. Trỏ THẲNG vào thư mục MvImport (nằm cùng cấp với file này)
# ============================================================
mvimport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
sys.path.append(mvimport_path)

# ============================================================
# 3. Import kiểu module thường (không có tiền tố MvImport.)
# ============================================================
from MvCameraControl_class import *

# ============================================================
# 4. Import cho phần PatchCore (anomalib)
# ============================================================
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
THU_MUC_ANH_GOC = r"D:\TongHop\RTC Technologi\PCB\captured_raw"

# Ảnh CHUẨN CỐ ĐỊNH để xác định hướng trái/phải khi crop vật thể chính.
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\crop5\NG\Image_20260723100841117_roi.png"

# --- Cấu hình PatchCore ---
# Trỏ tới file .ckpt được tạo ra sau khi chạy script train_anomalib_patchcore.py
# (thường nằm trong <KET_QUA_DIR>\pcb\Patchcore\<version>\weights\lightning\model.ckpt)
CHECKPOINT_PATH = r"D:\TongHop\RTC Technologi\PCB\crop5\Patchcore\pcb\v4\weights\lightning\model.ckpt"

HEATMAP_ALPHA = 0.45

# Ngưỡng phân loại OK/NG dựa trên anomaly score (điều chỉnh theo kết quả
# đánh giá thực tế trên tập test khi train). Để None nếu chỉ muốn xem
# điểm số, không cần phân loại nhị phân.
NGUONG_ANOMALY_SCORE = None

# Thư mục tạm để lưu ảnh crop trước khi đưa vào engine.predict() (anomalib
# yêu cầu 1 đường dẫn file/thư mục, không nhận thẳng mảng numpy trong bộ
# nhớ ở API predict cấp cao này).
THU_MUC_TAM_PREDICT = r"D:\TongHop\RTC Technologi\PCB\_tmp_predict"
# =====================================================================

SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000
MIN_INLIERS_HUONG = 15
PAD_MASK_SIFT = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Từ PyTorch 2.6, torch.load mặc định weights_only=True nên sẽ chặn 1 số
# class nội bộ của anomalib (vd anomalib.PrecisionType) có trong checkpoint.
# Vì đây là checkpoint do chính mình train ra (nguồn tin cậy), ép
# weights_only=False cho toàn bộ chương trình (engine.predict() cũng gọi
# torch.load nội bộ khi nạp checkpoint nên cần patch từ đây, không chỉ ở
# chỗ load model).
_orig_torch_load = torch.load


def _torch_load_khong_gioi_han(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_khong_gioi_han


# ====================================================================
# CÁC HÀM XỬ LÝ ẢNH - CROP VẬT THỂ CHÍNH (giữ nguyên từ script gốc)
# ====================================================================

def tim_contour_tu_dong(img, target_w=1200):
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
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k(41), k(41)), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return (contour.astype(np.float32) / scale).astype(np.int32)


def tinh_m_align_va_roi(contour):
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

    return M_align, (x, y, w, h)


def trich_sift_anh(img, target_w=SIFT_TARGET_WIDTH, mask_full=None, nfeatures=SIFT_NFEATURES):
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
    for kpt in kp:
        kpt.pt = (kpt.pt[0] / scale, kpt.pt[1] / scale)
    return kp, des


def uoc_luong_M_sift(kp1, des1, kp2, des2, min_inliers=MIN_INLIERS_HUONG):
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


def chuan_bi_anh_chuan(duong_dan_anh_chuan):
    img_chuan = cv2.imread(duong_dan_anh_chuan)
    if img_chuan is None:
        raise FileNotFoundError(f"Không đọc được ảnh chuẩn cố định: {duong_dan_anh_chuan}")

    contour_chuan = tim_contour_tu_dong(img_chuan)
    if contour_chuan is None:
        raise RuntimeError(f"Không tìm thấy vật thể trong ảnh chuẩn: {duong_dan_anh_chuan}")

    m_align_chuan, roi_chuan = tinh_m_align_va_roi(contour_chuan)
    canonical_w, canonical_h = roi_chuan[2], roi_chuan[3]

    xg, yg, wg, hg = cv2.boundingRect(contour_chuan)
    mask_chuan = np.zeros(img_chuan.shape[:2], np.uint8)
    mask_chuan[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
               max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
    kp_chuan, des_chuan = trich_sift_anh(img_chuan, mask_full=mask_chuan)

    R_chuan_inv = cv2.invertAffineTransform(m_align_chuan)[:, :2]
    dir_chuan = R_chuan_inv @ np.array([1.0, 0.0])

    return {
        "canonical_w": canonical_w,
        "canonical_h": canonical_h,
        "kp_chuan": kp_chuan,
        "des_chuan": des_chuan,
        "dir_chuan": dir_chuan,
    }


def crop_va_dong_bo_huong(img, anh_chuan_info):
    contour = tim_contour_tu_dong(img)
    if contour is None:
        return None, "Không tìm thấy vật thể trong ảnh vừa chụp."

    m_align, roi = tinh_m_align_va_roi(contour)
    x, y, w, h = roi

    warped = cv2.warpAffine(img, m_align, (img.shape[1], img.shape[0]))
    crop = warped[y:y + h, x:x + w]

    canonical_w = anh_chuan_info["canonical_w"]
    canonical_h = anh_chuan_info["canonical_h"]
    if crop.shape[1] != canonical_w or crop.shape[0] != canonical_h:
        crop = cv2.resize(crop, (canonical_w, canonical_h))

    canh_bao = None
    xg, yg, wg, hg = cv2.boundingRect(contour)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
         max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
    kp, des = trich_sift_anh(img, mask_full=mask)

    M, so_inlier = uoc_luong_M_sift(kp, des, anh_chuan_info["kp_chuan"], anh_chuan_info["des_chuan"])
    if M is None:
        canh_bao = f"Không đủ tin cậy để xác định hướng ({so_inlier} match) - giữ nguyên mặc định."
    else:
        R_inv = cv2.invertAffineTransform(m_align)[:, :2]
        dir_nay = R_inv @ np.array([1.0, 0.0])
        R_sift = M[:, :2]
        dir_nay_trong_he_chuan = R_sift @ dir_nay

        goc_do = np.degrees(np.arccos(
            np.clip(np.dot(dir_nay_trong_he_chuan, anh_chuan_info["dir_chuan"]) /
                    (np.linalg.norm(dir_nay_trong_he_chuan) * np.linalg.norm(anh_chuan_info["dir_chuan"]) + 1e-9),
                    -1, 1)
        ))
        if goc_do > 90:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
            canh_bao = f"Đã lật 180° (lệch góc {goc_do:.0f}°, {so_inlier} inlier)."
        else:
            canh_bao = f"Giữ nguyên hướng (lệch góc {goc_do:.0f}°, {so_inlier} inlier)."

    return crop, canh_bao


def resize_de_hien_thi(img, max_dim=900):
    h, w = img.shape[:2]
    ti_le = min(1.0, max_dim / max(h, w))
    if ti_le >= 1.0:
        return img
    return cv2.resize(img, (int(w * ti_le), int(h * ti_le)))


# ====================================================================
# LOAD MODEL PATCHCORE (anomalib) + HÀM DỰ ĐOÁN
# ====================================================================

def chuan_bi_patchcore(checkpoint_path):
    """Chuẩn bị model (kiến trúc rỗng, CHƯA nạp trọng số) + Engine dùng
    chung cho toàn bộ phiên chạy. Trọng số sẽ được chính engine.predict()
    nạp từ checkpoint_path ở MỖI lần dự đoán - đây là cách dùng CHÍNH THỨC
    được anomalib khuyến nghị, đảm bảo pre_processor/post_processor (resize,
    center-crop, normalize, threshold hoá score...) được áp dụng ĐÚNG HỆT
    như lúc train. (Trước đó mình tự viết transform + gọi thẳng model(...)
    khiến ảnh bị tiền xử lý sai cách -> score luôn ra 1.0 dù ảnh OK.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint PatchCore tại: {checkpoint_path}")

    model = Patchcore()
    engine = Engine()
    print(f"Đã chuẩn bị xong Engine + model PatchCore (checkpoint: {checkpoint_path})\n")
    return model, engine


def du_doan_patchcore(crop_bgr, model, engine, checkpoint_path, thu_muc_tam):
    """Nhận ảnh đã crop/xoay (BGR, chưa cắt nhỏ thêm) -> trả về
    (anomaly_score, anomaly_map [H,W trong khoảng 0..1], pred_label)."""
    os.makedirs(thu_muc_tam, exist_ok=True)
    duong_dan_tam = os.path.join(thu_muc_tam, "_anh_cho_predict.png")
    cv2.imwrite(duong_dan_tam, crop_bgr)

    dataset = PredictDataset(path=duong_dan_tam)
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=checkpoint_path)

    if not predictions:
        raise RuntimeError("engine.predict() không trả về kết quả nào.")

    pred = predictions[0]

    pred_score = pred.pred_score
    anomaly_map = pred.anomaly_map
    pred_label = getattr(pred, "pred_label", None)

    score = float(pred_score.squeeze().cpu().item()) if hasattr(pred_score, "cpu") else float(pred_score)
    amap = anomaly_map.squeeze().cpu().numpy() if hasattr(anomaly_map, "cpu") else np.asarray(anomaly_map).squeeze()

    if pred_label is not None:
        pred_label = bool(pred_label.squeeze().cpu().item()) if hasattr(pred_label, "cpu") else bool(pred_label)

    # Chỉ chuẩn hoá amap về 0..1 ĐỂ HIỂN THỊ heatmap cho trực quan; giá trị
    # score dùng để quyết định OK/NG vẫn lấy từ pred.pred_score (đã được
    # post_processor của model tính đúng, KHÔNG tự ý chuẩn hoá lại).
    amap_hien_thi = amap.copy()
    if amap_hien_thi.max() > amap_hien_thi.min():
        amap_hien_thi = (amap_hien_thi - amap_hien_thi.min()) / (amap_hien_thi.max() - amap_hien_thi.min())

    return score, amap_hien_thi, pred_label


def overlay_heatmap(img_bgr, amap, alpha=HEATMAP_ALPHA):
    H, W = img_bgr.shape[:2]
    amap_resized = cv2.resize(amap, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * amap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return overlay


# ====================================================================
# PHẦN CAMERA: chụp ảnh liên tục, nhấn 'c' để chụp + crop + PatchCore
# ====================================================================

def main():
    os.makedirs(THU_MUC_ANH_GOC, exist_ok=True)

    print("Đang chuẩn bị ảnh chuẩn để xác định hướng...")
    anh_chuan_info = chuan_bi_anh_chuan(DUONG_DAN_ANH_CHUAN_CO_DINH)
    print(f"Kích thước ROI chuẩn: {anh_chuan_info['canonical_w']}x{anh_chuan_info['canonical_h']}\n")

    print("Đang chuẩn bị model + engine PatchCore...")
    model, engine = chuan_bi_patchcore(CHECKPOINT_PATH)

    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("SDK Version:", hex(SDKVersion))

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Không tìm thấy camera nào! Mã lỗi:", ret)
        sys.exit()

    print(f"Tìm thấy {deviceList.nDeviceNum} camera")

    target_index = 0
    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            print(f"[{i}] GigE Camera: {strModeName}")
            if "MV-CS200" in strModeName:
                target_index = i
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            strModeName = "".join(chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            print(f"[{i}] USB Camera: {strModeName}")

    stDeviceInfo = cast(deviceList.pDeviceInfo[target_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print("Tạo handle lỗi:", ret)
        sys.exit()

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Mở camera lỗi:", ret)
        cam.MV_CC_DestroyHandle()
        sys.exit()

    if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing lỗi:", ret)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * nPayloadSize)()

    print("\nĐang xem trực tiếp... Nhấn 'c' để chụp + crop + PatchCore, 'q' để thoát.")

    so_thu_tu = 0

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                print("Không lấy được frame, mã lỗi:", ret)
                continue

            raw = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
            raw = raw.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            pixel_type = stFrameInfo.enPixelType

            if pixel_type == PixelType_Gvsp_BayerRG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
            elif pixel_type == PixelType_Gvsp_BayerGB8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)
            elif pixel_type == PixelType_Gvsp_BayerGR8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GR2BGR)
            elif pixel_type == PixelType_Gvsp_BayerBG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)
            elif pixel_type == PixelType_Gvsp_Mono8:
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                print("Pixel format chưa được xử lý:", pixel_type)
                continue

            img_show = cv2.resize(img, (960, 640))
            cv2.imshow("Hikrobot Camera - nhan 'c' de chup, 'q' de thoat", img_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                so_thu_tu += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                ten_file = f"capture_{timestamp}_{so_thu_tu:03d}"

                duong_dan_goc = os.path.join(THU_MUC_ANH_GOC, f"{ten_file}.png")
                cv2.imwrite(duong_dan_goc, img)
                print(f"\n[Chụp #{so_thu_tu}] Đã lưu ảnh gốc: {duong_dan_goc}")

                crop, thong_bao = crop_va_dong_bo_huong(img, anh_chuan_info)
                if crop is None:
                    print(f"  -> KHÔNG crop được: {thong_bao}")
                    continue

                print(f"  -> {thong_bao}")
                cv2.imshow("Anh da crop", resize_de_hien_thi(crop))

                # --- Đưa THẲNG ảnh đã crop vào PatchCore (không cắt nhỏ thêm) ---
                score, amap, pred_label = du_doan_patchcore(
                    crop, model, engine, CHECKPOINT_PATH, THU_MUC_TAM_PREDICT)

                if NGUONG_ANOMALY_SCORE is not None:
                    nhan = "NG" if score >= NGUONG_ANOMALY_SCORE else "OK"
                    print(f"  -> Anomaly score: {score:.4f}  =>  {nhan} "
                          f"(nguong tu dat={NGUONG_ANOMALY_SCORE})")
                elif pred_label is not None:
                    nhan = "NG" if pred_label else "OK"
                    print(f"  -> Anomaly score: {score:.4f}  =>  {nhan} "
                          f"(theo nguong model tu hoc khi train)")
                else:
                    nhan = ""
                    print(f"  -> Anomaly score: {score:.4f}")

                overlay = overlay_heatmap(crop, amap)
                label = f"score={score:.4f}" + (f" ({nhan})" if nhan else "")
                cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)
                cv2.imshow("Ket qua PatchCore", resize_de_hien_thi(overlay, max_dim=800))
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("\nĐã đóng camera.")


if __name__ == "__main__":
    main()