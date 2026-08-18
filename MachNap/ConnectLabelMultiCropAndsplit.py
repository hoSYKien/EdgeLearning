r"""
06b_ve_box_tao_dataset.py
-------------------------
Gán nhãn dataset theo kiểu "vẽ khoanh vùng lỗi" - dễ cho người vận hành.
VẼ XONG BẤM ENTER LÀ ẢNH ĐƯỢC CẮT VÀ CHIA VÀO DATASET LUÔN.

========================= CÁCH CHIA PART =========================
Ý tưởng: chia ảnh thành N PHẦN CHÍNH dọc theo một chiều, rồi thêm N-1
PHẦN GỐI nằm ĐÚNG GIỮA hai phần chính cạnh nhau. Phần gối để cứu những
lỗi nhỏ nằm ngay vết cắt (bị xẻ đôi ở phần chính thì vẫn nguyên vẹn ở
phần gối).

    N = 3  ->  5 part : part1, part2, part3, part12, part23
    N = 4  ->  7 part : part1..part4, part12, part23, part34
    N = 2  ->  3 part : part1, part2, part12
    Tổng quát: 2N - 1 part.

CHE_DO_CHIA = "auto"    -> script tự nhìn tỉ lệ dài/rộng của ảnh gốc để
                           quyết định CẮT THEO CHIỀU NÀO (luôn cắt dọc
                           theo cạnh dài) và CHIA LÀM MẤY PHẦN, sao cho
                           mỗi phần gần vuông (đỡ méo khi resize 224x224).

CHE_DO_CHIA = "manual"  -> tự khai báo 2 giá trị:
                           CHIA_THEO      = "ngang" hoặc "doc"
                           SO_PHAN_CHINH  = 2, 3, 4, ...

    CHIA_THEO = "ngang": các part xếp cạnh nhau TRÁI -> PHẢI
                         (vết cắt là đường THẲNG ĐỨNG, cắt theo chiều rộng)
    CHIA_THEO = "doc"  : các part xếp TRÊN -> DƯỚI
                         (vết cắt là đường NẰM NGANG, cắt theo chiều cao)

Cấu hình chia được ghi ra FILE_CAU_HINH để script chạy thật (inference)
đọc lại đúng y hệt, không sợ lệch vùng so với lúc train.
==================================================================

2 CHẾ ĐỘ NGUỒN ẢNH:

    NGUON = "doc_file"  -> Duyệt ảnh CROP FULL có sẵn trong THU_MUC_ANH_FULL.
                           Hiện full ảnh, người vận hành CHỈ VIỆC VẼ ROI.

    NGUON = "chup_anh"  -> Mở camera Hikrobot, nhấn 'c' để chụp. Ảnh chụp
                           chạy qua đủ các bước: tìm contour -> làm thẳng
                           -> ép USB về bên PHẢI -> cắt ROI có padding ->
                           ÁP MASK HÌNH HỌC (nền đen) -> cắt sát mask.
                           Xong mới hiện lên cho vẽ ROI.

LUỒNG CHẠY:
    vẽ ROI -> Enter -> tự tính OK/NG cho từng part -> tự cắt part -> tự bỏ
    vào <THU_MUC_DATASET>\PART..\train|val\OK|NG theo tỉ lệ TI_LE_TRAIN.

LOGIC GÁN NHÃN:
    Với MỖI part và MỖI box:
        ti_le = (diện tích phần box nằm trong part) / (diện tích cả box)
    Part nào có ít nhất 1 box đạt ti_le >= TI_LE_BOX_TOI_THIEU -> NG.
    Ngưỡng xét ĐỘC LẬP từng part: box bị chia đôi thì CẢ HAI part đều NG.

PHÍM TẮT khi vẽ:
    Kéo chuột trái : vẽ 1 box lỗi
    Chuột phải     : xoá box nằm dưới con trỏ
    u              : undo box vừa vẽ
    c              : xoá hết box của ảnh này
    Enter / Space  : LƯU + CẮT + CHIA VÀO DATASET, sang ảnh kế
    n / b          : sang ảnh kế / quay lại (chỉ chế độ đọc file)
    x              : BỎ QUA ảnh này (nếu đã xuất thì gỡ khỏi dataset)
    r              : XUẤT LẠI TOÀN BỘ (dùng khi đổi cách chia hoặc ngưỡng)
    q / ESC        : lưu rồi thoát
"""

import os
import sys
import json
import time
import shutil

import cv2
import numpy as np

# ====================== CHỈNH Ở ĐÂY ======================
NGUON = "chup_anh"        # "doc_file" hoặc "chup_anh"

# ===== CÁCH CHIA PART =====
CHE_DO_CHIA = "auto"      # "auto" hoặc "manual"

# --- chỉ dùng khi CHE_DO_CHIA = "manual" ---
CHIA_THEO = "ngang"       # "ngang" (cắt theo chiều rộng) | "doc" (theo chiều cao)
SO_PHAN_CHINH = 3         # 3 -> 5 part (1, 12, 2, 23, 3)

# --- chỉ dùng khi CHE_DO_CHIA = "auto" ---
# Mỗi phần chính sẽ dài khoảng (AUTO_TI_LE_MUC_TIEU x cạnh ngắn của ảnh).
#   0.6  -> phần hơi dẹt theo cạnh ngắn, chia được nhiều phần (mặc định)
#   1.0  -> mỗi phần gần VUÔNG, chia ít phần hơn
# Giảm số này = part nhỏ hơn, nhiều hơn, lỗi bé chiếm nhiều pixel hơn khi
# resize về 224 nhưng tốn thêm model.
AUTO_TI_LE_MUC_TIEU = 0.6
AUTO_SO_PHAN_MIN = 2
AUTO_SO_PHAN_MAX = 8

TAO_PART_GOI = True       # False = chỉ có N phần chính, không có part gối

# Thư mục ảnh CROP FULL (chưa chia part).
#   - "doc_file": script đọc ảnh từ đây
#   - "chup_anh": script LƯU ảnh vừa crop vào đây
THU_MUC_ANH_FULL = r"D:\TongHop\RTC Technologi\PCB\crop_full"

FILE_NHAN = r"D:\TongHop\RTC Technologi\PCB\nhan_box.json"

# ===== DATASET =====
# Thư mục gốc. Mỗi part tự có thư mục con: <goc>\PART1\train\OK, ...\train\NG,
# ...\val\OK, ...\val\NG. Part gối đặt tên PART12, PART23...
THU_MUC_DATASET = r"D:\TongHop\RTC Technologi\PCB\dataset15"

# Muốn part nào nằm ở ổ/thư mục khác thì khai báo đè ở đây, ví dụ:
#   THU_MUC_PART_RIENG = {"part1": r"E:\du_lieu\PART1"}
THU_MUC_PART_RIENG = {}

# File ghi lại cách chia (để script chạy thật đọc đúng vùng như lúc train).
FILE_CAU_HINH = os.path.join(THU_MUC_DATASET, "cau_hinh_part.json")

TI_LE_TRAIN = 0.8         # 0.8 = 80% train / 20% val

# >>> Tỉ lệ tối thiểu của box nằm trong part thì part đó mới tính NG <<<
# 0.0 -> dính tí là NG ; 0.15 -> dưới 15% coi như không đáng kể (OK)
TI_LE_BOX_TOI_THIEU = 0.15

# Chốt an toàn: box vụn đến mức KHÔNG part nào đạt ngưỡng -> ép part chứa
# phần lớn nhất thành NG để lỗi không biến mất khỏi dataset.
KHONG_DE_BOX_BIEN_MAT = True

XUAT_LAI_TOAN_BO = False  # True = xoá dataset rồi dựng lại từ JSON, xong thoát

MAX_HIEN_THI_W = 1500
MAX_HIEN_THI_H = 780
BOX_TOI_THIEU_PX = 8

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

# ============ CHỈ DÙNG CHO CHẾ ĐỘ "chup_anh" ============
DLL_DIR_MVS = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\crop5\train\OK\Image_20260723101631285_roi.png"

KHUNG_CHUAN_W = 3187
KHUNG_CHUAN_H = 1777

PAD_ROI_NGANG = 300
PAD_ROI_DOC = 60

NGUONG_TI_LE_LOM = 2.0
AP_MASK_HINH_HOC = True

BOARD_X1 = 0.0948
BOARD_X2 = 0.9049
BOARD_Y1 = 0.0360
BOARD_Y2 = 0.9657
BOARD_BO_GOC = 0.0141

USB_X1 = 0.8472
USB_X2 = 0.9576
USB_Y1 = 0.2746
USB_Y2 = 0.6989
USB_BO_GOC = 0.0113

KHUYET_X2 = 0.1403
KHUYET_Y1 = 0.1435
KHUYET_Y2 = 0.8694
KHUYET_BO_GOC = 0.0535

NOI_MASK = 0
LAM_MUOT_BIEN = 5
CAT_SAT_MASK = True
LE_TRONG = 15

SIFT_TARGET_WIDTH = 1400
SIFT_NFEATURES = 3000
MIN_INLIERS_HUONG = 15
PAD_MASK_SIFT = 60
# =========================================================

CAC_TAP = ("train", "val")
CAC_LOP = ("OK", "NG")

# Cấu hình chia hiện hành - do thiet_lap_chia() điền vào.
CHIEU = None      # "ngang" | "doc"
SO_PHAN = None    # số phần chính
RATIOS = []       # [(ten_part, start, end), ...] start/end là % cạnh bị cắt


# ====================================================================
# TÍNH CÁCH CHIA PART TỪ KÍCH THƯỚC ẢNH GỐC
# ====================================================================

def tinh_ratios(so_phan, tao_part_goi=True):
    """N phần chính + (N-1) phần gối nằm giữa. N=3 -> part1,2,3,12,23."""
    ratios = [(f"part{i + 1}", i / so_phan, (i + 1) / so_phan)
              for i in range(so_phan)]
    if tao_part_goi:
        for i in range(so_phan - 1):
            ratios.append((f"part{i + 1}{i + 2}",
                           (i + 0.5) / so_phan, (i + 1.5) / so_phan))
    return ratios


def tu_chon_cach_chia(w, h):
    """Chế độ auto: cắt dọc theo CẠNH DÀI, số phần tính sao cho mỗi phần dài
    khoảng AUTO_TI_LE_MUC_TIEU x cạnh ngắn."""
    chieu = "ngang" if w >= h else "doc"
    canh_dai, canh_ngan = (w, h) if w >= h else (h, w)
    so_phan = int(round(canh_dai / (AUTO_TI_LE_MUC_TIEU * canh_ngan)))
    so_phan = max(AUTO_SO_PHAN_MIN, min(AUTO_SO_PHAN_MAX, so_phan))
    return chieu, so_phan


def thiet_lap_chia(w, h):
    """Điền CHIEU / SO_PHAN / RATIOS. Gọi 1 lần trước khi gán nhãn."""
    global CHIEU, SO_PHAN, RATIOS
    if CHE_DO_CHIA == "auto":
        CHIEU, SO_PHAN = tu_chon_cach_chia(w, h)
        nguon_cf = "auto"
    elif CHE_DO_CHIA == "manual":
        if CHIA_THEO not in ("ngang", "doc"):
            raise ValueError('CHIA_THEO phải là "ngang" hoặc "doc"')
        if SO_PHAN_CHINH < 2:
            raise ValueError("SO_PHAN_CHINH phải >= 2")
        CHIEU, SO_PHAN = CHIA_THEO, SO_PHAN_CHINH
        nguon_cf = "manual"
    else:
        raise ValueError('CHE_DO_CHIA phải là "auto" hoặc "manual"')

    RATIOS = tinh_ratios(SO_PHAN, TAO_PART_GOI)
    canh = "chiều rộng (cắt thẳng đứng)" if CHIEU == "ngang" \
        else "chiều cao (cắt nằm ngang)"
    print(f"Ảnh gốc {w}x{h} (tỉ lệ {max(w, h) / min(w, h):.2f}) -> chia theo "
          f"{canh}, {SO_PHAN} phần chính"
          f"{' + ' + str(SO_PHAN - 1) + ' phần gối' if TAO_PART_GOI else ''}"
          f" = {len(RATIOS)} part  [{nguon_cf}]")
    for ten, s, e in RATIOS:
        v = vung_part(w, h, s, e)
        print(f"   {ten:8s}  {s:.3f} -> {e:.3f}   "
              f"({v[2] - v[0]}x{v[3] - v[1]} px)")
    return RATIOS


def luu_cau_hinh():
    """Ghi cách chia ra file để script chạy thật dùng lại đúng vùng."""
    os.makedirs(os.path.dirname(FILE_CAU_HINH) or ".", exist_ok=True)
    with open(FILE_CAU_HINH, "w", encoding="utf-8") as f:
        json.dump({"chieu": CHIEU, "so_phan": SO_PHAN,
                   "tao_part_goi": TAO_PART_GOI,
                   "ti_le_box_toi_thieu": TI_LE_BOX_TOI_THIEU,
                   "ti_le_train": TI_LE_TRAIN,
                   "ratios": [[t, s, e] for t, s, e in RATIOS]},
                  f, ensure_ascii=False, indent=1)


def canh_bao_doi_cach_chia():
    """Nếu dataset cũ được chia kiểu khác thì báo để người dùng bấm 'r'."""
    if not os.path.isfile(FILE_CAU_HINH):
        return
    try:
        with open(FILE_CAU_HINH, "r", encoding="utf-8") as f:
            cu = json.load(f)
    except Exception:
        return
    if cu.get("chieu") != CHIEU or cu.get("so_phan") != SO_PHAN \
            or cu.get("tao_part_goi") != TAO_PART_GOI:
        print("\n*** CẢNH BÁO: dataset hiện có đang chia theo kiểu "
              f"{cu.get('chieu')} / {cu.get('so_phan')} phần, khác với cấu hình "
              f"bây giờ ({CHIEU} / {SO_PHAN} phần).")
        print("    Nên bấm 'r' (hoặc đặt XUAT_LAI_TOAN_BO = True) để dựng lại "
              "dataset từ đầu.\n")


def vung_part(w, h, start, end):
    """Trả về (x0, y0, x1, y1) của part trên ảnh w x h."""
    if CHIEU == "doc":
        y0 = max(0, min(int(round(h * start)), h))
        y1 = max(0, min(int(round(h * end)), h))
        return 0, y0, w, y1
    x0 = max(0, min(int(round(w * start)), w))
    x1 = max(0, min(int(round(w * end)), w))
    return x0, 0, x1, h


def thu_muc_goc_part(ten_part):
    if ten_part in THU_MUC_PART_RIENG:
        return THU_MUC_PART_RIENG[ten_part]
    return os.path.join(THU_MUC_DATASET, ten_part.upper())


# ====================================================================
# TÍNH NHÃN TỪ BOX
# ====================================================================

def ti_le_box_trong_vung(box, vung):
    """Diện tích phần box nằm trong vùng, chia cho diện tích cả box."""
    bx1, by1, bx2, by2 = box
    x0, y0, x1, y1 = vung
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return 0.0
    iw = max(0, min(bx2, x1) - max(bx1, x0))
    ih = max(0, min(by2, y1) - max(by1, y0))
    return (iw * ih) / float(bw * bh)


def nhan_cac_part(boxes, w, h, nguong=None):
    """Trả về list [(ten_part, vung, 'OK'/'NG', ti_le_lon_nhat), ...]

    Ngưỡng xét ĐỘC LẬP cho từng part và từng box: một box bị chia đôi thì CẢ
    HAI part chứa nó đều NG (mỗi bên 50% >= ngưỡng)."""
    if nguong is None:
        nguong = TI_LE_BOX_TOI_THIEU

    vung = {ten: vung_part(w, h, s, e) for ten, s, e in RATIOS}
    nhan = {ten: "OK" for ten in vung}
    tl_max = {ten: 0.0 for ten in vung}

    for b in boxes:
        tl_theo_part = {}
        for ten, v in vung.items():
            tl = ti_le_box_trong_vung(b, v)
            tl_theo_part[ten] = tl
            tl_max[ten] = max(tl_max[ten], tl)
            if tl > 0 and tl >= nguong:
                nhan[ten] = "NG"

        if KHONG_DE_BOX_BIEN_MAT and not any(
                t >= nguong and t > 0 for t in tl_theo_part.values()):
            ten_lon_nhat = max(tl_theo_part, key=tl_theo_part.get)
            if tl_theo_part[ten_lon_nhat] > 0:
                nhan[ten_lon_nhat] = "NG"

    return [(ten, vung[ten], nhan[ten], tl_max[ten]) for ten, _, _ in RATIOS]


# ====================================================================
# LƯU / ĐỌC FILE NHÃN
# ====================================================================

def doc_nhan(duong_dan):
    if os.path.isfile(duong_dan):
        with open(duong_dan, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def ghi_nhan(duong_dan, du_lieu):
    thu_muc = os.path.dirname(duong_dan)
    if thu_muc:
        os.makedirs(thu_muc, exist_ok=True)
    with open(duong_dan, "w", encoding="utf-8") as f:
        json.dump(du_lieu, f, ensure_ascii=False, indent=1)


def liet_ke_anh(thu_muc):
    if not os.path.isdir(thu_muc):
        raise FileNotFoundError(f"Không có thư mục ảnh: {thu_muc}")
    return sorted(f for f in os.listdir(thu_muc)
                  if f.lower().endswith(VALID_EXT)
                  and os.path.isfile(os.path.join(thu_muc, f)))


# ====================================================================
# CHIA ẢNH VÀO DATASET NGAY SAU KHI GÁN NHÃN
# ====================================================================

def thu_muc_dich(ten_part, tap, lop):
    return os.path.join(thu_muc_goc_part(ten_part), tap, lop)


def tao_san_thu_muc():
    for ten_part, _, _ in RATIOS:
        for tap in CAC_TAP:
            for lop in CAC_LOP:
                os.makedirs(thu_muc_dich(ten_part, tap, lop), exist_ok=True)


def dem_hien_co():
    """Đếm ảnh đang có trong từng ô để giữ đúng tỉ lệ train/val ở phiên sau."""
    dem = {}
    for ten_part, _, _ in RATIOS:
        for tap in CAC_TAP:
            for lop in CAC_LOP:
                d = thu_muc_dich(ten_part, tap, lop)
                dem[(ten_part, tap, lop)] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return dem


def chon_tap(dem, ten_part, lop):
    """Chọn train hay val để tỉ lệ luôn bám sát TI_LE_TRAIN.
    Với 0.8: ảnh thứ 1-4 vào train, ảnh thứ 5 vào val, rồi lặp lại."""
    n_train = dem.get((ten_part, "train", lop), 0)
    n_val = dem.get((ten_part, "val", lop), 0)
    return "train" if n_train < (n_train + n_val + 1) * TI_LE_TRAIN else "val"


def _xoa_ban_cu(gc, dem):
    """Xoá các file part đã xuất trước đó của ảnh này (khi sửa nhãn)."""
    for ten_part, thong_tin in (gc.get("xuat") or {}).items():
        tap, lop, duong_dan = thong_tin
        if os.path.isfile(duong_dan):
            os.remove(duong_dan)
        khoa = (ten_part, tap, lop)
        if khoa in dem:
            dem[khoa] = max(0, dem[khoa] - 1)
    gc["xuat"] = {}


def xuat_mot_anh(img, ten_file, nhan_all, dem):
    """Cắt các part của 1 ảnh và bỏ vào train/val - OK/NG."""
    gc = nhan_all[ten_file]
    _xoa_ban_cu(gc, dem)

    if gc.get("bo_qua"):
        ghi_nhan(FILE_NHAN, nhan_all)
        return "BO QUA - da go khoi dataset"

    img_h, img_w = img.shape[:2]
    goc = os.path.splitext(ten_file)[0]
    tom_tat = []
    for ten_part, (x0, y0, x1, y1), nhan, tl in nhan_cac_part(
            gc.get("boxes", []), img_w, img_h):
        if x1 <= x0 or y1 <= y0:
            continue
        tap = chon_tap(dem, ten_part, nhan)
        d = thu_muc_dich(ten_part, tap, nhan)
        os.makedirs(d, exist_ok=True)
        duong_dan = os.path.join(d, f"{goc}_{ten_part}.png")
        cv2.imwrite(duong_dan, img[y0:y1, x0:x1])
        dem[(ten_part, tap, nhan)] = dem.get((ten_part, tap, nhan), 0) + 1
        gc.setdefault("xuat", {})[ten_part] = [tap, nhan, duong_dan]
        tom_tat.append(f"{ten_part}:{nhan}/{tap}")

    ghi_nhan(FILE_NHAN, nhan_all)
    return " | ".join(tom_tat)


def in_thong_ke(dem):
    print("-" * 64)
    for ten_part, _, _ in RATIOS:
        o = {(t, l): dem.get((ten_part, t, l), 0) for t in CAC_TAP for l in CAC_LOP}
        print(f"{ten_part:8s}: tong {sum(o.values()):4d} | train OK {o[('train','OK')]:4d} "
              f"NG {o[('train','NG')]:4d} | val OK {o[('val','OK')]:4d} "
              f"NG {o[('val','NG')]:4d}")
    print("-" * 64)


def xuat_lai_toan_bo(nhan_all):
    """Xoá sạch các thư mục part rồi dựng lại từ JSON. Dùng khi đổi cách chia,
    đổi TI_LE_BOX_TOI_THIEU hoặc TI_LE_TRAIN."""
    print("\nĐang xoá dataset cũ và xuất lại từ đầu...")
    da_xoa = set()
    for ten_part, _, _ in RATIOS:
        d = thu_muc_goc_part(ten_part)
        if d not in da_xoa and os.path.isdir(d):
            shutil.rmtree(d)
            da_xoa.add(d)
    tao_san_thu_muc()
    dem = dem_hien_co()

    for gc in nhan_all.values():
        gc["xuat"] = {}

    so = 0
    for ten_file in sorted(nhan_all):
        duong_dan = os.path.join(THU_MUC_ANH_FULL, ten_file)
        if not os.path.isfile(duong_dan):
            continue
        img = cv2.imread(duong_dan)
        if img is None:
            print(f"  Không đọc được: {ten_file}")
            continue
        xuat_mot_anh(img, ten_file, nhan_all, dem)
        so += 1
    ghi_nhan(FILE_NHAN, nhan_all)
    luu_cau_hinh()
    print(f"Đã xuất lại {so} ảnh full. Ngưỡng box = {TI_LE_BOX_TOI_THIEU:.2f}, "
          f"train/val = {TI_LE_TRAIN:.0%}/{1 - TI_LE_TRAIN:.0%}")
    in_thong_ke(dem)
    return dem


# ====================================================================
# GIAO DIỆN VẼ BOX
# ====================================================================

TEN_CUA_SO_VE = ("Ve khoanh vung loi - Enter: luu & chia dataset | u: undo | "
                 "c: xoa het | x: bo qua | r: xuat lai | q: thoat")

MAU_NG = (0, 0, 255)
MAU_OK = (0, 200, 0)
MAU_BOX = (0, 165, 255)


class BoAnhVe:
    def __init__(self):
        self.dang_ve = False
        self.diem_dau = (0, 0)
        self.diem_hien_tai = (0, 0)
        self.boxes = []          # toạ độ THEO ẢNH GỐC
        self.scale = 1.0
        self.xoa_yeu_cau = None

    def chuot(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dang_ve = True
            self.diem_dau = (x, y)
            self.diem_hien_tai = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.diem_hien_tai = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dang_ve:
            self.dang_ve = False
            x0, y0 = self.diem_dau
            if abs(x - x0) >= BOX_TOI_THIEU_PX and abs(y - y0) >= BOX_TOI_THIEU_PX:
                bx1, bx2 = sorted((x0, x))
                by1, by2 = sorted((y0, y))
                s = self.scale
                self.boxes.append([int(bx1 / s), int(by1 / s),
                                   int(bx2 / s), int(by2 / s)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.xoa_yeu_cau = (x / self.scale, y / self.scale)


def ve_khung_hien_thi(anh_nho, state, w_goc, h_goc, dong_tieu_de, bo_qua):
    """Vẽ box + ranh giới các part + bảng trạng thái."""
    ra = anh_nho.copy()
    s = state.scale
    H_nho, W_nho = ra.shape[:2]

    kq = nhan_cac_part(state.boxes, w_goc, h_goc)

    # phần chính vẽ ở nửa này, phần gối vẽ ở nửa kia cho đỡ rối
    for i, (ten, (x0, y0, x1, y1), nhan, tl) in enumerate(kq):
        mau = MAU_NG if nhan == "NG" else MAU_OK
        chinh = i < SO_PHAN
        if CHIEU == "ngang":
            X0, X1 = int(x0 * s), max(0, int(x1 * s) - 1)
            y_bat_dau = 0 if chinh else H_nho // 2
            cv2.line(ra, (X0, y_bat_dau), (X0, H_nho), mau, 1)
            cv2.line(ra, (X1, y_bat_dau), (X1, H_nho), mau, 1)
            cv2.putText(ra, f"{ten}:{nhan}", (X0 + 6, 22 if chinh else H_nho - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)
        else:
            Y0, Y1 = int(y0 * s), max(0, int(y1 * s) - 1)
            x_bat_dau = 0 if chinh else W_nho // 2
            cv2.line(ra, (x_bat_dau, Y0), (W_nho, Y0), mau, 1)
            cv2.line(ra, (x_bat_dau, Y1), (W_nho, Y1), mau, 1)
            cv2.putText(ra, f"{ten}:{nhan}",
                        (10 if chinh else W_nho // 2 + 10, Y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)

    for b in state.boxes:
        cv2.rectangle(ra, (int(b[0] * s), int(b[1] * s)),
                      (int(b[2] * s), int(b[3] * s)), MAU_BOX, 2)

    if state.dang_ve:
        cv2.rectangle(ra, state.diem_dau, state.diem_hien_tai, (255, 255, 0), 1)

    if bo_qua:
        cv2.putText(ra, "BO QUA", (W_nho // 2 - 90, H_nho // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    # bảng trạng thái: tự xuống dòng nếu nhiều part
    moi_dong = max(1, W_nho // 230)
    so_dong = (len(kq) + moi_dong - 1) // moi_dong
    bang = np.zeros((34 + 26 * so_dong, W_nho, 3), np.uint8)
    cv2.putText(bang, f"{dong_tieu_de}   |   {len(state.boxes)} box   |   "
                      f"chia {CHIEU} {SO_PHAN} phan   |   nguong "
                      f"{TI_LE_BOX_TOI_THIEU:.2f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    for i, (ten, _, nhan, tl) in enumerate(kq):
        mau = MAU_NG if nhan == "NG" else MAU_OK
        cv2.putText(bang, f"{ten}: {nhan} ({tl * 100:4.0f}%)",
                    (10 + (i % moi_dong) * 230, 48 + (i // moi_dong) * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mau, 2)
    return np.vstack([ra, bang])


def ve_roi_mot_anh(img, ten_file, nhan_all, dem, dong_tieu_de,
                   cho_dieu_huong=True):
    """Hiện 1 ảnh full cho người vận hành vẽ ROI. Enter = lưu nhãn + cắt part
    + chia vào dataset ngay.

    Trả về: "tiep" | "bo" | "truoc" | "xuat_lai" | "thoat"."""
    state = BoAnhVe()
    h_goc, w_goc = img.shape[:2]
    state.scale = min(1.0, MAX_HIEN_THI_W / w_goc, MAX_HIEN_THI_H / h_goc)
    anh_nho = cv2.resize(img, (int(w_goc * state.scale), int(h_goc * state.scale))) \
        if state.scale < 1.0 else img.copy()

    ghi_chu = nhan_all.get(ten_file, {})
    state.boxes = [list(b) for b in ghi_chu.get("boxes", [])]
    bo_qua = bool(ghi_chu.get("bo_qua", False))

    cv2.namedWindow(TEN_CUA_SO_VE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(TEN_CUA_SO_VE, state.chuot)

    def luu_va_chia():
        cu = nhan_all.get(ten_file, {})
        nhan_all[ten_file] = {"w": w_goc, "h": h_goc, "boxes": state.boxes,
                              "bo_qua": bo_qua, "xuat": cu.get("xuat", {})}
        print(f"  {ten_file}  ->  {xuat_mot_anh(img, ten_file, nhan_all, dem)}")

    while True:
        if state.xoa_yeu_cau is not None:
            px, py = state.xoa_yeu_cau
            state.xoa_yeu_cau = None
            for k in range(len(state.boxes) - 1, -1, -1):
                bx1, by1, bx2, by2 = state.boxes[k]
                if bx1 <= px <= bx2 and by1 <= py <= by2:
                    state.boxes.pop(k)
                    break

        cv2.imshow(TEN_CUA_SO_VE,
                   ve_khung_hien_thi(anh_nho, state, w_goc, h_goc,
                                     dong_tieu_de, bo_qua))
        phim = cv2.waitKey(20) & 0xFF

        if phim in (13, 32):                 # Enter / Space
            luu_va_chia()
            return "tiep"
        if phim == ord('u'):
            if state.boxes:
                state.boxes.pop()
        elif phim == ord('c'):
            state.boxes = []
        elif phim == ord('x'):
            bo_qua = not bo_qua
        elif phim == ord('r'):
            luu_va_chia()
            return "xuat_lai"
        elif phim in (ord('q'), 27):
            luu_va_chia()
            return "thoat"
        elif cho_dieu_huong and phim == ord('n'):
            return "bo"
        elif cho_dieu_huong and phim == ord('b'):
            return "truoc"


# ====================================================================
# CHẾ ĐỘ 1: ĐỌC FILE CÓ SẴN
# ====================================================================

def chay_doc_file(nhan_all, dem):
    danh_sach = liet_ke_anh(THU_MUC_ANH_FULL)
    da_gan = sum(1 for f in danh_sach if f in nhan_all)
    print(f"{len(danh_sach)} ảnh trong thư mục, {da_gan} ảnh đã có nhãn từ trước.")
    print("Kéo chuột trái để khoanh lỗi. Enter = lưu + chia vào dataset.\n")

    i = 0
    while 0 <= i < len(danh_sach):
        ten_file = danh_sach[i]
        img = cv2.imread(os.path.join(THU_MUC_ANH_FULL, ten_file))
        if img is None:
            print(f"Không đọc được ảnh, bỏ qua: {ten_file}")
            i += 1
            continue

        hd = ve_roi_mot_anh(img, ten_file, nhan_all, dem,
                            f"[{i + 1}/{len(danh_sach)}] {ten_file}")
        if hd in ("tiep", "bo"):
            i += 1
        elif hd == "truoc":
            i = max(0, i - 1)
        else:
            cv2.destroyAllWindows()
            return hd

    cv2.destroyAllWindows()
    print("Đã duyệt hết ảnh.")
    return "thoat"


# ====================================================================
# XỬ LÝ ẢNH CHỤP: CROP VẬT THỂ + MASK HÌNH HỌC
# (đồng bộ với 01_tao_template_v7.py - chỉ dùng cho chế độ "chup_anh")
# ====================================================================

def tim_contour_tu_dong(img, target_w=1200):
    """Threshold Saturation + lọc thành phần liên thông lớn nhất + lấp lỗ."""
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


def _do_do_lom(contour_aligned):
    """Đo độ lõm ăn vào từ mép trái và mép phải (contour đã xoay thẳng)."""
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
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    M_align = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
    x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    if h > w:
        M_xoay_them = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 90, 1.0)
        M_align = (np.vstack([M_xoay_them, [0, 0, 1]]) @
                   np.vstack([M_align, [0, 0, 1]]))[:2, :]
        contour_aligned = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M_align)
        x, y, w, h = cv2.boundingRect(contour_aligned.astype(np.int32))

    lom_trai, lom_phai = _do_do_lom(contour_aligned.astype(np.int32))
    if lom_trai > lom_phai:
        M_180 = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), 180, 1.0)
        M_align = (np.vstack([M_180, [0, 0, 1]]) @
                   np.vstack([M_align, [0, 0, 1]]))[:2, :]
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
    _chu_nhat_bo_goc(mask, int(BOARD_X1 * W), int(BOARD_Y1 * H),
                     int(BOARD_X2 * W), int(BOARD_Y2 * H), int(BOARD_BO_GOC * H), 255)
    _chu_nhat_bo_goc(mask, int(USB_X1 * W), int(USB_Y1 * H),
                     int(USB_X2 * W), int(USB_Y2 * H), int(USB_BO_GOC * H), 255)
    if KHUYET_X2 > 0:
        _chu_nhat_bo_goc(mask, 0, int(KHUYET_Y1 * H),
                         int(KHUYET_X2 * W), int(KHUYET_Y2 * H),
                         int(KHUYET_BO_GOC * H), 0)
    if NOI_MASK > 0:
        mask = cv2.dilate(mask, np.ones((2 * NOI_MASK + 1,) * 2, np.uint8))
    elif NOI_MASK < 0:
        mask = cv2.erode(mask, np.ones((2 * -NOI_MASK + 1,) * 2, np.uint8))
    if LAM_MUOT_BIEN and LAM_MUOT_BIEN >= 3:
        mask = cv2.medianBlur(mask, LAM_MUOT_BIEN | 1)
    return mask


def kich_thuoc_output_chuan():
    """Kích thước ảnh crop cuối - suy ra từ mask, không cần chụp thử."""
    if not AP_MASK_HINH_HOC or not CAT_SAT_MASK:
        return KHUNG_CHUAN_W, KHUNG_CHUAN_H
    mask = tao_mask_hinh_hoc(KHUNG_CHUAN_W, KHUNG_CHUAN_H)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return (min(KHUNG_CHUAN_W, bx + bw + LE_TRONG) - max(0, bx - LE_TRONG),
            min(KHUNG_CHUAN_H, by + bh + LE_TRONG) - max(0, by - LE_TRONG))


def trich_sift_anh(img, target_w=SIFT_TARGET_WIDTH, mask_full=None,
                   nfeatures=SIFT_NFEATURES):
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
    """Dữ liệu SIFT của ảnh chuẩn - CHỈ dùng cho nhánh dự phòng."""
    img_chuan = cv2.imread(duong_dan_anh_chuan)
    if img_chuan is None:
        raise FileNotFoundError(f"Không đọc được ảnh chuẩn cố định: {duong_dan_anh_chuan}")
    contour_chuan = tim_contour_tu_dong(img_chuan)
    if contour_chuan is None:
        raise RuntimeError(f"Không tìm thấy vật thể trong ảnh chuẩn: {duong_dan_anh_chuan}")
    m_align_chuan, _, ti_le_lom_chuan = tinh_m_align_va_roi(contour_chuan)

    xg, yg, wg, hg = cv2.boundingRect(contour_chuan)
    mask_chuan = np.zeros(img_chuan.shape[:2], np.uint8)
    mask_chuan[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
               max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
    kp_chuan, des_chuan = trich_sift_anh(img_chuan, mask_full=mask_chuan)

    R_chuan_inv = cv2.invertAffineTransform(m_align_chuan)[:, :2]
    dir_chuan = R_chuan_inv @ np.array([1.0, 0.0])
    return {"kp_chuan": kp_chuan, "des_chuan": des_chuan,
            "dir_chuan": dir_chuan, "ti_le_lom": ti_le_lom_chuan}


def crop_va_dong_bo_huong(img, anh_chuan_info):
    """Trả về (ảnh crop đã xóa nền, thông báo) hoặc (None, lý do lỗi)."""
    contour = tim_contour_tu_dong(img)
    if contour is None:
        return None, "Không tìm thấy vật thể trong ảnh vừa chụp."

    m_align, roi, ti_le_lom = tinh_m_align_va_roi(contour)
    x, y, w, h = roi
    warped = cv2.warpAffine(img, m_align, (img.shape[1], img.shape[0]))
    crop = cat_an_toan(warped, x, y, w, h)

    if crop.shape[1] != KHUNG_CHUAN_W or crop.shape[0] != KHUNG_CHUAN_H:
        crop = cv2.resize(crop, (KHUNG_CHUAN_W, KHUNG_CHUAN_H))

    if ti_le_lom > NGUONG_TI_LE_LOM:
        thong_bao = f"Hướng xác định bằng hình học (tỉ lệ lõm {ti_le_lom:.1f})."
    else:
        xg, yg, wg, hg = cv2.boundingRect(contour)
        mask_sift = np.zeros(img.shape[:2], np.uint8)
        mask_sift[max(0, yg - PAD_MASK_SIFT):yg + hg + PAD_MASK_SIFT,
                  max(0, xg - PAD_MASK_SIFT):xg + wg + PAD_MASK_SIFT] = 255
        kp, des = trich_sift_anh(img, mask_full=mask_sift)
        M, so_inlier = uoc_luong_M_sift(kp, des, anh_chuan_info["kp_chuan"],
                                        anh_chuan_info["des_chuan"])
        if M is None:
            thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) và SIFT cũng không đủ "
                         f"tin cậy ({so_inlier} match) - KIỂM TRA BẰNG MẮT.")
        else:
            R_inv = cv2.invertAffineTransform(m_align)[:, :2]
            dir_nay = R_inv @ np.array([1.0, 0.0])
            dir_nay_trong_he_chuan = M[:, :2] @ dir_nay
            goc_do = np.degrees(np.arccos(
                np.clip(np.dot(dir_nay_trong_he_chuan, anh_chuan_info["dir_chuan"]) /
                        (np.linalg.norm(dir_nay_trong_he_chuan) *
                         np.linalg.norm(anh_chuan_info["dir_chuan"]) + 1e-9), -1, 1)))
            if goc_do > 90:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT LẬT 180° "
                             f"(lệch {goc_do:.0f}°, {so_inlier} inlier).")
            else:
                thong_bao = (f"Hình học không chắc ({ti_le_lom:.1f}) -> SIFT giữ nguyên "
                             f"(lệch {goc_do:.0f}°, {so_inlier} inlier).")

    if AP_MASK_HINH_HOC:
        mask = tao_mask_hinh_hoc(crop.shape[1], crop.shape[0])
        crop[mask == 0] = 0
        if CAT_SAT_MASK:
            bx, by, bw, bh = cv2.boundingRect(mask)
            crop = crop[max(0, by - LE_TRONG):min(crop.shape[0], by + bh + LE_TRONG),
                        max(0, bx - LE_TRONG):min(crop.shape[1], bx + bw + LE_TRONG)]

    return crop, thong_bao


# ====================================================================
# CHẾ ĐỘ 2: CHỤP TỪ CAMERA HIKROBOT
# ====================================================================

def chay_chup_anh(nhan_all, dem, anh_chuan_info):
    """Xem trực tiếp -> 'c' chụp -> crop + mask -> vẽ ROI -> chia dataset."""
    os.add_dll_directory(DLL_DIR_MVS)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport"))
    from MvCameraControl_class import (
        MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO, MVCC_INTVALUE,
        MV_FRAME_OUT_INFO_EX, MV_GIGE_DEVICE, MV_USB_DEVICE,
        MV_ACCESS_Exclusive, MV_TRIGGER_MODE_OFF,
        PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerBG8, PixelType_Gvsp_Mono8,
        cast, POINTER, byref, memset, sizeof, c_ubyte,
    )

    os.makedirs(THU_MUC_ANH_FULL, exist_ok=True)

    print("SDK Version:", hex(MvCamera.MV_CC_GetSDKVersion()))
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Không tìm thấy camera nào! Mã lỗi:", ret)
        return "thoat"
    print(f"Tìm thấy {deviceList.nDeviceNum} camera")

    target_index = 0
    for i in range(deviceList.nDeviceNum):
        info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == MV_GIGE_DEVICE:
            ten = "".join(chr(c) for c in info.SpecialInfo.stGigEInfo.chModelName if c != 0)
            print(f"[{i}] GigE Camera: {ten}")
            if "MV-CS200" in ten:
                target_index = i
        elif info.nTLayerType == MV_USB_DEVICE:
            ten = "".join(chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
            print(f"[{i}] USB Camera: {ten}")

    stDeviceInfo = cast(deviceList.pDeviceInfo[target_index],
                        POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    if cam.MV_CC_CreateHandle(stDeviceInfo) != 0:
        print("Tạo handle lỗi")
        return "thoat"
    if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        print("Mở camera lỗi")
        cam.MV_CC_DestroyHandle()
        return "thoat"

    if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
        n = cam.MV_CC_GetOptimalPacketSize()
        if n > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", n)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    if cam.MV_CC_StartGrabbing() != 0:
        print("Start grabbing lỗi")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return "thoat"

    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * nPayloadSize)()

    TEN_CUA_SO_LIVE = "Camera - 'c' chup & ve ROI, 'r' xuat lai, 'q' thoat"
    print("\nĐang xem trực tiếp... 'c' = chụp + vẽ ROI, 'q' = thoát.")

    so_thu_tu = 0
    ket_thuc = "thoat"
    try:
        while True:
            if cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000) != 0:
                continue
            raw = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
            raw = raw.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            pt = stFrameInfo.enPixelType

            if pt == PixelType_Gvsp_BayerRG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
            elif pt == PixelType_Gvsp_BayerGB8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)
            elif pt == PixelType_Gvsp_BayerGR8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_GR2BGR)
            elif pt == PixelType_Gvsp_BayerBG8:
                img = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)
            elif pt == PixelType_Gvsp_Mono8:
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                print("Pixel format chưa được xử lý:", pt)
                continue

            cv2.imshow(TEN_CUA_SO_LIVE, cv2.resize(img, (960, 640)))
            phim = cv2.waitKey(1) & 0xFF

            if phim in (ord('q'), 27):
                break
            if phim == ord('r'):
                ket_thuc = "xuat_lai"
                break
            if phim != ord('c'):
                continue

            # ---- chụp: crop + mask rồi mới cho vẽ ROI ----
            so_thu_tu += 1
            crop, thong_bao = crop_va_dong_bo_huong(img, anh_chuan_info)
            if crop is None:
                print(f"\n[Chụp #{so_thu_tu}] KHÔNG crop được: {thong_bao}")
                continue
            print(f"\n[Chụp #{so_thu_tu}] {thong_bao}")

            ten_file = f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{so_thu_tu:03d}_crop.png"
            duong_dan = os.path.join(THU_MUC_ANH_FULL, ten_file)
            cv2.imwrite(duong_dan, crop)
            print(f"  -> Đã lưu ảnh crop: {duong_dan}")

            hd = ve_roi_mot_anh(crop, ten_file, nhan_all, dem,
                                f"[Chup #{so_thu_tu}] {ten_file}",
                                cho_dieu_huong=False)
            cv2.destroyWindow(TEN_CUA_SO_VE)

            if hd in ("xuat_lai", "thoat"):
                ket_thuc = hd
                break
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("\nĐã đóng camera.")

    return ket_thuc


# ====================================================================
def kich_thuoc_anh_mau():
    """Kích thước ảnh gốc dùng để quyết định cách chia.
    - doc_file : lấy từ ảnh đầu tiên trong thư mục
    - chup_anh : suy ra từ mask hình học, không cần chụp thử"""
    if NGUON == "chup_anh":
        return kich_thuoc_output_chuan()
    for ten in liet_ke_anh(THU_MUC_ANH_FULL):
        img = cv2.imread(os.path.join(THU_MUC_ANH_FULL, ten))
        if img is not None:
            return img.shape[1], img.shape[0]
    raise RuntimeError(f"Không đọc được ảnh nào trong: {THU_MUC_ANH_FULL}")


def main():
    nhan_all = doc_nhan(FILE_NHAN)

    w, h = kich_thuoc_anh_mau()
    thiet_lap_chia(w, h)
    tao_san_thu_muc()
    canh_bao_doi_cach_chia()
    luu_cau_hinh()

    if XUAT_LAI_TOAN_BO:
        xuat_lai_toan_bo(nhan_all)
        return

    dem = dem_hien_co()
    print(f"\nDataset hiện có (train/val = {TI_LE_TRAIN:.0%}/{1 - TI_LE_TRAIN:.0%}, "
          f"ngưỡng box = {TI_LE_BOX_TOI_THIEU:.2f}):")
    in_thong_ke(dem)

    if NGUON == "chup_anh":
        print("=== CHẾ ĐỘ CHỤP ẢNH (có crop + mask hình học) ===")
        print("Đang chuẩn bị ảnh chuẩn (dự phòng cho nhánh SIFT)...")
        anh_chuan_info = chuan_bi_anh_chuan(DUONG_DAN_ANH_CHUAN_CO_DINH)
        print(f"  Ảnh chuẩn có tỉ lệ lõm {anh_chuan_info['ti_le_lom']:.1f}")
        ket_thuc = chay_chup_anh(nhan_all, dem, anh_chuan_info)
    elif NGUON == "doc_file":
        print("=== CHẾ ĐỘ ĐỌC FILE (chỉ vẽ ROI) ===")
        ket_thuc = chay_doc_file(nhan_all, dem)
    else:
        raise ValueError('NGUON phải là "doc_file" hoặc "chup_anh"')

    if ket_thuc == "xuat_lai":
        xuat_lai_toan_bo(nhan_all)
    else:
        print("\nKết quả dataset:")
        in_thong_ke(dem)
    print(f"Nhãn lưu tại      : {FILE_NHAN}")
    print(f"Cấu hình chia part: {FILE_CAU_HINH}")


if __name__ == "__main__":
    main()