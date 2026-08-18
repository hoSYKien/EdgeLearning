r"""
cauhinh.py - TẤT CẢ tham số nằm ở đây, các file khác chỉ đọc.
Sửa dự án mới thì chỉ động vào file này.
"""

import os

# ====================== ĐƯỜNG DẪN ======================
# Ảnh CROP FULL (đã cắt/xoay/xoá nền, chưa chia part).
#   - chay_doc_file.py : đọc ảnh từ đây
#   - chay_chup_anh.py : lưu ảnh vừa crop vào đây
THU_MUC_ANH_FULL = r"D:\TongHop\RTC Technologi\PCB\crop_full"

# File JSON lưu box người vận hành vẽ.
FILE_NHAN = r"D:\TongHop\RTC Technologi\PCB\nhan_box.json"

# Thư mục dataset gốc. Mỗi part tự có thư mục con:
#   <goc>\PART1\train\OK , ...\train\NG , ...\val\OK , ...\val\NG
THU_MUC_DATASET = r"D:\TongHop\RTC Technologi\PCB\dataset15"

# Part nào cần để ổ khác thì khai báo đè, vd {"part1": r"E:\du_lieu\PART1"}
THU_MUC_PART_RIENG = {}

# File ghi lại cách chia để script chạy thật (inference) đọc đúng vùng.
FILE_CAU_HINH = os.path.join(THU_MUC_DATASET, "cau_hinh_part.json")


# ====================== CÁCH CHIA PART ======================
# Chia N phần chính dọc theo 1 chiều, + (N-1) phần GỐI nằm giữa 2 phần
# chính cạnh nhau (để lỗi nhỏ nằm ngay vết cắt không bị xẻ đôi).
#   N = 3 -> 5 part : part1, part2, part3, part12, part23
#   Tổng quát: 2N - 1 part.
CHE_DO_CHIA = "auto"      # "auto" | "manual"

# --- chỉ dùng khi CHE_DO_CHIA = "manual" ---
CHIA_THEO = "ngang"       # "ngang" = cắt thẳng đứng (theo chiều rộng)
                          # "doc"   = cắt nằm ngang (theo chiều cao)
SO_PHAN_CHINH = 3         # >= 2

# --- chỉ dùng khi CHE_DO_CHIA = "auto" ---
# Auto luôn cắt dọc theo CẠNH DÀI. Mỗi phần chính dài khoảng
# (AUTO_TI_LE_MUC_TIEU x cạnh ngắn):
#   0.6 -> part hơi dẹt, chia được nhiều phần (mặc định)
#   1.0 -> part gần vuông, chia ít phần
AUTO_TI_LE_MUC_TIEU = 0.6
AUTO_SO_PHAN_MIN = 2
AUTO_SO_PHAN_MAX = 8

TAO_PART_GOI = True       # False = chỉ có N phần chính, bỏ phần gối


# ====================== GÁN NHÃN ======================
# Tỉ lệ tối thiểu của box phải nằm trong part thì part đó mới tính NG.
#   0.0  -> dính 1 pixel là NG
#   0.15 -> part chỉ chứa dưới 15% cái box thì vẫn coi là OK
TI_LE_BOX_TOI_THIEU = 0.1

# Box vụn đến mức KHÔNG part nào đạt ngưỡng -> ép part chứa phần lớn nhất
# thành NG, để lỗi không biến mất khỏi dataset.
KHONG_DE_BOX_BIEN_MAT = True

TI_LE_TRAIN = 0.8         # 0.8 = 80% train / 20% val

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
CAC_TAP = ("train", "val")
CAC_LOP = ("OK", "NG")


# ====================== GIAO DIỆN ======================
MAX_HIEN_THI_W = 1500
MAX_HIEN_THI_H = 780
BOX_TOI_THIEU_PX = 8      # kéo nhỏ hơn ngần này coi như lỡ tay, bỏ


# ============ CHỈ DÙNG CHO chay_chup_anh.py ============
DLL_DIR_MVS = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
TEN_MODEL_CAMERA = "MV-CS200"   # ưu tiên chọn camera có tên chứa chuỗi này

# Ảnh chuẩn cố định - chỉ dùng cho nhánh dự phòng SIFT.
DUONG_DAN_ANH_CHUAN_CO_DINH = r"D:\TongHop\RTC Technologi\PCB\crop5\train\OK\Image_20260723101631285_roi.png"

# --- PHẢI GIỐNG HỆT 01_tao_template_v7.py và script chạy thật ---
KHUNG_CHUAN_W = 3187
KHUNG_CHUAN_H = 1777

PAD_ROI_NGANG = 300
PAD_ROI_DOC = 60

NGUONG_TI_LE_LOM = 2.0
AP_MASK_HINH_HOC = True

# Thân board: chữ nhật bo góc
BOARD_X1 = 0.0948
BOARD_X2 = 0.9049
BOARD_Y1 = 0.0360
BOARD_Y2 = 0.9657
BOARD_BO_GOC = 0.0141

# Khối USB Type-C nhô ra bên PHẢI
USB_X1 = 0.8472
USB_X2 = 0.9576
USB_Y1 = 0.2746
USB_Y2 = 0.6989
USB_BO_GOC = 0.0113

# Vết khuyết (tab bẻ) khoét ĐEN ở cạnh TRÁI. Đặt 0 để tắt.
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


if __name__ == "__main__":
    print("Đường dẫn đang dùng:")
    for ten in ("THU_MUC_ANH_FULL", "FILE_NHAN", "THU_MUC_DATASET",
                "FILE_CAU_HINH", "DUONG_DAN_ANH_CHUAN_CO_DINH"):
        gia_tri = globals()[ten]
        co = os.path.exists(gia_tri)
        print(f"  {'[co]  ' if co else '[THIEU]'} {ten:28s} = {gia_tri}")
    print(f"\nChia: {CHE_DO_CHIA}", end="")
    if CHE_DO_CHIA == "manual":
        print(f" ({CHIA_THEO}, {SO_PHAN_CHINH} phần chính)")
    else:
        print(f" (mục tiêu {AUTO_TI_LE_MUC_TIEU}, "
              f"{AUTO_SO_PHAN_MIN}-{AUTO_SO_PHAN_MAX} phần)")
    print(f"Ngưỡng box: {TI_LE_BOX_TOI_THIEU} | train/val: "
          f"{TI_LE_TRAIN:.0%}/{1 - TI_LE_TRAIN:.0%}")
