"""
Chia dữ liệu ảnh đã phân loại OK/NG thành 3 tập train/test/val theo tỉ lệ
cho trước (mặc định 75:10:15), mỗi tập vẫn giữ cấu trúc con OK/NG.

Cấu trúc thư mục ĐẦU VÀO (đã có sẵn):
    roi_1/
        OK/   (chứa các ảnh loại OK)
        NG/   (chứa các ảnh loại NG)

Cấu trúc thư mục ĐẦU RA (script tạo ra):
    roi_1_split/
        train/
            OK/
            NG/
        val/
            OK/
            NG/
        test/
            OK/
            NG/

Chia RIÊNG cho từng lớp OK và NG (không trộn chung rồi chia) - đảm bảo mỗi
tập train/val/test đều có đủ tỉ lệ OK/NG tương ứng, không bị lệch lớp.

Cách dùng:
    Sửa đường dẫn ở phần "CHỈNH ĐƯỜNG DẪN Ở ĐÂY" bên dưới rồi chạy:
        python chia_train_val_test.py
"""

import os
import random
import shutil

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_GOC = r"D:\TongHop\RTC Technologi\PCB\cropVung\roi_1"        # thư mục chứa OK/ và NG/
DUONG_DAN_OUTPUT = r"D:\TongHop\RTC Technologi\PCB\cropVung\roi_1_split"  # thư mục chứa train/val/test

TI_LE_TRAIN = 0.75
TI_LE_VAL = 0.10
TI_LE_TEST = 0.15

CAC_LOP = ["OK", "NG"]          # tên các thư mục con (lớp) cần chia
CHE_DO_COPY = True              # True = copy (giữ nguyên ảnh gốc), False = move (di chuyển hẳn)
RANDOM_SEED = 42                # cố định để chia lại vẫn ra cùng kết quả, đổi số khác nếu muốn xáo trộn khác
CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")
# =====================================================================


def chia_1_lop(danh_sach_file: list, ti_le_train: float, ti_le_val: float, ti_le_test: float):
    """Xáo trộn rồi cắt danh sách file thành 3 phần theo tỉ lệ."""
    files = danh_sach_file[:]
    random.shuffle(files)

    n = len(files)
    n_train = round(n * ti_le_train)
    n_val = round(n * ti_le_val)
    # phần còn lại dồn hết cho test để tổng luôn khớp đúng n (tránh lệch do làm tròn)
    n_test = n - n_train - n_val

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    assert len(train) + len(val) + len(test) == n
    return {"train": train, "val": val, "test": test}, n_test


def chia_train_val_test(duong_dan_goc: str, duong_dan_output: str):
    tong = TI_LE_TRAIN + TI_LE_VAL + TI_LE_TEST
    if abs(tong - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ train+val+test phải bằng 1.0, hiện đang là {tong}")

    random.seed(RANDOM_SEED)

    print(f"Tỉ lệ chia: train={TI_LE_TRAIN:.0%}  val={TI_LE_VAL:.0%}  test={TI_LE_TEST:.0%}\n")

    thong_ke = {}
    for lop in CAC_LOP:
        thu_muc_lop = os.path.join(duong_dan_goc, lop)
        if not os.path.isdir(thu_muc_lop):
            print(f"CẢNH BÁO: không thấy thư mục '{thu_muc_lop}', bỏ qua lớp '{lop}'.")
            continue

        danh_sach_file = sorted(
            f for f in os.listdir(thu_muc_lop) if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
        )
        if not danh_sach_file:
            print(f"CẢNH BÁO: thư mục '{thu_muc_lop}' không có ảnh nào, bỏ qua.")
            continue

        phan_chia, _ = chia_1_lop(danh_sach_file, TI_LE_TRAIN, TI_LE_VAL, TI_LE_TEST)
        thong_ke[lop] = {k: len(v) for k, v in phan_chia.items()}

        for tap, danh_sach in phan_chia.items():
            thu_muc_dich = os.path.join(duong_dan_output, tap, lop)
            os.makedirs(thu_muc_dich, exist_ok=True)
            for ten_file in danh_sach:
                nguon = os.path.join(thu_muc_lop, ten_file)
                dich = os.path.join(thu_muc_dich, ten_file)
                if CHE_DO_COPY:
                    shutil.copy2(nguon, dich)
                else:
                    shutil.move(nguon, dich)

        print(f"[{lop}] Tổng {len(danh_sach_file)} ảnh -> "
              f"train={thong_ke[lop]['train']}, val={thong_ke[lop]['val']}, "
              f"test={thong_ke[lop]['test']}")

    print(f"\nĐã lưu vào: {duong_dan_output}")
    print("Cấu trúc:")
    for tap in ["train", "val", "test"]:
        tong_tap = sum(thong_ke[lop][tap] for lop in thong_ke)
        chi_tiet = ", ".join(f"{lop}={thong_ke[lop][tap]}" for lop in thong_ke)
        print(f"  {tap}/  (tổng {tong_tap}: {chi_tiet})")


if __name__ == "__main__":
    chia_train_val_test(DUONG_DAN_GOC, DUONG_DAN_OUTPUT)