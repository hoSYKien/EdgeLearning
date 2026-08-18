r"""
06_tao_dataset_train_val.py
---------------------------
Gom ảnh đã lọc thành cấu trúc dataset cho 5 part.

NGUỒN:
    crop9\part1..part5   -> toàn bộ là ảnh NG
    crop10\part1..part5  -> toàn bộ là ảnh OK

KẾT QUẢ:
    <THU_MUC_DICH>\PART1\train\OK\...
                        \train\NG\...
                        \val\OK\...
                        \val\NG\...
    ... tương tự PART2 -> PART5

Cách chia: mỗi part, mỗi class được chia riêng theo TI_LE_TRAIN (chia phân
tầng - tỉ lệ OK/NG trong train và val giống nhau). Thứ tự xáo trộn cố định
theo SEED nên chạy lại nhiều lần vẫn ra đúng bộ chia đó.

Cách dùng:
    1. Sửa đường dẫn + tham số bên dưới.
    2. Chạy thử với CHE_DO_THU = True để xem thống kê, KHÔNG copy gì cả.
    3. Ưng ý thì đặt CHE_DO_THU = False và chạy lại.
"""
 
import os
import random
import shutil

# ====================== CHỈNH Ở ĐÂY ======================
THU_MUC_NG = r"D:\TongHop\RTC Technologi\PCB\crop9"    # chứa part1..part5, toàn ảnh NG
THU_MUC_OK = r"D:\TongHop\RTC Technologi\PCB\crop10"   # chứa part1..part5, toàn ảnh OK
THU_MUC_DICH = r"D:\TongHop\RTC Technologi\PCB\dataset14"

TEN_CAC_PART = ["part1", "part2", "part3", "part4", "part5"]

TI_LE_TRAIN = 0.8    # 0.8 = 80% train, 20% val
SEED = 42            # cố định để chạy lại vẫn ra đúng bộ chia cũ

CHE_DO_THU = False    # True = chỉ in thống kê, KHÔNG copy file
DI_CHUYEN = False    # False = copy (giữ nguyên file gốc), True = di chuyển
XOA_DICH_CU = False  # True = xoá sạch THU_MUC_DICH trước khi tạo mới

# Tên thư mục part ở đích. "PART{n}" -> PART1, PART2...
MAU_TEN_PART_DICH = "PART{n}"

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
# =========================================================


def liet_ke_anh(thu_muc):
    if not os.path.isdir(thu_muc):
        return None
    return sorted(f for f in os.listdir(thu_muc)
                  if f.lower().endswith(VALID_EXT) and
                  os.path.isfile(os.path.join(thu_muc, f)))


def chia_train_val(danh_sach, ti_le_train, rng):
    """Xáo trộn rồi cắt. Đảm bảo val có ít nhất 1 ảnh khi tổng >= 2."""
    ds = list(danh_sach)
    rng.shuffle(ds)
    n_train = int(round(len(ds) * ti_le_train))
    if len(ds) >= 2:
        n_train = min(max(n_train, 1), len(ds) - 1)
    return ds[:n_train], ds[n_train:]


def main():
    if XOA_DICH_CU and not CHE_DO_THU and os.path.isdir(THU_MUC_DICH):
        print(f"Đang xoá thư mục đích cũ: {THU_MUC_DICH}")
        shutil.rmtree(THU_MUC_DICH)

    thao_tac = "DI CHUYỂN" if DI_CHUYEN else "COPY"
    print(f"Nguồn NG: {THU_MUC_NG}")
    print(f"Nguồn OK: {THU_MUC_OK}")
    print(f"Đích    : {THU_MUC_DICH}")
    print(f"Chia    : train {TI_LE_TRAIN:.0%} / val {1 - TI_LE_TRAIN:.0%} | "
          f"seed {SEED} | thao tác: {thao_tac}")
    if CHE_DO_THU:
        print("*** CHẾ ĐỘ THỬ - chỉ in thống kê, KHÔNG tạo/ghi file nào ***")
    print()

    tong = {"train": {"OK": 0, "NG": 0}, "val": {"OK": 0, "NG": 0}}
    loi = []

    for i, ten_part in enumerate(TEN_CAC_PART, start=1):
        ten_dich = MAU_TEN_PART_DICH.format(n=i)
        print(f"[{ten_dich}]  (nguồn: {ten_part})")

        nguon = {"NG": os.path.join(THU_MUC_NG, ten_part),
                 "OK": os.path.join(THU_MUC_OK, ten_part)}

        anh = {}
        thieu = False
        for lop, duong_dan in nguon.items():
            ds = liet_ke_anh(duong_dan)
            if ds is None:
                print(f"   LỖI: không có thư mục {duong_dan}")
                loi.append(duong_dan)
                thieu = True
            elif not ds:
                print(f"   LỖI: thư mục rỗng {duong_dan}")
                loi.append(duong_dan)
                thieu = True
            else:
                anh[lop] = ds
        if thieu:
            print()
            continue

        # mỗi part + mỗi class dùng seed riêng nhưng suy ra từ SEED
        for lop in ("OK", "NG"):
            rng = random.Random(f"{SEED}-{ten_dich}-{lop}")
            train, val = chia_train_val(anh[lop], TI_LE_TRAIN, rng)
            print(f"   {lop}: {len(anh[lop]):4d} ảnh -> train {len(train):4d} | val {len(val):4d}")

            for tap, ds in (("train", train), ("val", val)):
                thu_muc = os.path.join(THU_MUC_DICH, ten_dich, tap, lop)
                tong[tap][lop] += len(ds)
                if CHE_DO_THU:
                    continue
                os.makedirs(thu_muc, exist_ok=True)
                for ten_file in ds:
                    src = os.path.join(nguon[lop], ten_file)
                    dst = os.path.join(thu_muc, ten_file)
                    if os.path.exists(dst):
                        goc, duoi = os.path.splitext(ten_file)
                        k = 1
                        while os.path.exists(dst):
                            dst = os.path.join(thu_muc, f"{goc}_{k}{duoi}")
                            k += 1
                    if DI_CHUYEN:
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
        print()

    print("=" * 56)
    for tap in ("train", "val"):
        n_ok, n_ng = tong[tap]["OK"], tong[tap]["NG"]
        t = n_ok + n_ng
        ti_le = f" (OK {n_ok / t:.0%} / NG {n_ng / t:.0%})" if t else ""
        print(f"{tap:5s}: {t:5d} ảnh - OK {n_ok:5d} | NG {n_ng:5d}{ti_le}")
    print(f"TỔNG : {sum(tong[t][l] for t in tong for l in tong[t])} ảnh")

    if loi:
        print(f"\nCó {len(loi)} thư mục nguồn thiếu/rỗng:")
        for d in loi:
            print(f"  - {d}")

    if CHE_DO_THU:
        print("\nĐây mới là chạy thử. Đặt CHE_DO_THU = False rồi chạy lại để thực sự "
              f"{'di chuyển' if DI_CHUYEN else 'copy'} file.")
    else:
        print(f"\nXong. Dataset nằm tại: {THU_MUC_DICH}")


if __name__ == "__main__":
    main()