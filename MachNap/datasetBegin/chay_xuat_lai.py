r"""
chay_xuat_lai.py - DỰNG LẠI TOÀN BỘ DATASET từ file nhãn.

Xoá sạch các thư mục PART rồi cắt lại từ đầu theo FILE_NHAN. KHÔNG phải
vẽ lại ảnh nào - box đã nằm trong JSON.

Dùng khi vừa đổi một trong các tham số:
    TI_LE_BOX_TOI_THIEU   (ngưỡng box)
    TI_LE_TRAIN           (tỉ lệ train/val)
    CHE_DO_CHIA / CHIA_THEO / SO_PHAN_CHINH / TAO_PART_GOI  (cách chia part)

    python chay_xuat_lai.py          -> hỏi xác nhận rồi làm
    python chay_xuat_lai.py -y       -> làm luôn, không hỏi
"""

import sys

import cauhinh as cf
import chia_part as cp
import kho_dulieu as kd


def main():
    nhan_all = kd.doc_nhan()
    if not nhan_all:
        print(f"Chưa có nhãn nào trong: {cf.FILE_NHAN}")
        return

    mau = next(iter(nhan_all.values()))
    cp.thiet_lap(mau["w"], mau["h"])
    print(f"\n{len(nhan_all)} ảnh trong file nhãn.")
    print(f"Ngưỡng box = {cf.TI_LE_BOX_TOI_THIEU} | train/val = "
          f"{cf.TI_LE_TRAIN:.0%}/{1 - cf.TI_LE_TRAIN:.0%}")
    print("\nSẼ XOÁ các thư mục sau rồi dựng lại:")
    for d in dict.fromkeys(cp.thu_muc_goc_part(t) for t, _, _ in cp.RATIOS):
        print(f"   {d}")

    if "-y" not in sys.argv:
        if input("\nĐồng ý? (go/khong): ").strip().lower() not in ("go", "y", "yes"):
            print("Đã huỷ, không đụng gì tới dataset.")
            return

    print()
    kd.xuat_lai_toan_bo(nhan_all)
    print(f"Cấu hình chia part: {cf.FILE_CAU_HINH}")


if __name__ == "__main__":
    main()
