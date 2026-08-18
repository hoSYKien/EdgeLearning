r"""
chay_doc_file.py - CHẾ ĐỘ ĐỌC FILE.

Duyệt các ảnh CROP FULL có sẵn trong THU_MUC_ANH_FULL, hiện full ảnh lên,
người vận hành CHỈ VIỆC VẼ ROI. Vẽ xong bấm Enter là ảnh tự cắt part và
chia vào dataset theo tỉ lệ 80:20.

    python chay_doc_file.py

Ảnh đã gán nhãn trước đó mở lại vẫn thấy box cũ, sửa được; file part cũ
tự bị xoá và ghi lại theo nhãn mới.
"""

import os

import cv2

import cauhinh as cf
import chia_part as cp
import giao_dien as gd
import kho_dulieu as kd


def kich_thuoc_anh_dau_tien():
    for ten in kd.liet_ke_anh():
        img = cv2.imread(os.path.join(cf.THU_MUC_ANH_FULL, ten))
        if img is not None:
            return img.shape[1], img.shape[0]
    raise RuntimeError(f"Không đọc được ảnh nào trong: {cf.THU_MUC_ANH_FULL}")


def main():
    nhan_all = kd.doc_nhan()

    w, h = kich_thuoc_anh_dau_tien()
    cp.thiet_lap(w, h)
    kd.tao_san_thu_muc()
    cp.canh_bao_doi_cach_chia()
    cp.luu_cau_hinh()

    dem = kd.dem_hien_co()
    print(f"\nDataset hiện có (train/val = {cf.TI_LE_TRAIN:.0%}/"
          f"{1 - cf.TI_LE_TRAIN:.0%}, ngưỡng box = {cf.TI_LE_BOX_TOI_THIEU:.2f}):")
    kd.in_thong_ke(dem)

    danh_sach = kd.liet_ke_anh()
    da_gan = sum(1 for f in danh_sach if f in nhan_all)
    print(f"{len(danh_sach)} ảnh trong thư mục, {da_gan} ảnh đã có nhãn từ trước.")
    print("Kéo chuột trái để khoanh lỗi. Enter = lưu + chia vào dataset.\n")

    i, ket_thuc = 0, "thoat"
    while 0 <= i < len(danh_sach):
        ten_file = danh_sach[i]
        img = cv2.imread(os.path.join(cf.THU_MUC_ANH_FULL, ten_file))
        if img is None:
            print(f"Không đọc được ảnh, bỏ qua: {ten_file}")
            i += 1
            continue

        hd = gd.ve_roi_mot_anh(img, ten_file, nhan_all, dem,
                               f"[{i + 1}/{len(danh_sach)}] {ten_file}")
        if hd in ("tiep", "bo"):
            i += 1
        elif hd == "truoc":
            i = max(0, i - 1)
        else:
            ket_thuc = hd
            break
    else:
        print("Đã duyệt hết ảnh.")

    cv2.destroyAllWindows()

    if ket_thuc == "dung_lai":
        print()
        dem = kd.xuat_lai_toan_bo(nhan_all)
    else:
        print("\nKết quả dataset:")
        kd.in_thong_ke(dem)
    print(f"Nhãn lưu tại      : {cf.FILE_NHAN}")
    print(f"Cấu hình chia part: {cf.FILE_CAU_HINH}")


if __name__ == "__main__":
    main()
