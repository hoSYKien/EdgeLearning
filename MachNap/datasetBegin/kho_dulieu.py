r"""
kho_dulieu.py - Quản lý file nhãn JSON và ghi ảnh part vào dataset.

Nhiệm vụ:
    - đọc/ghi FILE_NHAN (box người vận hành vẽ)
    - cắt part theo nhãn rồi bỏ vào <PART>\train|val\OK|NG
    - giữ tỉ lệ train/val bằng bộ đếm (không random, không lệch)
    - sửa nhãn thì xoá file part cũ rồi ghi lại

Chạy riêng để kiểm tra tình trạng dataset:
    python kho_dulieu.py
"""

import json
import os
import shutil

import cv2

import cauhinh as cf
import chia_part as cp


# ==================== FILE NHÃN ====================

def doc_nhan(duong_dan=None):
    duong_dan = duong_dan or cf.FILE_NHAN
    if os.path.isfile(duong_dan):
        with open(duong_dan, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def ghi_nhan(du_lieu, duong_dan=None):
    duong_dan = duong_dan or cf.FILE_NHAN
    thu_muc = os.path.dirname(duong_dan)
    if thu_muc:
        os.makedirs(thu_muc, exist_ok=True)
    with open(duong_dan, "w", encoding="utf-8") as f:
        json.dump(du_lieu, f, ensure_ascii=False, indent=1)


def liet_ke_anh(thu_muc=None):
    thu_muc = thu_muc or cf.THU_MUC_ANH_FULL
    if not os.path.isdir(thu_muc):
        raise FileNotFoundError(f"Không có thư mục ảnh: {thu_muc}")
    return sorted(f for f in os.listdir(thu_muc)
                  if f.lower().endswith(cf.VALID_EXT)
                  and os.path.isfile(os.path.join(thu_muc, f)))


# ==================== THƯ MỤC DATASET ====================

def thu_muc_dich(ten_part, tap, lop):
    return os.path.join(cp.thu_muc_goc_part(ten_part), tap, lop)


def tao_san_thu_muc():
    for ten_part, _, _ in cp.RATIOS:
        for tap in cf.CAC_TAP:
            for lop in cf.CAC_LOP:
                os.makedirs(thu_muc_dich(ten_part, tap, lop), exist_ok=True)


def dem_hien_co():
    """Đếm ảnh đang có trong từng ô -> phiên sau chạy tiếp vẫn đúng tỉ lệ."""
    dem = {}
    for ten_part, _, _ in cp.RATIOS:
        for tap in cf.CAC_TAP:
            for lop in cf.CAC_LOP:
                d = thu_muc_dich(ten_part, tap, lop)
                dem[(ten_part, tap, lop)] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return dem


def chon_tap(dem, ten_part, lop):
    """Chọn train hay val để bám sát TI_LE_TRAIN.
    Với 0.8: ảnh 1-4 vào train, ảnh 5 vào val, rồi lặp lại."""
    n_train = dem.get((ten_part, "train", lop), 0)
    n_val = dem.get((ten_part, "val", lop), 0)
    return "train" if n_train < (n_train + n_val + 1) * cf.TI_LE_TRAIN else "val"


# ==================== GHI ẢNH PART ====================

def _xoa_ban_cu(gc, dem):
    """Xoá file part đã xuất trước đó của ảnh này (khi sửa nhãn)."""
    for ten_part, (tap, lop, duong_dan) in (gc.get("xuat") or {}).items():
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
        ghi_nhan(nhan_all)
        return "BO QUA - da go khoi dataset"

    img_h, img_w = img.shape[:2]
    goc = os.path.splitext(ten_file)[0]
    tom_tat = []
    for ten_part, (x0, y0, x1, y1), nhan, _ in cp.nhan_cac_part(
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

    ghi_nhan(nhan_all)
    return " | ".join(tom_tat)


def in_thong_ke(dem):
    print("-" * 64)
    for ten_part, _, _ in cp.RATIOS:
        o = {(t, l): dem.get((ten_part, t, l), 0)
             for t in cf.CAC_TAP for l in cf.CAC_LOP}
        print(f"{ten_part:8s}: tong {sum(o.values()):4d} | "
              f"train OK {o[('train', 'OK')]:4d} NG {o[('train', 'NG')]:4d} | "
              f"val OK {o[('val', 'OK')]:4d} NG {o[('val', 'NG')]:4d}")
    print("-" * 64)


def xuat_lai_toan_bo(nhan_all):
    """Xoá sạch thư mục part rồi dựng lại từ JSON. Dùng khi đổi cách chia,
    đổi TI_LE_BOX_TOI_THIEU hoặc TI_LE_TRAIN."""
    print("Đang xoá dataset cũ và dựng lại từ đầu...")
    da_xoa = set()
    for ten_part, _, _ in cp.RATIOS:
        d = cp.thu_muc_goc_part(ten_part)
        if d not in da_xoa and os.path.isdir(d):
            shutil.rmtree(d)
            da_xoa.add(d)
    tao_san_thu_muc()
    dem = dem_hien_co()

    for gc in nhan_all.values():
        gc["xuat"] = {}

    so, thieu = 0, 0
    for ten_file in sorted(nhan_all):
        duong_dan = os.path.join(cf.THU_MUC_ANH_FULL, ten_file)
        img = cv2.imread(duong_dan) if os.path.isfile(duong_dan) else None
        if img is None:
            thieu += 1
            continue
        xuat_mot_anh(img, ten_file, nhan_all, dem)
        so += 1
    ghi_nhan(nhan_all)
    cp.luu_cau_hinh()
    print(f"Đã dựng lại {so} ảnh full"
          f"{f' (thiếu {thieu} ảnh trong JSON không tìm thấy trên đĩa)' if thieu else ''}. "
          f"Ngưỡng box = {cf.TI_LE_BOX_TOI_THIEU:.2f}, train/val = "
          f"{cf.TI_LE_TRAIN:.0%}/{1 - cf.TI_LE_TRAIN:.0%}")
    in_thong_ke(dem)
    return dem


def kiem_tra_toan_ven(nhan_all):
    """Soát xem JSON và đĩa có khớp nhau không."""
    thieu_anh, thieu_part, chua_gan = [], [], []
    for ten_file, gc in nhan_all.items():
        if not os.path.isfile(os.path.join(cf.THU_MUC_ANH_FULL, ten_file)):
            thieu_anh.append(ten_file)
            continue
        if gc.get("bo_qua"):
            continue
        for ten_part, (_, _, duong_dan) in (gc.get("xuat") or {}).items():
            if not os.path.isfile(duong_dan):
                thieu_part.append(f"{ten_file} -> {ten_part}")
    try:
        for f in liet_ke_anh():
            if f not in nhan_all:
                chua_gan.append(f)
    except FileNotFoundError as e:
        print(e)

    print(f"Ảnh đã gán nhãn      : {len(nhan_all)}")
    print(f"Ảnh chưa gán nhãn    : {len(chua_gan)}")
    print(f"Nhãn mất ảnh gốc     : {len(thieu_anh)}")
    print(f"File part bị mất     : {len(thieu_part)}")
    for ds, ten in ((thieu_anh, "MẤT ẢNH GỐC"), (thieu_part, "MẤT FILE PART")):
        for x in ds[:5]:
            print(f"   [{ten}] {x}")
        if len(ds) > 5:
            print(f"   ... và {len(ds) - 5} cái nữa")
    if thieu_part:
        print("\n-> Chạy  python chay_xuat_lai.py  để dựng lại cho khớp.")


if __name__ == "__main__":
    nhan_all = doc_nhan()
    if not nhan_all:
        print(f"Chưa có nhãn nào trong: {cf.FILE_NHAN}")
        raise SystemExit

    mau = next(iter(nhan_all.values()))
    cp.thiet_lap(mau["w"], mau["h"])
    print()
    kiem_tra_toan_ven(nhan_all)
    print()
    in_thong_ke(dem_hien_co())
