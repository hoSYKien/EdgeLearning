r"""
chia_part.py - Tính CÁCH CHIA PART và NHÃN OK/NG cho từng part.
Không đụng tới file ảnh, không đụng tới đĩa (trừ luu_cau_hinh).

Chạy riêng để xem cách chia:
    python chia_part.py 2900 1650
    python chia_part.py 900 2400
"""

import json
import os
import sys

import cauhinh as cf

# Trạng thái cách chia hiện hành - do thiet_lap() điền vào.
CHIEU = None      # "ngang" | "doc"
SO_PHAN = None    # số phần chính
RATIOS = []       # [(ten_part, start, end), ...] theo % cạnh bị cắt


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
    """Auto: cắt dọc theo CẠNH DÀI, số phần sao cho mỗi phần dài khoảng
    AUTO_TI_LE_MUC_TIEU x cạnh ngắn."""
    chieu = "ngang" if w >= h else "doc"
    canh_dai, canh_ngan = (w, h) if w >= h else (h, w)
    so_phan = int(round(canh_dai / (cf.AUTO_TI_LE_MUC_TIEU * canh_ngan)))
    so_phan = max(cf.AUTO_SO_PHAN_MIN, min(cf.AUTO_SO_PHAN_MAX, so_phan))
    return chieu, so_phan


def thiet_lap(w, h, in_ra=True):
    """Điền CHIEU / SO_PHAN / RATIOS. Gọi 1 lần trước khi gán nhãn."""
    global CHIEU, SO_PHAN, RATIOS
    if cf.CHE_DO_CHIA == "auto":
        CHIEU, SO_PHAN = tu_chon_cach_chia(w, h)
    elif cf.CHE_DO_CHIA == "manual":
        if cf.CHIA_THEO not in ("ngang", "doc"):
            raise ValueError('CHIA_THEO phải là "ngang" hoặc "doc"')
        if cf.SO_PHAN_CHINH < 2:
            raise ValueError("SO_PHAN_CHINH phải >= 2")
        CHIEU, SO_PHAN = cf.CHIA_THEO, cf.SO_PHAN_CHINH
    else:
        raise ValueError('CHE_DO_CHIA phải là "auto" hoặc "manual"')

    RATIOS = tinh_ratios(SO_PHAN, cf.TAO_PART_GOI)
    if in_ra:
        in_cach_chia(w, h)
    return RATIOS


def in_cach_chia(w, h):
    canh = "chiều rộng (cắt thẳng đứng)" if CHIEU == "ngang" \
        else "chiều cao (cắt nằm ngang)"
    goi = f" + {SO_PHAN - 1} phần gối" if cf.TAO_PART_GOI else ""
    print(f"Ảnh gốc {w}x{h} (tỉ lệ {max(w, h) / min(w, h):.2f}) -> chia theo "
          f"{canh}, {SO_PHAN} phần chính{goi} = {len(RATIOS)} part "
          f"[{cf.CHE_DO_CHIA}]")
    for ten, s, e in RATIOS:
        x0, y0, x1, y1 = vung_part(w, h, s, e)
        print(f"   {ten:8s}  {s:.3f} -> {e:.3f}   ({x1 - x0}x{y1 - y0} px)")


def vung_part(w, h, start, end):
    """Trả về (x0, y0, x1, y1) của part trên ảnh w x h."""
    if CHIEU == "doc":
        y0 = max(0, min(int(round(h * start)), h))
        y1 = max(0, min(int(round(h * end)), h))
        return 0, y0, w, y1
    x0 = max(0, min(int(round(w * start)), w))
    x1 = max(0, min(int(round(w * end)), w))
    return x0, 0, x1, h


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
    """Trả về [(ten_part, (x0,y0,x1,y1), 'OK'/'NG', ti_le_lon_nhat), ...]

    Ngưỡng xét ĐỘC LẬP cho từng part và từng box: box bị chia đôi thì CẢ HAI
    part chứa nó đều NG (mỗi bên 50% >= ngưỡng). Ngưỡng chỉ để bỏ mấy mẩu
    vụn lòi sang part bên cạnh."""
    if not RATIOS:
        raise RuntimeError("Chưa gọi chia_part.thiet_lap(w, h)")
    if nguong is None:
        nguong = cf.TI_LE_BOX_TOI_THIEU

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

        if cf.KHONG_DE_BOX_BIEN_MAT and not any(
                t >= nguong and t > 0 for t in tl_theo_part.values()):
            ten_lon_nhat = max(tl_theo_part, key=tl_theo_part.get)
            if tl_theo_part[ten_lon_nhat] > 0:
                nhan[ten_lon_nhat] = "NG"

    return [(ten, vung[ten], nhan[ten], tl_max[ten]) for ten, _, _ in RATIOS]


def thu_muc_goc_part(ten_part):
    if ten_part in cf.THU_MUC_PART_RIENG:
        return cf.THU_MUC_PART_RIENG[ten_part]
    return os.path.join(cf.THU_MUC_DATASET, ten_part.upper())


def luu_cau_hinh():
    """Ghi cách chia ra file để script chạy thật dùng lại đúng vùng."""
    os.makedirs(os.path.dirname(cf.FILE_CAU_HINH) or ".", exist_ok=True)
    with open(cf.FILE_CAU_HINH, "w", encoding="utf-8") as f:
        json.dump({"chieu": CHIEU, "so_phan": SO_PHAN,
                   "tao_part_goi": cf.TAO_PART_GOI,
                   "ti_le_box_toi_thieu": cf.TI_LE_BOX_TOI_THIEU,
                   "ti_le_train": cf.TI_LE_TRAIN,
                   "ratios": [[t, s, e] for t, s, e in RATIOS]},
                  f, ensure_ascii=False, indent=1)


def canh_bao_doi_cach_chia():
    """Dataset cũ chia kiểu khác -> nhắc chạy chay_xuat_lai.py."""
    if not os.path.isfile(cf.FILE_CAU_HINH):
        return False
    try:
        with open(cf.FILE_CAU_HINH, "r", encoding="utf-8") as f:
            cu = json.load(f)
    except Exception:
        return False
    if (cu.get("chieu") != CHIEU or cu.get("so_phan") != SO_PHAN
            or cu.get("tao_part_goi") != cf.TAO_PART_GOI):
        print(f"\n*** CẢNH BÁO: dataset hiện có chia theo {cu.get('chieu')} / "
              f"{cu.get('so_phan')} phần, khác cấu hình bây giờ "
              f"({CHIEU} / {SO_PHAN} phần).")
        print("    Chạy  python chay_xuat_lai.py  để dựng lại dataset.\n")
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) == 3:
        w, h = int(sys.argv[1]), int(sys.argv[2])
    else:
        w, h = 2900, 1650
        print("(không truyền kích thước -> dùng mặc định 2900x1650)")
        print("Cách dùng: python chia_part.py <rong> <cao>\n")
    thiet_lap(w, h)

    # thử 1 box nằm ngay ranh giới 2 phần chính đầu tiên để xem nhãn
    if CHIEU == "ngang":
        giua = int(w / SO_PHAN)
        box = [giua - 40, h // 2 - 40, giua + 40, h // 2 + 40]
    else:
        giua = int(h / SO_PHAN)
        box = [w // 2 - 40, giua - 40, w // 2 + 40, giua + 40]
    print(f"\nThử 1 box 80x80 nằm ĐÚNG vết cắt đầu tiên: {box}")
    for ten, _, nhan, tl in nhan_cac_part([box], w, h):
        print(f"   {ten:8s} {nhan}  ({tl * 100:.0f}% cua box)")
