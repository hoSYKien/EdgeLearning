# Gán nhãn dataset bằng cách vẽ ROI

Người vận hành chỉ nhìn ảnh full và khoanh vùng lỗi. Việc chia part, gán
OK/NG, chia train/val do script lo.

## Các file

| File | Nhiệm vụ | Chạy riêng được? |
|---|---|---|
| `cauhinh.py` | Toàn bộ tham số. Sửa dự án mới chỉ động vào đây | `python cauhinh.py` → soi đường dẫn nào thiếu |
| `chia_part.py` | Tính cách chia part + nhãn OK/NG từ box | `python chia_part.py 2900 1650` → xem sẽ chia thế nào |
| `tim_vat_the.py` | **Tìm vật bằng trừ nền** (mọi màu) | `python tim_vat_the.py anh.png` → xem mask + contour |
| `hoc_nen.py` | Chụp nền trống, lưu lại | ✔ chạy 1 lần lúc lắp đặt |
| `xu_ly_anh.py` | Align + xoá nền + SIFT | `python xu_ly_anh.py anh.png` → crop thử 1 ảnh |
| `tao_mask_chuan.py` | Sinh mask chuẩn từ N ảnh OK | ✔ khi sang vật mới |
| `kho_dulieu.py` | File nhãn JSON + ghi ảnh part vào dataset | `python kho_dulieu.py` → kiểm tra dataset có khớp nhãn không |
| `giao_dien.py` | Cửa sổ vẽ ROI | `python giao_dien.py anh.png` → vẽ thử, không ghi gì |
| `chay_doc_file.py` | **Gán nhãn ảnh có sẵn** | ✔ việc chính |
| `chay_chup_anh.py` | **Chụp camera → crop → gán nhãn** | ✔ việc chính |
| `chay_xuat_lai.py` | Dựng lại dataset từ nhãn | ✔ khi đổi tham số |

Phụ thuộc: `chay_*` → `giao_dien` → `kho_dulieu` → `chia_part` → `cauhinh`,
và `xu_ly_anh` → `tim_vat_the` → `cauhinh`. Chỉ một chiều, không vòng.

## Lắp đặt lần đầu / sang vật mới

```bash
python hoc_nen.py            # dọn trống bàn, học nền (bắt buộc, 1 lần)
python tim_vat_the.py x.png  # kiểm tra bắt được vật chưa
# chụp ~30 ảnh OK với NGUON_MASK = "tat", LUU_MASK_ALIGN = True
python tao_mask_chuan.py     # sinh mask chuẩn -> đổi NGUON_MASK = "file"
```

Học lại nền khi: đổi đèn, đụng camera, đổi đồ gá, ánh sáng phòng đổi nhiều.

## Dùng hằng ngày

```bash
python chay_chup_anh.py     # có camera: chụp + vẽ luôn
python chay_doc_file.py     # không camera: vẽ trên ảnh đã chụp
```

Vẽ xong bấm Enter → ảnh tự cắt part, tự vào `PART..\train|val\OK|NG` theo
80:20. Sửa nhãn ảnh cũ thì file part cũ tự bị xoá và ghi lại.

## Đổi tham số

Sửa trong `cauhinh.py` rồi:

```bash
python chay_xuat_lai.py     # dựng lại toàn bộ, KHÔNG phải vẽ lại
```

Ba nhóm hay đổi:

- `TI_LE_BOX_TOI_THIEU` – part chứa dưới bao nhiêu % của box thì vẫn tính OK
- `CHE_DO_CHIA` / `CHIA_THEO` / `SO_PHAN_CHINH` – cách chia part
- `TI_LE_TRAIN` – tỉ lệ train/val

## Cách chia part

N phần chính dọc theo một chiều, cộng N−1 phần **gối** nằm đúng giữa hai
phần chính, để lỗi nhỏ nằm ngay vết cắt không bị xẻ đôi. N=3 → 5 part:
`part1, part2, part3, part12, part23`.

`auto` cắt dọc theo cạnh dài và tự chọn N; `manual` thì tự khai chiều cắt
và số phần.

## Lưu ý khi ghép với script chạy thật

Script inference phải đọc `cau_hinh_part.json` (nằm trong thư mục dataset)
để lấy đúng `ratios` và chiều cắt, đừng hardcode lại. Và nên `import
xu_ly_anh` thay vì copy hàm crop/mask sang — copy hai bản là nguồn gốc của
lỗi lệch hệ toạ độ khó tìm nhất.
