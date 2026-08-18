r"""
giao_dien.py - Cửa sổ cho người vận hành vẽ ROI lỗi lên ảnh full.

Hiện ranh giới các part theo thời gian thực, đổi màu đỏ/xanh ngay khi vẽ,
kèm bảng ghi rõ mỗi part đang OK hay NG và box chiếm bao nhiêu %.

PHÍM TẮT:
    Kéo chuột trái : vẽ 1 box lỗi
    Chuột phải     : xoá box nằm dưới con trỏ
    u              : undo box vừa vẽ
    c              : xoá hết box của ảnh này
    Enter / Space  : LƯU + CẮT + CHIA VÀO DATASET, sang ảnh kế
    n / b          : sang ảnh kế / quay lại ảnh trước
    x              : BỎ QUA ảnh này (nếu đã xuất thì gỡ khỏi dataset)
    r              : dựng lại toàn bộ dataset
    q / ESC        : lưu rồi thoát

Chạy riêng để thử vẽ trên 1 ảnh (KHÔNG ghi gì vào dataset):
    python giao_dien.py anh_full.png
"""

import sys

import cv2
import numpy as np

import cauhinh as cf
import chia_part as cp
import kho_dulieu as kd

TEN_CUA_SO = ("Ve khoanh vung loi - Enter: luu & chia dataset | u: undo | "
              "c: xoa het | x: bo qua | r: dung lai | q: thoat")

MAU_NG = (0, 0, 255)
MAU_OK = (0, 200, 0)
MAU_BOX = (0, 165, 255)


class BoAnhVe:
    """Giữ trạng thái chuột + danh sách box (toạ độ theo ẢNH GỐC)."""

    def __init__(self):
        self.dang_ve = False
        self.diem_dau = (0, 0)
        self.diem_hien_tai = (0, 0)
        self.boxes = []
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
            if abs(x - x0) >= cf.BOX_TOI_THIEU_PX and abs(y - y0) >= cf.BOX_TOI_THIEU_PX:
                bx1, bx2 = sorted((x0, x))
                by1, by2 = sorted((y0, y))
                s = self.scale
                self.boxes.append([int(bx1 / s), int(by1 / s),
                                   int(bx2 / s), int(by2 / s)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.xoa_yeu_cau = (x / self.scale, y / self.scale)

    def xu_ly_xoa(self):
        if self.xoa_yeu_cau is None:
            return
        px, py = self.xoa_yeu_cau
        self.xoa_yeu_cau = None
        for k in range(len(self.boxes) - 1, -1, -1):
            bx1, by1, bx2, by2 = self.boxes[k]
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                self.boxes.pop(k)
                return


def ve_khung(anh_nho, state, w_goc, h_goc, tieu_de, bo_qua):
    """Ảnh + ranh giới part + box + bảng trạng thái."""
    ra = anh_nho.copy()
    s = state.scale
    H, W = ra.shape[:2]
    kq = cp.nhan_cac_part(state.boxes, w_goc, h_goc)

    # phần chính vẽ nửa này, phần gối vẽ nửa kia cho đỡ rối mắt
    for i, (ten, (x0, y0, x1, y1), nhan, _) in enumerate(kq):
        mau = MAU_NG if nhan == "NG" else MAU_OK
        chinh = i < cp.SO_PHAN
        if cp.CHIEU == "ngang":
            X0, X1 = int(x0 * s), max(0, int(x1 * s) - 1)
            y_dau = 0 if chinh else H // 2
            cv2.line(ra, (X0, y_dau), (X0, H), mau, 1)
            cv2.line(ra, (X1, y_dau), (X1, H), mau, 1)
            cv2.putText(ra, f"{ten}:{nhan}", (X0 + 6, 22 if chinh else H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)
        else:
            Y0, Y1 = int(y0 * s), max(0, int(y1 * s) - 1)
            x_dau = 0 if chinh else W // 2
            cv2.line(ra, (x_dau, Y0), (W, Y0), mau, 1)
            cv2.line(ra, (x_dau, Y1), (W, Y1), mau, 1)
            cv2.putText(ra, f"{ten}:{nhan}",
                        (10 if chinh else W // 2 + 10, Y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)

    for b in state.boxes:
        cv2.rectangle(ra, (int(b[0] * s), int(b[1] * s)),
                      (int(b[2] * s), int(b[3] * s)), MAU_BOX, 2)
    if state.dang_ve:
        cv2.rectangle(ra, state.diem_dau, state.diem_hien_tai, (255, 255, 0), 1)
    if bo_qua:
        cv2.putText(ra, "BO QUA", (W // 2 - 90, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    moi_dong = max(1, W // 230)
    so_dong = (len(kq) + moi_dong - 1) // moi_dong
    bang = np.zeros((34 + 26 * so_dong, W, 3), np.uint8)
    cv2.putText(bang, f"{tieu_de}   |   {len(state.boxes)} box   |   chia "
                      f"{cp.CHIEU} {cp.SO_PHAN} phan   |   nguong "
                      f"{cf.TI_LE_BOX_TOI_THIEU:.2f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    for i, (ten, _, nhan, tl) in enumerate(kq):
        mau = MAU_NG if nhan == "NG" else MAU_OK
        cv2.putText(bang, f"{ten}: {nhan} ({tl * 100:4.0f}%)",
                    (10 + (i % moi_dong) * 230, 48 + (i // moi_dong) * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mau, 2)
    return np.vstack([ra, bang])


def ve_roi_mot_anh(img, ten_file, nhan_all, dem, tieu_de,
                   cho_dieu_huong=True, ghi_dataset=True):
    """Hiện 1 ảnh cho vẽ ROI. Enter = lưu nhãn + cắt part + chia dataset.

    Trả về: "tiep" | "bo" | "truoc" | "dung_lai" | "thoat"."""
    state = BoAnhVe()
    h_goc, w_goc = img.shape[:2]
    state.scale = min(1.0, cf.MAX_HIEN_THI_W / w_goc, cf.MAX_HIEN_THI_H / h_goc)
    anh_nho = cv2.resize(img, (int(w_goc * state.scale), int(h_goc * state.scale))) \
        if state.scale < 1.0 else img.copy()

    ghi_chu = nhan_all.get(ten_file, {})
    state.boxes = [list(b) for b in ghi_chu.get("boxes", [])]
    bo_qua = bool(ghi_chu.get("bo_qua", False))

    cv2.namedWindow(TEN_CUA_SO, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(TEN_CUA_SO, state.chuot)

    def luu_va_chia():
        cu = nhan_all.get(ten_file, {})
        nhan_all[ten_file] = {"w": w_goc, "h": h_goc, "boxes": state.boxes,
                              "bo_qua": bo_qua, "xuat": cu.get("xuat", {})}
        if ghi_dataset:
            print(f"  {ten_file}  ->  "
                  f"{kd.xuat_mot_anh(img, ten_file, nhan_all, dem)}")

    while True:
        state.xu_ly_xoa()
        cv2.imshow(TEN_CUA_SO, ve_khung(anh_nho, state, w_goc, h_goc,
                                        tieu_de, bo_qua))
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
            return "dung_lai"
        elif phim in (ord('q'), 27):
            luu_va_chia()
            return "thoat"
        elif cho_dieu_huong and phim == ord('n'):
            return "bo"
        elif cho_dieu_huong and phim == ord('b'):
            return "truoc"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python giao_dien.py anh_full.png")
        print("(thử vẽ ROI, KHÔNG ghi gì vào dataset)")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Không đọc được ảnh: {sys.argv[1]}")
        raise SystemExit(1)

    cp.thiet_lap(img.shape[1], img.shape[0])
    print("\n*** CHẾ ĐỘ THỬ - không ghi dataset, không ghi file nhãn ***")
    nhan_all = {}
    ve_roi_mot_anh(img, "thu.png", nhan_all, {}, "THU VE ROI",
                   cho_dieu_huong=False, ghi_dataset=False)
    cv2.destroyAllWindows()

    boxes = nhan_all.get("thu.png", {}).get("boxes", [])
    print(f"\nĐã vẽ {len(boxes)} box: {boxes}")
    for ten, _, nhan, tl in cp.nhan_cac_part(boxes, img.shape[1], img.shape[0]):
        print(f"   {ten:8s} {nhan}  ({tl * 100:.0f}% cua box)")
