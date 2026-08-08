"""
Duyệt qua từng ảnh trong 1 thư mục, hiển thị lên màn hình để bạn gán nhãn
bằng tay:
    - Nhấn 'a' -> ảnh này là OK, di chuyển vào thư mục OK
    - Nhấn 'r' -> ảnh này là NG, di chuyển vào thư mục NG
    - Nhấn 's' -> bỏ qua ảnh này (không phân loại, giữ nguyên chỗ cũ)
    - Nhấn 'q' -> dừng lại giữa chừng (các ảnh chưa duyệt vẫn còn nguyên
      trong thư mục gốc, chạy lại script sẽ tiếp tục từ đầu các ảnh còn lại)

Cách chạy:
    python 03_gan_nhan_ok_ng.py
"""

import os
import shutil
import cv2

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DUONG_DAN_THU_MUC_ANH = r"D:\TongHop\RTC Technologi\PCB\crop5"
THU_MUC_OK = os.path.join(DUONG_DAN_THU_MUC_ANH, "OK")
THU_MUC_NG = os.path.join(DUONG_DAN_THU_MUC_ANH, "NG")
# =====================================================================

CAC_DUOI_ANH_HOP_LE = (".jpg", ".jpeg", ".png", ".bmp")
DO_RONG_HIEN_THI = 1000   # ảnh lớn quá thì thu nhỏ lại cho vừa màn hình


def resize_de_hien_thi(img, max_width=DO_RONG_HIEN_THI):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def main():
    os.makedirs(THU_MUC_OK, exist_ok=True)
    os.makedirs(THU_MUC_NG, exist_ok=True)

    danh_sach_anh = sorted(
        f for f in os.listdir(DUONG_DAN_THU_MUC_ANH)
        if f.lower().endswith(CAC_DUOI_ANH_HOP_LE)
        and os.path.isfile(os.path.join(DUONG_DAN_THU_MUC_ANH, f))
    )

    if not danh_sach_anh:
        print(f"Không tìm thấy ảnh nào trong: {DUONG_DAN_THU_MUC_ANH}")
        return

    print(f"Tìm thấy {len(danh_sach_anh)} ảnh cần gán nhãn.")
    print("Điều khiển: 'a' = OK | 'r' = NG | 's' = bỏ qua | 'q' = dừng\n")

    so_ok, so_ng, so_bo_qua = 0, 0, 0

    for idx, ten_file in enumerate(danh_sach_anh, start=1):
        duong_dan_goc = os.path.join(DUONG_DAN_THU_MUC_ANH, ten_file)
        img = cv2.imread(duong_dan_goc)
        if img is None:
            print(f"[{idx}/{len(danh_sach_anh)}] Không đọc được ảnh, bỏ qua: {ten_file}")
            continue

        img_show = resize_de_hien_thi(img)
        tieu_de = f"[{idx}/{len(danh_sach_anh)}] {ten_file}  |  a=OK  r=NG  s=Bo qua  q=Dung"
        cv2.imshow(tieu_de, img_show)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                shutil.move(duong_dan_goc, os.path.join(THU_MUC_OK, ten_file))
                print(f"[{idx}/{len(danh_sach_anh)}] {ten_file} -> OK")
                so_ok += 1
                break
            elif key == ord('r'):
                shutil.move(duong_dan_goc, os.path.join(THU_MUC_NG, ten_file))
                print(f"[{idx}/{len(danh_sach_anh)}] {ten_file} -> NG")
                so_ng += 1
                break
            elif key == ord('s'):
                print(f"[{idx}/{len(danh_sach_anh)}] {ten_file} -> Bỏ qua (giữ nguyên)")
                so_bo_qua += 1
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\nĐã dừng giữa chừng. OK: {so_ok} | NG: {so_ng} | Bỏ qua: {so_bo_qua} "
                      f"| Còn lại chưa duyệt: {len(danh_sach_anh) - idx}")
                return
            # phím khác -> chờ tiếp, không làm gì

        cv2.destroyWindow(tieu_de)

    cv2.destroyAllWindows()
    print(f"\nHoàn tất. OK: {so_ok} | NG: {so_ng} | Bỏ qua: {so_bo_qua}")
    print(f"Ảnh OK -> {THU_MUC_OK}")
    print(f"Ảnh NG -> {THU_MUC_NG}")


if __name__ == "__main__":
    main()