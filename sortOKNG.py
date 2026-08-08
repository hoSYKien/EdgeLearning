"""
DUYET QUA TUNG ANH TRONG 1 THU MUC, BAM PHIM DE PHAN LOAI VAO OK / NG.

Phim tat:
  O (hoac o)  -> chuyen anh vao thu muc OK
  N (hoac n)  -> chuyen anh vao thu muc NG
  S (hoac s)  -> bo qua, khong lam gi, sang anh tiep theo
  ESC         -> dung han

Cach dung:
    python sort_ok_ng.py
"""

import os
import shutil

import cv2

# ==========================================================================
# CONFIG
# ==========================================================================

INPUT_DIR = r"D:\TongHop\RTC Technologi\PCB_Candidates_crop"

OK_DIR = os.path.join(INPUT_DIR, "OK")
NG_DIR = os.path.join(INPUT_DIR, "NG")

DISPLAY_MAX_WIDTH = 900   # thu nho anh hien thi neu anh goc qua lon


def resize_for_display(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale*0.5), int(h * scale*0.5)))


def main():
    os.makedirs(OK_DIR, exist_ok=True)
    os.makedirs(NG_DIR, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    # Chi lay anh nam TRUC TIEP trong INPUT_DIR (khong lay lai anh da nam
    # trong thu muc con OK/ hoac NG/ tu lan chay truoc)
    files = [f for f in sorted(os.listdir(INPUT_DIR))
             if f.lower().endswith(exts) and os.path.isfile(os.path.join(INPUT_DIR, f))]

    if not files:
        print(f"Khong tim thay anh nao truc tiep trong: {INPUT_DIR}")
        return

    print(f"Tim thay {len(files)} anh. Phim tat: O = OK | N = NG | S = bo qua | ESC = dung\n")

    n_ok, n_ng, n_skip = 0, 0, 0
    for i, fname in enumerate(files):
        src_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f"  [BO QUA] Loi doc anh: {fname}")
            continue

        disp = resize_for_display(img, DISPLAY_MAX_WIDTH)
        text = f"[{i+1}/{len(files)}] {fname}   O=OK  N=NG  S=bo qua  ESC=dung"
        cv2.putText(disp, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(disp, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Phan loai OK / NG", disp)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key in (ord('a'), ord('A')):
            shutil.move(src_path, os.path.join(OK_DIR, fname))
            n_ok += 1
            print(f"  {fname} -> OK")
        elif key in (ord('d'), ord('D')):
            shutil.move(src_path, os.path.join(NG_DIR, fname))
            n_ng += 1
            print(f"  {fname} -> NG")
        elif key == 27:   # ESC
            print("\nDa nhan ESC - dung som.")
            break
        else:
            n_skip += 1
            print(f"  {fname} -> bo qua")

    print(f"\nXong. OK: {n_ok} | NG: {n_ng} | Bo qua: {n_skip}")


if __name__ == "__main__":
    main()