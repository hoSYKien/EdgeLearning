"""
Train model PHÁT HIỆN BẤT THƯỜNG (anomaly detection) bằng PatchCore
(thư viện anomalib) - CHỈ CẦN ảnh OK để học, không cần ảnh NG để train
(khác hẳn classifier cũ - đây là hướng "one-class").

Cài đặt trước khi chạy (chỉ cần làm 1 lần):
    pip install anomalib

CẤU TRÚC THƯ MỤC DATASET CẦN CHUẨN BỊ:

    D:\TongHop\RTC Technologi\PCB\anomalib_dataset\
        train\
            OK\              <- CHỈ ảnh OK, càng nhiều càng tốt (model chỉ học từ đây)
        test\
            OK\               <- 1 ít ảnh OK để đánh giá (không dùng để train)
            NG\               <- 1 ít ảnh NG để đánh giá (không dùng để train)

Cách chạy:
    python train_anomalib_patchcore.py

GHI CHÚ VỀ HIỆN TƯỢNG "TREO" (xem thêm cuối file):
    Bước rút gọn memory bank (coreset subsampling, chạy tự động cuối
    engine.fit()) tính khoảng cách giữa RẤT NHIỀU vector đặc trưng ->
    trên CPU có thể mất vài chục phút mà KHÔNG in log gì thêm, trông
    giống hệt bị treo. Mở Task Manager kiểm tra python.exe có đang chiếm
    CPU cao không trước khi kết luận là treo thật.
"""

import os
import warnings
import logging

# Tắt cảnh báo không cần thiết từ thư viện bên thứ ba
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

# ====================== CHỈNH ĐƯỜNG DẪN Ở ĐÂY ======================
DATASET_ROOT = r"D:\TongHop\RTC Technologi\PCB\crop5"
KET_QUA_DIR = r"D:\TongHop\RTC Technologi\PCB\crop5"
# =====================================================================

IMAGE_SIZE = (256, 256)   # PatchCore mặc định hay dùng 256x256, có thể tăng nếu lỗi rất nhỏ

# Giảm/tăng để đánh đổi tốc độ train (bước coreset subsampling) với độ chi
# tiết của memory bank. Nếu máy bạn CPU yếu hoặc dataset OK nhiều ảnh,
# GIẢM giá trị này (vd 0.05 hoặc thấp hơn) để bước rút gọn memory bank
# chạy nhanh hơn nhiều - đây thường là bước "trông như treo" nhất.
CORESET_SAMPLING_RATIO = 0.1

# Trên Windows, num_workers > 0 dùng multiprocessing kiểu 'spawn' - rất dễ
# treo khi chạy trong IDE nhúng (vd Antigravity, VSCode debug console...).
# Để 0 cho an toàn (chậm hơn 1 chút khi load ảnh nhưng ổn định hơn nhiều).
SO_LUONG_WORKER = 0

# =====================================================================


def main():
    print("Đang khởi tạo datamodule từ thư mục:", DATASET_ROOT)

    datamodule = Folder(
        name="pcb",
        root=DATASET_ROOT,
        normal_dir=os.path.join("train", "OK"),
        abnormal_dir=os.path.join("test", "NG"),
        normal_test_dir=os.path.join("test", "OK"),
        # normal_split_ratio=0 vì đã có sẵn OK riêng cho test, không cần
        # tự động trích thêm từ tập train.
        normal_split_ratio=0,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=SO_LUONG_WORKER,
        # LƯU Ý: bản anomalib đang cài KHÔNG nhận image_size ở đây (khác
        # với các datamodule có sẵn như MVTecAD). Kích thước ảnh được
        # cấu hình qua pre_processor của MODEL bên dưới thay vì ở đây.
    )

    print("Đang khởi tạo model PatchCore...")
    # Cấu hình pre_processor với đúng IMAGE_SIZE mong muốn - đây là cách
    # CHÍNH THỨC được anomalib khuyến nghị để chỉnh kích thước resize/crop
    # đầu vào (thay vì set ở datamodule).
    pre_processor = Patchcore.configure_pre_processor(image_size=IMAGE_SIZE)
    model = Patchcore(
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        pre_processor=pre_processor,
    )

    # Chỉ định rõ accelerator/devices để tránh Lightning tự "đoán" và có
    # thể bị treo khi khởi tạo CUDA context nếu driver/CUDA không khớp.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Sẽ train trên: {accelerator.upper()}")

    engine = Engine(
        default_root_dir=KET_QUA_DIR,
        accelerator=accelerator,
        devices=1,
    )

    print("\nBắt đầu train (PatchCore không train mạng nơ-ron, chỉ trích "
          "đặc trưng ảnh OK để xây 'memory bank' - thường NHANH hơn nhiều "
          "so với train classifier)...\n")
    print("Lưu ý: sau khi trích xong đặc trưng, sẽ có bước RÚT GỌN memory "
          "bank (coreset subsampling) - bước này có thể chạy lâu và KHÔNG "
          "in thêm log nào, đừng vội tưởng bị treo, hãy kiểm tra CPU usage.\n")
    engine.fit(datamodule=datamodule, model=model)

    print("\nĐang đánh giá trên tập test (ảnh OK + NG chưa từng dùng để train)...")
    test_results = engine.test(datamodule=datamodule, model=model)
    print("\nKết quả đánh giá:")
    for r in test_results:
        for k, v in r.items():
            print(f"  {k}: {v}")

    print(f"\nĐã train xong. Checkpoint + log được lưu trong: {KET_QUA_DIR}")
    print("Tìm file .ckpt trong thư mục con 'weights/lightning/' để dùng cho bước dự đoán.")


if __name__ == "__main__":
    main()