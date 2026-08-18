# -*- coding: utf-8 -*-
"""
CẤU HÌNH HỆ THỐNG ĐỒNG BỘ ĐA CAMERA 2D (MULTICAMERA 2.0)
"""
import os

# Đường dẫn thư mục gốc của module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# CẤU HÌNH CAMERA & MẠNG GIGE
# -------------------------------------------------------------
TARGET_FPS = 10.0            # Tốc độ khung hình (10.0 FPS chuẩn cho 2-4 camera 5MP)
GEV_PACKET_SIZE = 1500       # MTU chuẩn Ethernet (1500 bytes)
GEV_PACKET_DELAY = 1200      # Độ trễ giữa các packet (ticks) chống nghẽn Switch
IMAGE_NODE_NUM = 15          # Buffer frame trên RAM máy tính
MAX_EXPOSURE_TIME = 80000.0  # Giới hạn Exposure Time (us) để không bị tụt FPS

MAX_CAMERAS = 4              # Hỗ trợ tối đa 4 camera đồng thời

# -------------------------------------------------------------
# CHẾ ĐỘ MASTER - SLAVE (CAM 1 DETECT -> CHIẾU SANG CÁC CAM SAU)
# -------------------------------------------------------------
MASTER_CAM = "cam1"          # Camera chính (Color): Phát hiện vật, mask, đọc barcode và đánh ID
DRAW_QUAD_POLYGON = True     # Vẽ cả khung đa giác 4 góc xoay thực tế khi chiếu sang Cam 2/3/4

# -------------------------------------------------------------
# CẤU HÌNH ĐỒNG BỘ ID & KHÔNG GIAN MẶT BÀN
# -------------------------------------------------------------
MERGE_DIST_CM = 8.0          # Khoảng cách tối đa (cm) để gộp vật thể giữa các camera
TRACK_DIST_CM = 15.0         # Khoảng cách tối đa (cm) để duy trì ID qua các frame
FORGET_AFTER_SEC = 1.5       # Thời gian (giây) xóa vật thể nếu mất dấu

# File lưu ma trận Homography
HOMOGRAPHY_FILE = os.path.join(BASE_DIR, "homographies.json")

# Thư mục lưu ảnh snapshot
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

# -------------------------------------------------------------
# MỐC TỌA ĐỘ VẬT LÝ MẶT BÀN KHI HIỆU CHUẨN (CALIBRATE 2D)
# -------------------------------------------------------------
# Mặc định hình chữ nhật 25cm (ngang) x 20cm (dọc)
# Thứ tự click chuột: Trên-Trái -> Trên-Phải -> Dưới-Phải -> Dưới-Trái
CALIB_TABLE_POINTS = [
    (0.0, 0.0),     # 1: Trên - Trái (Gốc tọa độ O)
    (25.0, 0.0),    # 2: Trên - Phải
    (25.0, 20.0),   # 3: Dưới - Phải
    (0.0, 20.0),    # 4: Dưới - Trái
]

# -------------------------------------------------------------
# CẤU HÌNH XỬ LÝ ẢNH & BARCODE (CHỈ CHẠY TRÊN MASTER CAM)
# -------------------------------------------------------------
RESIZE_DIV = 4               # Tỉ lệ thu nhỏ khi trừ nền để tăng tốc độ
BG_THRESH = 30               # Ngưỡng trừ nền
MIN_AREA_RATIO = 0.002       # Diện tích vật tối thiểu so với khung hình
MAX_AREA_RATIO = 0.90        # Diện tích vật tối đa so với khung hình
