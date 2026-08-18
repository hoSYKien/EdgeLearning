# 📷 HỆ THỐNG ĐỒNG BỘ ĐA CAMERA 2D (MULTICAMERA 2.0)
> **Kiến trúc Master-Slave: Cam 1 (Color) Mask/Phát hiện vật & Đọc Barcode $\rightarrow$ Chiếu Bounding Box & Đồng bộ ID sang Cam 2, 3, 4!**

---

## 🌟 1. Cơ chế Master - Slave 2D

```
  [ CAMERA 1: MASTER (Color) ]
  - Chịu trách nhiệm Mask / Trừ nền phát hiện vật thể
  - Trích xuất Bounding Box [x1, y1, x2, y2]
  - Quét mã vạch (Barcode / QR) từ ảnh màu nét
  - Đánh và duy trì ID ổn định (ID: 1, ID: 2...)
              │
              │  Chiếu tọa độ qua Homography: H_{1 -> i} = inv(H_i) @ H_1
              ▼
  ┌──────────────────────────────────────────────────────────┐
  │  CÁC CAMERA PHỤ (SLAVE): CAM 2 (Mono), CAM 3, CAM 4      │
  │  - KHÔNG CẦN TỰ TRỪ NỀN (Khử sạch 100% nhiễu nền/bóng)   │
  │  - Tự động nhận Bounding Box / Đa giác góc xoay từ Cam 1 │
  │  - Tự động mang đúng [ID] và [Barcode] từ Cam 1          │
  └──────────────────────────────────────────────────────────┘
```

---

## 📁 2. Cấu trúc thư mục

| File | Chức năng |
| :--- | :--- |
| **`config.py`** | Cấu hình: `MASTER_CAM = "cam1"`, `TARGET_FPS = 10.0`, mốc Calibrate 4 điểm mặt bàn. |
| **`camera_driver.py`** | Driver camera GigE/USB: MTU 1500, Delay 1200 ticks, giải mã BGR chuẩn SDK, không xước ảnh. |
| **`detector.py`** | Thuật toán trừ nền động & Quét barcode siêu tốc (zxing-cpp / pyzbar) cho Master Camera. |
| **`coordinator.py`** | Bộ điều phối chiếu Bbox và ID từ Cam 1 sang Cam 2/3/4 qua ma trận $H_{1 \to i}$. |
| **`calibrate.py`** | Module hiệu chuẩn 4 điểm mốc mặt bàn để tính ma trận Homography và lưu `homographies.json`. |
| **`run_multicam.py`** | **File chạy chính**: Khởi động hệ thống Master-Slave, hiển thị Grid View kèm Box & ID đồng bộ. |
| **`test_cameras.py`** | Tool chẩn đoán kết nối và đo FPS thực tế của tất cả camera. |

---

## 🚀 3. Hướng dẫn sử dụng

### 🔹 Bước 1: Hiệu chuẩn Homography 2D cho các camera
```powershell
python multiCamera2/calibrate.py
```
- Click lần lượt 4 điểm mốc trên `cam1` $\rightarrow$ Nhấn **`s`** để lưu.
- Click tiếp 4 điểm mốc tương ứng trên `cam2` $\rightarrow$ Nhấn **`s`** để lưu.
- Ma trận sẽ tự động lưu vào `homographies.json`.

---

### 🔹 Bước 2: Chạy hệ thống đồng bộ
```powershell
python multiCamera2/run_multicam.py
```
- Trong 3 giây đầu: **Giữ mặt bàn TRỐNG** để Cam 1 học nền.
- Đặt sản phẩm lên bàn:
  - **Cam 1** phát hiện vật, vẽ khung và đọc mã vạch `ID: 1`.
  - **Cam 2 (Mono) và các camera sau** ngay lập tức hiện khung Bounding Box / Đa giác ôm trọn sản phẩm, mang đúng nhãn `[SYNC] ID: 1` và mã vạch đọc được từ Cam 1!

---

## ⌨️ 4. Bảng phím tắt điều khiển

| Phím tắt | Chức năng |
| :---: | :--- |
| **`c`** | **Calibrate 2D**: Mở chế độ hiệu chuẩn Homography trực tiếp. |
| **`r`** | **Học lại nền**: Học lại nền cho Cam 1 khi ánh sáng thay đổi. |
| **`s`** | **Snapshot**: Lưu ảnh đồng thời từ tất cả camera vào `captures/`. |
| **`g`** | **Grid View**: Đổi chế độ xem Ghép màn hình $\leftrightarrow$ Cửa sổ riêng. |
| **`q`** / **`ESC`** | **Thoát an toàn**: Đóng camera và giải phóng tài nguyên. |
