"""
Chạy MobileNetV3 pretrained ImageNet qua camera Intel RealSense D435I,
chỉ nhận diện trong 1 vùng ROI do người dùng tự chọn bằng chuột.

Cài thư viện cần thiết trước khi chạy (nếu chưa có):
    python -m pip install pyrealsense2 opencv-python

Cách dùng:
    1. Cửa sổ chọn ROI hiện lên đầu tiên -> kéo chuột vẽ khung quanh vật cần nhận diện
    2. Nhấn ENTER hoặc SPACE để xác nhận
    3. Cửa sổ chính hiện lên, chỉ nhận diện trong đúng khung đã chọn
    4. Nhấn 'q' để thoát, nhấn 'r' để chọn lại ROI mới
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from torchvision import models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

TOP_K = 3

# ----------------------------
# 1. Load model pretrained ImageNet
# ----------------------------
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)
model = model.to(DEVICE)
model.eval()

class_names = weights.meta["categories"]
transform = weights.transforms()
print(f"Model đã sẵn sàng, nhận diện được {len(class_names)} loại vật thể khác nhau.")

# ----------------------------
# 2. Hàm dự đoán 1 vùng ảnh (crop)
# ----------------------------
def predict_crop(crop_bgr):
    frame_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, TOP_K)
    return [(class_names[idx], prob.item()) for idx, prob in zip(top_idxs, top_probs)]

# ----------------------------
# 3. Khởi động camera RealSense (chỉ luồng màu)
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
except Exception as e:
    print(f"Không khởi động được camera RealSense: {e}")
    print("Kiểm tra: RealSense Viewer có đang mở và chiếm camera không? Đóng app đó rồi thử lại.")
    exit()

print("Đã kết nối RealSense D435I.")

def get_frame():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())

# ----------------------------
# 4. Cho người dùng chọn ROI bằng chuột
# ----------------------------
def select_roi():
    print("\nHãy kéo chuột vẽ khung quanh vật cần nhận diện, rồi nhấn ENTER/SPACE.")
    frame = None
    for _ in range(30):  # thử vài lần lấy frame đầu tiên cho ổn định
        frame = get_frame()
        if frame is not None:
            break
    if frame is None:
        return None
    roi = cv2.selectROI("Chon ROI - keo chuot roi nhan ENTER", frame, showCrosshair=True)
    cv2.destroyWindow("Chon ROI - keo chuot roi nhan ENTER")
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)


roi = select_roi()
if roi is None:
    print("Chưa chọn ROI hợp lệ, thoát chương trình.")
    pipeline.stop()
    exit()

print(f"ROI đã chọn: {roi}")
print("Đang chạy... Nhấn 'q' để thoát, 'r' để chọn lại ROI.")

# ----------------------------
# 5. Vòng lặp chính - chỉ classify trong ROI
# ----------------------------
try:
    while True:
        frame = get_frame()
        if frame is None:
            continue

        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]

        results = predict_crop(crop) if crop.size > 0 else []

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_start_y = max(y - 10 - (len(results) - 1) * 25, 20)
        for i, (label, prob) in enumerate(results):
            text = f"{label}: {prob:.1%}"
            text_y = text_start_y + i * 25
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("MobileNetV3 - RealSense ROI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            new_roi = select_roi()
            if new_roi is not None:
                roi = new_roi
                print(f"ROI mới: {roi}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()