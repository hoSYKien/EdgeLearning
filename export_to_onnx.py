"""
Script chuyển đổi model .pt + OOD sang 1 file ONNX DUY NHẤT (All-In-One ONNX)
Model ONNX xuất ra sẽ tự động trả về:
1. logits         : Xác suất phân loại
2. pred_class_id  : ID class dự đoán
3. ood_score      : Điểm tương đồng (Cosine similarity)
4. is_unknown     : 1 nếu là Dị vật/Lạ, 0 nếu là Hàng chuẩn
"""
import os
import glob
import json
import traceback
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ROOT_MODEL_DIR = r"D:\TongHop\RTC Technologi\G8\modelClassifierOOD\runs\20260817_233900"


class UnifiedClassifierOODModel(nn.Module):
    """Gộp toàn bộ Backbone + Head Classifier + OOD Gate vào 1 đồ thị nơ-ron duy nhất"""
    def __init__(self, full_model, centroids, threshold):
        super().__init__()
        self.features = full_model.features
        self.classifier = full_model.classifier
        
        # Đưa centroids và threshold vào buffer đồ thị ONNX
        self.register_buffer("centroids", torch.tensor(centroids, dtype=torch.float32))  # [num_classes, embedding_dim]
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))

    def forward(self, x):
        # 1. Trích xuất đặc trưng
        feat = self.features(x)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        emb = torch.flatten(pooled, 1)  # [B, D]

        # 2. Phân loại
        logits = self.classifier(emb)  # [B, num_classes]
        probs = F.softmax(logits, dim=1)
        pred_class_id = torch.argmax(probs, dim=1)  # [B]

        # 3. Tính điểm OOD Cosine Similarity
        emb_norm = emb / (torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_sims = torch.matmul(emb_norm, self.centroids.t())  # [B, num_classes]
        ood_score, _ = torch.max(cosine_sims, dim=1)  # [B]

        # 4. Kiểm tra Dị vật (1 = Dị vật/Unknown, 0 = Hàng chuẩn)
        is_unknown = (ood_score < self.threshold).to(torch.int32)  # [B]

        return logits, pred_class_id, ood_score, is_unknown


def find_latest_file(root_dir, pattern):
    files = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
    if not files:
        files = glob.glob(os.path.join(root_dir, pattern))
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    return None


def build_model_architecture(backbone_name, num_classes):
    if backbone_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(960, num_classes))
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(576, num_classes))
    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone_name}")
    return model


def export_unified_onnx():
    # 1. Tìm file .pt và .joblib mới nhất
    pt_path = find_latest_file(ROOT_MODEL_DIR, "*.pt")
    joblib_path = find_latest_file(ROOT_MODEL_DIR, "*.joblib")

    if not pt_path or not os.path.exists(pt_path):
        print(f"[!] Không tìm thấy file model .pt trong: {ROOT_MODEL_DIR}")
        return

    print(f"-> Nạp PyTorch Model : {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    backbone_name = ckpt.get("backbone_name", "mobilenet_v3_large")
    num_classes = len(class_names)

    # Nạp OOD parameters
    if joblib_path and os.path.exists(joblib_path):
        print(f"-> Nạp OOD Detector  : {joblib_path}")
        ood_data = joblib.load(joblib_path)
        centroids = ood_data["centroids"]
        threshold = ood_data["threshold"]
    else:
        print(f"[!] Không tìm thấy file .joblib, sử dụng ngưỡng mặc định 0.75")
        centroids = np.zeros((num_classes, ckpt.get("embedding_dim", 960)), dtype=np.float32)
        threshold = 0.75

    # 2. Xây dựng mô hình gộp
    base_model = build_model_architecture(backbone_name, num_classes)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()

    unified_model = UnifiedClassifierOODModel(base_model, centroids, threshold).eval()

    # 3. Xuất file ONNX
    onnx_path = os.path.join(ROOT_MODEL_DIR, f"edge_classifier_unified_{backbone_name}.onnx")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"\nĐang xuất Unified All-In-One ONNX ({num_classes} classes)...")
    try:
        try:
            torch.onnx.export(
                unified_model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["logits", "pred_class_id", "ood_score", "is_unknown"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                    "pred_class_id": {0: "batch_size"},
                    "ood_score": {0: "batch_size"},
                    "is_unknown": {0: "batch_size"},
                },
                opset_version=13,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                unified_model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["logits", "pred_class_id", "ood_score", "is_unknown"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                    "pred_class_id": {0: "batch_size"},
                    "ood_score": {0: "batch_size"},
                    "is_unknown": {0: "batch_size"},
                },
                opset_version=13,
                do_constant_folding=True,
            )

        # Lưu kèm 1 file nhãn class_labels.json nhỏ để app C++/C#/Python đọc tên class
        labels_json_path = os.path.join(ROOT_MODEL_DIR, "class_labels.json")
        with open(labels_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "class_names": class_names,
                "num_classes": num_classes,
                "threshold": float(threshold),
                "backbone": backbone_name,
            }, f, ensure_ascii=False, indent=2)

        print("=" * 65)
        print(f"✅ XUẤT ALL-IN-ONE ONNX THÀNH CÔNG RỰC RỠ!")
        print(f"📁 File ONNX  : {onnx_path}")
        print(f"📁 File Nhãn  : {labels_json_path}")
        print(f"📦 Dung lượng : {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        print("🎯 Đầu ra ONNX : ['logits', 'pred_class_id', 'ood_score', 'is_unknown']")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"❌ Lỗi khi xuất ONNX: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    export_unified_onnx()
