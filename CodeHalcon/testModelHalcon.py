"""
Script chạy inference TỪNG ẢNH riêng lẻ:
  - In ra class dự đoán + confidence score
  - Sinh heatmap Grad-CAM (vùng ảnh model "chú ý" khi ra quyết định)
  - Lưu ảnh heatmap overlay ra thư mục output

Yêu cầu: HALCON 25.11 (native) + pip install mvtec-halcon==25110.0.0
"""

import halcon as ha
import os

# ============================================================
# CẤU HÌNH - sửa lại theo project của bạn
# ============================================================
MODEL_PATH = r"D:\TongHop\RTC Technologi\PCB\model\model12\New Project\Training-260731-160153\best_model.hdl"
PREPROCESS_PARAM_PATH = r"D:\TongHop\RTC Technologi\PCB\model\model12\New Project\Training-260731-160153\dl_preprocess_param.hdict"
IMAGE_DIR = r"D:\TongHop\RTC Technologi\PCB\crop5\val"
OUTPUT_DIR = r"D:\TongHop\RTC Technologi\PCB\crop5\heatmap_results"


# ============================================================
# HELPER: gọi HDevelop procedure an toàn - tự báo tên tham số đúng nếu sai
# ============================================================
def call_proc(proc_name, _inspect=False, _iconic=None, **kwargs):
    """
    _iconic: dict các tham số ẢNH (iconic), ví dụ {"Images": image_object}
             Khác với kwargs (**kwargs) vốn chỉ dành cho tham số control (số/text/handle).
    """
    proc = ha.HDevProcedure.load_external(proc_name)
    input_names = list(proc.input_control_param_names)
    output_names = list(proc.output_control_param_names)
    iconic_input_names = list(proc.input_iconic_param_names)
    iconic_output_names = list(proc.output_iconic_param_names)

    if _inspect:
        print(f"[INSPECT] '{proc_name}'")
        print(f"    CONTROL INPUT  : {input_names}")
        print(f"    ICONIC  INPUT  : {iconic_input_names}")
        print(f"    CONTROL OUTPUT : {output_names}")
        print(f"    ICONIC  OUTPUT : {iconic_output_names}")
        return None

    _iconic = _iconic or {}

    unknown = [k for k in kwargs if k not in input_names]
    if unknown:
        raise ValueError(
            f"[call_proc] Procedure '{proc_name}' không có control-param {unknown}.\n"
            f"  => CONTROL INPUT hợp lệ: {input_names}\n"
            f"  => ICONIC INPUT hợp lệ : {iconic_input_names}"
        )
    unknown_iconic = [k for k in _iconic if k not in iconic_input_names]
    if unknown_iconic:
        raise ValueError(
            f"[call_proc] Procedure '{proc_name}' không có iconic-param {unknown_iconic}.\n"
            f"  => ICONIC INPUT hợp lệ : {iconic_input_names}"
        )

    proc_call = ha.HDevProcedureCall(proc)
    for k, v in kwargs.items():
        proc_call.set_input_control_param_by_name(k, v)
    for k, v in _iconic.items():
        proc_call.set_input_iconic_param_by_name(k, v)
    proc_call.execute()

    result = {name: proc_call.get_output_control_param_by_name(name) for name in output_names}
    for name in iconic_output_names:
        result[name] = proc_call.get_output_iconic_param_by_name(name)
    return result


def colorize_heatmap(heatmap_real, width, height):
    """
    Chuyển heatmap xám (giá trị 0-1) thành ảnh MÀU kiểu jet colormap
    (xanh dương -> xanh lá -> vàng -> đỏ), dùng công thức xấp xỉ chuẩn.
    Trả về ảnh màu byte (RGB), đã resize khớp kích thước width x height.
    """
    # Resize heatmap khớp kích thước ảnh gốc
    heat_resized = ha.zoom_image_size(heatmap_real, width, height, "constant")

    # Công thức xấp xỉ jet colormap: v trong [0,1]
    #   r = clip(1.5 - |4v - 3|, 0, 1)
    #   g = clip(1.5 - |4v - 2|, 0, 1)
    #   b = clip(1.5 - |4v - 1|, 0, 1)
    zero_img = ha.gen_image_const("real", width, height)
    zero_img = ha.scale_image(zero_img, 0, 0)
    one_img = ha.scale_image(zero_img, 0, 1)

    def channel(v, center):
        tmp = ha.scale_image(v, 4, -center)      # 4v - center
        tmp = ha.abs_image(tmp)                   # |4v - center|
        tmp = ha.scale_image(tmp, -1, 1.5)         # 1.5 - |4v - center|
        tmp = ha.min_image(tmp, one_img)           # clip trên = 1
        tmp = ha.max_image(tmp, zero_img)          # clip dưới = 0
        return tmp

    r = channel(heat_resized, 3)
    g = channel(heat_resized, 2)
    b = channel(heat_resized, 1)

    r_byte = ha.convert_image_type(ha.scale_image(r, 255, 0), "byte")
    g_byte = ha.convert_image_type(ha.scale_image(g, 255, 0), "byte")
    b_byte = ha.convert_image_type(ha.scale_image(b, 255, 0), "byte")

    heat_color = ha.compose3(r_byte, g_byte, b_byte)
    return heat_color


def overlay_heatmap_on_image(orig_image, heat_color_byte, alpha=0.45):
    """Chồng heatmap màu lên ảnh gốc với độ trong suốt alpha (0-1)"""
    n_channels = ha.count_channels(orig_image)
    if int(n_channels[0]) == 1:
        orig_color = ha.compose3(orig_image, orig_image, orig_image)
    else:
        orig_color = orig_image

    orig_scaled = ha.scale_image(orig_color, 1 - alpha, 0)
    heat_scaled = ha.scale_image(heat_color_byte, alpha, 0)
    blended_real = ha.add_image(orig_scaled, heat_scaled, 1, 0)
    blended_byte = ha.convert_image_type(blended_real, "byte")
    return blended_byte


def save_annotated_heatmap(orig_image, heat_color_byte, out_path,
                            true_class, pred_class, pred_score):
    """
    Chồng heatmap lên ảnh gốc VÀ ghi chữ chú thích (Thật/Dự đoán/Score/Đúng-Sai)
    trực tiếp vào ảnh, dùng cửa sổ ẩn (off-screen) để "bake" text vào pixel.
    Nếu việc mở cửa sổ thất bại (ví dụ máy không có desktop session), sẽ tự
    động lưu bản KHÔNG có chữ thay vì crash.
    """
    blended = overlay_heatmap_on_image(orig_image, heat_color_byte, alpha=0.45)
    width_t, height_t = ha.get_image_size(blended)
    width_t, height_t = int(width_t[0]), int(height_t[0])

    is_correct = (true_class == pred_class)
    status_text = "DUNG" if is_correct else "SAI"
    label = f"That:{true_class} Doan:{pred_class} ({pred_score*100:.0f}%) {status_text}"

    try:
        win = ha.open_window(0, 0, width_t, height_t, 0, "buffer", "")
        ha.set_part(win, 0, 0, height_t - 1, width_t - 1)
        ha.disp_image(blended, win)
        ha.set_color(win, "green" if is_correct else "red")
        ha.set_tposition(win, 15, 10)
        ha.write_string(win, label)
        final_image = ha.dump_window_image(win)
        ha.close_window(win)
        final_byte = ha.convert_image_type(final_image, "byte")
        ha.write_image(final_byte, "png", 0, out_path)
    except Exception as e:
        # Nếu bake chữ thất bại (ví dụ do môi trường không có màn hình), vẫn lưu bản không chữ
        print(f"      (Không ghi được chữ vào ảnh: {e} - lưu bản không chữ)")
        ha.write_image(blended, "png", 0, out_path)


def list_all_images(image_dir):
    """Liệt kê tất cả ảnh trong các thư mục con (mỗi thư mục con = 1 lớp)"""
    image_files = []
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for class_name in sorted(os.listdir(image_dir)):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(valid_ext):
                image_files.append((class_name, os.path.join(class_dir, fname)))
    return image_files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Đang load model: {MODEL_PATH}")
    model_handle = ha.read_dl_model(MODEL_PATH)
    ha.set_dl_model_param(model_handle, "batch_size", 1)

    print(f"Đang load tham số tiền xử lý: {PREPROCESS_PARAM_PATH}")
    preprocess_param = ha.read_dict(PREPROCESS_PARAM_PATH, [], [])

    image_list = list_all_images(IMAGE_DIR)
    print(f"Tìm thấy {len(image_list)} ảnh trong '{IMAGE_DIR}'\n")

    # Dò tên tham số thật của 2 procedure trước khi chạy vòng lặp (chỉ 1 lần)
    print("--- Đang dò API (chỉ chạy 1 lần) ---")
    call_proc("gen_dl_samples_from_images", _inspect=True)
    call_proc("preprocess_dl_samples", _inspect=True)
    print("--- Nếu thấy lỗi ValueError ngay sau đây, gửi lại toàn bộ log cho mình ---\n")

    results_summary = []

    for idx, (true_class, img_path) in enumerate(image_list, start=1):
        try:
            image = ha.read_image(img_path)

            # Tạo DLSample dict từ ảnh (Images là tham số ICONIC, không phải control)
            out1 = call_proc("gen_dl_samples_from_images", _iconic={"Images": image})
            sample = out1["DLSampleBatch"]

            # Tiền xử lý sample cho đúng input model (resize/normalize theo lúc train)
            call_proc(
                "preprocess_dl_samples",
                DLSampleBatch=sample,
                DLPreprocessParam=preprocess_param,
            )

            # Inference
            dl_result_batch = ha.apply_dl_model(model_handle, sample, [])
            result = dl_result_batch[0]

            class_names = ha.get_dict_tuple(result, "classification_class_names")
            confidences = ha.get_dict_tuple(result, "classification_confidences")
            pred_class = class_names[0]
            pred_score = float(confidences[0])

            status = "✓" if pred_class == true_class else "✗ SAI"
            print(f"[{idx}/{len(image_list)}] {os.path.basename(img_path)} "
                  f"| Thật: {true_class} | Dự đoán: {pred_class} "
                  f"| Score: {pred_score:.4f} {status}")

            results_summary.append((img_path, true_class, pred_class, pred_score))

            # Sinh heatmap Grad-CAM cho lớp có confidence cao nhất
            heatmap_gen_param = ha.create_dict()
            heatmap_result_batch = ha.gen_dl_model_heatmap(
                model_handle, sample, "grad_cam", [], heatmap_gen_param
            )
            heatmap_result = heatmap_result_batch[0]

            # Kết quả heatmap nằm trong dict lồng nhau key 'heatmap_grad_cam'
            heatmap_keys = ha.get_dict_param(heatmap_result, "keys", [])
            grad_cam_dict = None
            for k in heatmap_keys:
                if "heatmap" in k or "grad_cam" in k:
                    grad_cam_dict = ha.get_dict_tuple(heatmap_result, k)
                    break

            if grad_cam_dict is not None:
                inner_keys = ha.get_dict_param(grad_cam_dict, "keys", [])
                # Tìm key dạng 'heatmap_image_class_<id>'
                heatmap_img_key = next(
                    (k for k in inner_keys if k.startswith("heatmap_image_class")), None
                )
                if heatmap_img_key:
                    heatmap_image = ha.get_dict_object(grad_cam_dict, heatmap_img_key)

                    out_name = f"{true_class}_{os.path.splitext(os.path.basename(img_path))[0]}_heatmap.png"
                    out_path = os.path.join(OUTPUT_DIR, out_name)

                    orig_width, orig_height = ha.get_image_size(image)
                    orig_width, orig_height = int(orig_width[0]), int(orig_height[0])

                    heat_color = colorize_heatmap(heatmap_image, orig_width, orig_height)
                    save_annotated_heatmap(
                        image, heat_color, out_path,
                        true_class, pred_class, pred_score
                    )
                    print(f"      -> Đã lưu heatmap: {out_path}")

        except Exception as e:
            print(f"[{idx}/{len(image_list)}] LỖI với ảnh {img_path}: {e}")

    # In tổng kết
    print("\n===== TỔNG KẾT =====")
    correct = sum(1 for _, t, p, _ in results_summary if t == p)
    total = len(results_summary)
    if total > 0:
        print(f"Đúng: {correct}/{total} ({100*correct/total:.1f}%)")

    ha.clear_dl_model(model_handle)
    print(f"\nHoàn tất. Heatmap được lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()