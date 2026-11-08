# Đề 14 — Phát hiện người đi xe máy không đội mũ bảo hiểm

Web Gradio nhận ảnh hoặc video và trả kết quả có khung chú thích, ảnh bằng chứng và báo cáo CSV. Pipeline offline dùng **YOLO26s → DEIMv2 → ghép đầu với xe → ByteTrack cho video → DINOv3-S probe**. CLIP là tùy chọn đối chiếu mũ vải. Màu xanh là có mũ, đỏ là không mũ và vàng là chưa xác định.

## Chạy trên Windows

Dùng Python 3.12. Cài FFmpeg và FFprobe trên PATH để đọc metadata và xuất video. Các lệnh dưới đây chạy tại thư mục gốc của repo.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-local.txt
.venv/Scripts/python.exe app.py
```

Mở [http://127.0.0.1:7861](http://127.0.0.1:7861/). Có thể đổi cổng bằng `--port 7860`. Web mặc định xử lý trên máy này khi chưa cấu hình GPU API và ưu tiên OpenVINO CPU khi khả dụng.

### Trọng số

- **Dùng thử trên máy mới.** Khi chưa có probe, web chọn YOLO26s + YOLO helmet. YOLO26s COCO được tải theo cơ chế của Ultralytics. Chuẩn bị trọng số YOLO helmet công khai bằng lệnh dưới đây.
- **Dùng pipeline DEIM và DINO.** Đặt probe đã huấn luyện tại `weights/dino_probe.joblib`, hoặc tạo theo [hướng dẫn huấn luyện probe](docs/model_training_proposal_2026-11-05.md). Encoder DINOv3-S cần quyền truy cập Hugging Face và phải có sẵn trong cache hoặc thư mục checkpoint local. Chạy `python scripts/prepare_accuracy_models.py` để tải DEIM và CLIP. Web dùng DEIM nếu đã có trọng số, kể cả khi CLIP tắt.
- Checkpoint, probe và dữ liệu nằm trong các thư mục local được Git bỏ qua.

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; from huggingface_hub import hf_hub_download; import shutil; Path('weights').mkdir(exist_ok=True); shutil.copy2(hf_hub_download('nnsohamnn/helmet-detection-yolo11', 'yolov11s(80 epochs).pt'), 'weights/helmet_yolo11s_nnsohamnn.pt')"
```

### Chạy CLI

```powershell
# Máy mới với trọng số YOLO helmet
.venv/Scripts/python.exe src/run_video.py path/to/video.mp4 --profile openvino_cpu --classifier yolo --estimated-heads --no-accuracy

# Cùng cấu hình với web khi đã có DEIM và probe DINO
.venv/Scripts/python.exe src/run_video.py path/to/video.mp4 --profile openvino_cpu --no-accuracy --f2
```

Video đầy đủ giữ thời lượng, độ phân giải, timestamp của các frame và audio tương thích MP4. Stride chỉ điều khiển tần suất inference. Các frame không phân tích giữ nguyên hình. Xem [cách xuất video và đọc audit](docs/full_length_video_2026-11-06.md).

## GPU và gán nhãn

- [Backend NVIDIA qua HTTP và GPU Kaggle](docs/gpu_api_web_2026-11-07.md) dùng chung pipeline với web local. Kaggle cần notebook và dataset riêng tư do người triển khai cấu hình.
- [Notebook Colab/Kaggle và quy trình benchmark](docs/cloud_workflow_2026-11-05.md) mô tả cách chuẩn bị môi trường và gói nguồn.
- [Trang gán nhãn video local](docs/video_labeling_guide_2026-11-06.md) dùng để xuất và rà soát nhãn theo frame.

## Cấu trúc và kiểm thử

| Đường dẫn | Nội dung |
| --- | --- |
| `app.py` | Giao diện ảnh/video và lựa chọn nơi xử lý |
| `src/de14/` | Model, pipeline, tracking, xuất video và GPU API |
| `src/run_video.py` | CLI chạy video |
| `configs/` | Cấu hình runtime, tracking và triển khai |
| `scripts/` | Chuẩn bị dữ liệu, chạy dịch vụ, benchmark và đóng gói |
| `notebooks/` và `kaggle/` | Workflow notebook và mã kernel thử nghiệm |
| `tests/` | Kiểm thử Python và JavaScript |
| `docs/` | Hướng dẫn, kết quả benchmark và [lịch sử thử nghiệm](docs/benchmark_history.md) |

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
node tests/test_annotation_ui.js
```

Bộ kiểm thử Python cần FFmpeg, FFprobe và các thư viện trong `requirements-local.txt`. Các ca kiểm thử HTTP GPU dùng model giả lập để kiểm tra giao thức và pipeline. Số liệu thực nghiệm nằm trong [kết quả benchmark](docs/results/) và [đối chiếu Kaggle với OpenVINO CPU](docs/kaggle_video_benchmark_2026-11-07.md).
