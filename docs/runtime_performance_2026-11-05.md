# Tối ưu runtime local/cloud — thực nghiệm 05–06/11/2026

Đã giảm latency bằng batch crop theo frame, cache text embedding CLIP và đổi execution backend của DEIMv2 sang OpenVINO. Giữ nguyên YOLO26s, head checkpoint, probe, CLIP, ngưỡng quyết định và thuật toán ghép đầu–xe. Chưa huấn luyện lại, chưa triển khai thay kiến trúc. Ngày trong tên artifact là ngày bắt đầu đợt đo; hoàn tất tài liệu/kiểm tra ứng dụng ngày 06/11.

## Kết quả đo trên Intel i7-1355U + Iris Xe

| Profile | Latency p50 / p95 (ms/frame) | FPS compute | Repeat | Hai ảnh regression |
| --- | ---: | ---: | ---: | --- |
| `baseline_cpu` | 5902 / 7318 | 0.167 | 2 | Qua |
| `optimized_cpu` — ONNX Runtime CPU | 4840 / 5637 | 0.203 | 2 | Qua |
| `openvino_cpu` — DEIM trên OpenVINO CPU | **3417 / 4658** | **0.282** | 2 | Qua |
| `openvino_gpu` — DEIM trên Iris Xe, phần Torch trên CPU | 3720 / 4391 | 0.275 | 1 | Qua |
| `experiment_dino_only` — bỏ CLIP | 4676 / 4955 | 0.215 | 1 | **Không qua: mũ lông bị nhận thành mũ bảo hiểm** |

OpenVINO CPU giảm p50 khoảng **42.1%** so với baseline trên mẫu đo này. Iris Xe chưa thắng CPU về p50/throughput trong phép đo; chỉ có một repeat nên chưa kết luận thứ hạng ổn định cho video khác. Các profile vẫn chậm hơn realtime. Không suy FPS video đầu-cuối từ bảng này: đo compute sau warmup, bỏ decode, tải model và export. Tải model khoảng 31–51 giây được ghi riêng trong từng report.

Nguồn số liệu: [bản tổng hợp JSON](results/runtime_2026-11-05.json), các report gốc tại `outputs/benchmarks/{baseline_cpu,optimized_cpu,openvino_cpu,openvino_gpu,experiment_dino_only}/report.json` và `predictions.json`. Log chạy ở `outputs/benchmark_*_20261105.log`. Các artifact `outputs/` được Git ignore; cần giữ riêng để tái kiểm chứng.

## Accuracy thực sự đã đo

Mẫu **16 frame duy nhất, 4 official-test clips, 32 quan sát xe có GT, 8 quan sát vi phạm**; các frame liền nhau có tương quan, 8 positives không phải 8 người/track độc lập. Xem [nguồn, split và metric contract](benchmark_data_2026-11-05.md). Mirror chỉ có 640×640, được resize lại 1920×1080 để khớp tọa độ GT; không phục hồi được chi tiết ảnh gốc.

Bốn profile đầy đủ model cho cùng kết quả:

| Metric ở mức xe/frame | Giá trị |
| --- | ---: |
| Precision / recall phát hiện vi phạm không mũ | 0.625 / 0.625 (TP 5, FP 3, FN 3) |
| Precision / recall khớp xe với GT | 0.473 / 0.813 (26/55 dự đoán, 26/32 GT) |
| Chưa xác định | 37/55 = 67.3% dự đoán xe |
| Sai số tuyệt đối trung bình số người | 0.50 trên 26 xe khớp GT |
| Đếm đúng số người | 16/26 = 61.5% xe khớp GT |

Ghép box theo IoU ≥ 0.3, một-một; unknown hoặc không tìm thấy GT-positive vẫn tính FN. Dự đoán vi phạm không khớp GT tính FP dù annotation có thể thiếu xe nền/xe đỗ. Đây là frame metrics (`n_min=1`), **không phải** head AP, crop accuracy hay đánh giá bỏ phiếu toàn track. Đoạn 4 frame chưa đủ warmup tracking như video đầy đủ. Hai repeat không nhân đôi số mẫu accuracy.

Không có thay đổi nhãn/count theo từng dự đoán giữa baseline, batch/cache và OpenVINO CPU; chênh xác suất lớn nhất khoảng 5.64e-7. Kiểm tra cache CLIP trên 2 crop thật có sai lệch margin tối đa 1.91e-6. Hai ảnh của người dùng được kiểm chứng riêng, không đưa vào training: ảnh 1 nhận 2 người đều không mũ bảo hiểm; ảnh 2 đếm các xe 1, 2 và 4 người (7 tổng), phân loại 4 đầu có mũ và giữ 3 đầu chưa xác định. Không khẳng định đã phân loại đúng mũ cho những crop không đủ chi tiết.

Thử bỏ CLIP có cùng P/R tổng hợp trên mẫu 16 frame nhưng làm sai ảnh mũ lông. Vì vậy **giữ CLIP**; không dùng aggregate metric để che regression đã biết. Thử nghiệm này có exit code 1 theo acceptance check, là candidate bị loại.

## Một codebase, các execution profile

`src/de14/runtime.py` xây detector/classifier/pipeline chung cho Gradio, CLI và notebook. `configs/runtime_profiles.json` điều khiển thiết bị Torch, head backend, CPU threads, batch và cache. Cấu hình CUDA tường minh báo lỗi nếu Torch CUDA hoặc ORT CUDA EP không sẵn sàng; không âm thầm ghi kết quả CPU thành GPU.

- `baseline_cpu` giữ per-vehicle classification và CLIP text forward mỗi lần để đối chứng.
- `optimized_cpu` giới hạn ORT threads, tắt spinning, gom crop theo frame, cache text CLIP. Trong 32 timing frames, classifier/CLIP image forwards giảm **58 → 30**, text forwards sau warmup **58 → 0**. Vẫn chạy đủ các model; không bỏ crop hoặc giảm input size.
- `openvino_cpu`/`openvino_gpu` chỉ chuyển DEIM ONNX sang OpenVINO; YOLO/DINO/CLIP vẫn Torch CPU. Metadata ghi actual execution devices.
- `baseline_cuda`/`cloud_cuda` dùng cùng GPU và model cho cặp đối chứng trên cloud; cùng CLI, không có bản pipeline thứ hai.
- `experiment_dino_only` chỉ dành cho ablation, không xuất hiện trong lựa chọn mặc định UI.

OpenVINO ban đầu không chuyển được GatherElements có non-axis dimension nhỏ hơn input của DEIM. Runtime thêm lowering tương đương ONNX bằng coordinate-grid GatherND; checkpoint không bị sửa. Ba test bằng OpenVINO thật bao phủ dimension không bằng nhau, axis 0, negative axis/indices. Giữ FP32, không INT8/FP16 trong phép đo local này. Cần `openvino>=2025.3`; đã chạy với 2026.4.1.

Mặc định portable trong code là `optimized_cpu`. Local hiện tại chọn `openvino_cpu` qua biến môi trường; máy không cài OpenVINO vẫn chọn được ONNX Runtime mà không sửa source.

```powershell
python -m pip install -r requirements-local.txt
$env:HELMET_RUNTIME_PROFILE='openvino_cpu'
python app.py --port 7861
python src/run_video.py path/to/video.mp4 --profile openvino_cpu --out outputs/local_video
```

```bash
# Notebook GPU, dùng đúng cùng CLI
python src/run_video.py /path/video.mp4 --profile cloud_cuda --out outputs/cloud_video
```

Tái đo bằng `scripts/prepare_benchmark_data.py --ipv4`, sau đó `scripts/benchmark_runtime.py --profile <profile> --repeats 2 --out outputs/benchmarks/<profile>`. Thêm `--regression-images outputs/user_review/case1_input.png outputs/user_review/case2_input.png` để kiểm tra hai ảnh. Giữ manifest SHA256 `3134f196ed85fb35f97f87fe1e8b839f947993ed271aa4794a6086027f83ea59` và checkpoint hashes trong report khi so sánh.

## Cloud và kiểm thử

[Notebook Colab/Kaggle](../notebooks/helmet_runtime_colab_kaggle.ipynb) và [hướng dẫn cloud](cloud_workflow_2026-11-05.md) chuẩn bị môi trường NVIDIA, secrets, probe provenance, chạy baseline/optimized CUDA trên cùng manifest, export report. ZIP source được tạo bởi `scripts/package_cloud_workspace.py`, không cần push code hoặc đưa checkpoint lớn vào Git. Bản `--include-probe` chỉ thêm probe nhỏ và provenance.

**Chưa chạy GPU cloud trong phiên này**: máy local không có CUDA, chưa có runtime/credentials cloud được kết nối. `outputs/benchmarks/cloud_cuda/status.json` ghi trạng thái chưa chạy; latency/FPS/VRAM NVIDIA mới đều null. Những kết quả T4 cũ ở README không thay cho benchmark pipeline mới. Benchmark có Torch peak + tùy chọn NVML process sampling để đo cả ORT; trên local Intel, CUDA VRAM không áp dụng, shared memory Iris Xe chưa đo.

**51 kiểm thử phần mềm đã qua** (`outputs/runtime_unit_tests_20261105.log`, `Ran 51 tests ... OK`): metrics, split/cache dữ liệu, batch ordering/unknown, cache math/API, runtime/device, OpenVINO operator và regression logic. Đây là kiểm tra code, tách biệt với các lần inference model thật ở trên; không phải 51 mẫu dữ liệu chứng minh accuracy. `compileall` và CLI help cũng qua; read-only review không còn lỗi Critical/Important đã biết trong phạm vi thay đổi.

Ngày 06/11, đã khởi động lại ứng dụng tại `http://127.0.0.1:7861/` với `openvino_cpu`. Kiểm tra HTTP config xác nhận profile/backend và 10 input API; request `/analyze` thật với ảnh 1 trả 1 xe, 2 người, nhãn không mũ. Evidence tại `outputs/runtime/app_acceptance_20261106.json`. Request này mất 6.6 giây ở phần phân tích, có first-use/export overhead, không thay số warm compute trong bảng. Hai ảnh đã qua real-model benchmark ở trên; API smoke này kiểm tra tích hợp thêm một ảnh.

## Quyết định tiếp theo

Chọn OpenVINO CPU trên máy đã đo; giữ kiến trúc đủ model. [Đề xuất dữ liệu/training](model_training_proposal_2026-11-05.md) bắt đầu bằng phân tích lỗi ghép/tracking và annotation head thực. Không train hoặc thay detector chỉ vì nhìn thấy FN tổng hợp; file `outputs/benchmarks/failure_audit.json` đã cho thấy raw/union candidate vẫn tồn tại ở một số quan sát bị miss cuối pipeline.
