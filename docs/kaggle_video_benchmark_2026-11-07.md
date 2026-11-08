# Kaggle GPU: video35giây và đối chứng Intel OpenVINO

## Kết quả đã nhận ngày07/11

ZIP người dùng gửi xác nhận TeslaT4 đã chạy351frame mẫu. Lượt CPU chạy lại đã hoàn tất với mã thoát0. Bộ so sánh chung xác nhận trùng video, weights, tensor/config encoder, code và cấu hình thí nghiệm.

| Chỉ số toàn clip | OpenVINO CPU local | Kaggle TeslaT4 |
|---|---:|---:|
| Phân tích351frame | 1514,8giây | 63,8giây |
| Render và xuất video | 65,2giây | 103,7giây |
| Tổng lượt video sau warmup | 1587,0giây | 177,4giây |
| Tải model và warmup | 70,0giây | 10,3giây |
| Track giữ lại | 52 | 49 |
| Track chưa xác định | 32/52 | 31/49 |

Phần phân tích tăng tốc23,7lần, cả lượt video tăng tốc8,9lần. Render/xuất chiếm58,5%thời gian lượt GPU. NVML đo processpeak1016MiB. Video GPU có1052frame,30FPS,1920×1080,35,0667giây. SHA256 dữ liệu packet AAC của nguồn và output trùng nhau.

Sáu ảnh phát triển cho cùng24/27đầu khớp box thô trên hai backend, các nhãn đầu trùng nhau. Người đội mũ lông vẫn bị gán có mũ. Trên18mốc video đã duyệt trùng frame phân tích, cả hai backend khớp8/22box đầu đã gán. Trace GPU có14box khớp sau detector và ghép đầu–xe,8đi tới crop/kết quả và8sauF2. Cần rà cách xác nhận track và giữ quan sát đầu trước crop. Helmet precision/recall toàn video vẫn cần nhãn đủ để đo. Các con số trên là một lượt đo cho mỗi backend và bộ ảnh phát triển nhỏ.

Bằng chứng lưu trong `outputs/kaggle_video_20261107/assessment.json`, `comparison/comparison.json`, `comparison/comparison_table.csv`, `cpu_full_retry/report.json`, `kaggle_received/report.json` và các trace đã ghi. Lượt CPU đầu dừng ở mẫu262. Các trạng thái chưa chạy ở phần hướng dẫn bên dưới phản ánh thời điểm chuẩn bị gói input.

Kaggle phù hợp để thử cùng YOLO26s, DEIMv2 head detector và DINOv3-S probe trong notebook GPU. Tài liệu Kaggle mô tả accelerator GPU và quota; GPU cụ thể/quota còn lại phải xem trong tài khoản, không coi là tài nguyên luôn sẵn. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).

Đây là đợt đánh giá miễn phí trong notebook, không thay giao diện web của dự án và không triển khai Kaggle làm API web. Không thuê Runpod hoặc mua tài nguyên trong đợt thử này.

## File đã chuẩn bị

- `notebooks/helmet_kaggle_video_benchmark.ipynb`: notebook chuyên cho video/ảnh khó, gọi CLI chung của repository.
- `outputs/cloud/helmet_kaggle_input.zip`: gói **private input**, khoảng240MB. Có code hiện tại, `1102.mp4`, YOLO26s, DEIM ONNX, probe/provenance, sáu ảnh phát triển và nhãn box đầu thô. Khác source ZIP thông thường, gói này chủ động chứa các weights/media nêu trên để giảm tải lại.
- Gói không có HF_TOKEN/API key, cache DINO gated, checkpoint SAM, model CLIP hoặc training data. DINOv3 tải bằng quyền riêng của bạn qua Secrets; không train probe mới.
- `scripts/benchmark_video_runtime.py`: cùng phép đo trên local CPU hoặc cloud GPU. `scripts/package_kaggle_video.py` tạo lại private input bằng whitelist file cố định.

Input có checksum từng file, gồm code/config/requirements/probe, được kiểm tra khi mở notebook hoặc chạy lại cell sau restart. Checksum tự chứa phát hiện file thay đổi/lỗi copy, không thay thế việc tin cậy nguồn code/probe. Chỉ dùng bundle do dự án này tạo, giữ Dataset và Notebook **private**.

## Thao tác trên Kaggle

1. Tạo Notebook mới, dùng Import Notebook để chọn `helmet_kaggle_video_benchmark.ipynb`.
2. Tạo private Dataset từ `helmet_kaggle_input.zip`, rồi Add Input vào notebook. Notebook nhận được cả ZIP còn nguyên lẫn Dataset đã tự giải nén. Nếu đính kèm nhiều bundle, điền `INPUT_ZIP` đúng đường dẫn, không đoán tên Dataset.
3. Trong Settings, bật Internet và Accelerator GPU. Ưu tiên T4 x2 nếu tài khoản được cấp; code dùng **cuda:0, một GPU**, không cộng VRAM/hứa tăng tốc gấp đôi. P100 chỉ dùng được nếu Torch wheel của môi trường hỗ trợ; cell kiểm tra thực thi CUDA sẽ dừng nếu không phù hợp.
4. Trong Add-ons → Secrets, thêm `HF_TOKEN` và cấp quyền cho notebook. Tài khoản Hugging Face phải được quyền tải `facebook/dinov3-vits16-pretrain-lvd1689m`. Không gửi token qua chat hoặc thêm vào Dataset.
5. Chạy mục1–3 để mở dữ liệu, kiểm tra CUDA và cài dependencies. Giữ Torch/torchvision GPU của Kaggle; pin Ultralytics/transformers/numpy theo input local, scikit-learn theo provenance probe, dùng ORT GPU theo stack chuẩn bị. Không cài requirements-local hoặc ORT CPU lên runtime GPU. Sau installation, restart Python, chạy lại mục1, bỏ qua2–3 và tiếp tục4.
6. Chạy mục4–7: kiểm tra provider/probe → đo video và ảnh khó → so CPU nếu có report → tải `helmet_kaggle_results.zip` ở `/kaggle/working`.
7. Dừng session GPU sau khi tải kết quả để giữ quota. Notebook chưa được thực thi trên Kaggle trong phiên làm việc local này vì không có phiên/API đăng nhập Kaggle được cấu hình.

CUDA12/cuDNN9 có pin ORT1.20.2 hoặc1.22.0 theo stack; CUDA13/cuDNN9 dùng1.30.0 từ lần T4 thực người dùng đã gửi. Nhánh1.20.x dùng1.20.2 vì pin1.20.1 cũ không còn có trên PyPI hiện tại. Đây là các stack đã chuẩn bị, không bảo đảm mọi image Kaggle tương thích. Torch kernel nhỏ, CUDA provider khả dụng và session DEIM thực đều được kiểm tra; phải xử lý lỗi stack trước khi đo. [ORT CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [ORTgpu1.20.2](https://pypi.org/project/onnxruntime-gpu/1.20.2/), [image GPU Kaggle chính thức](https://github.com/Kaggle/docker-python).

## Phép đo giữ cố định

| Mục | Giá trị |
|---|---|
| Video | 1102.mp4,1052frame,30FPS,1920×1080,video35,0667giây, có audio |
| Source SHA256 | `4239a39500fbccf4c9908099ff4cce1e570c7fcd91078da6bafaaee165d9e335` |
| Model | YOLO26s + DEIMv2 + DINOv3-S probe hiện có |
| imgsz YOLO | 1280 |
| Lấy mẫu inference | stride3 trên **toàn clip**, dự kiến351frame phân tích |
| Quan sát tối thiểu | 3, dùng logic known observations hiện có |
| CLIP | Tắt |
| F2 | Bật cho video; ảnh tĩnh không áp dụng bộ lọc thời gian |
| Output | Full-length, giữ frame/FPS/resolution/audio tương thích; frame không inference giữ nguyên |

Không đổi model, weights, thresholds, classifier hoặc training. Head session có CUDA ưu tiên và tắt cơ chế retry toàn session sang CPU; provider được đọc lại trước/sau inference. Điều này không tuyên bố tất cả operator trong graph chạy GPU: ORT vẫn có thể dùng CPU cho operator được phân vùng như trong môi trường đã chuẩn bị.

Lệnh local thực tế đã khởi chạy để tạo CPU report mới:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/benchmark_video_runtime.py path/to/1102.mp4 --profile openvino_cpu --stride 3 --min-observations 3 --hard-cases data/benchmark/crowded_20261106.json --out outputs/kaggle_video_20261107/openvino_cpu
```

CPU report chỉ có sau khi **cả video lẫn kiểm tra ảnh khó hoàn thành**. Log tiến độ: `outputs/kaggle_video_20261107/cpu_run.log`. Không dùng thời gian366,9giây của cấu hình cũ làm baseline tương đương cho cấu hình mới. Không chạy CPU Kaggle để đại diện i7-1355U/Iris Xe.

Nếu đóng gói sau khi CPU report đã hoàn thành:

```powershell
.venv/Scripts/python.exe scripts/package_kaggle_video.py path/to/1102.mp4 --cpu-report outputs/kaggle_video_20261107/openvino_cpu/report.json
```

Notebook sẽ tự tạo `comparison.json` và bảng CPU/GPU khi report có trong input. Nếu GPU đã chạy trước khi CPU xong, tải ZIP GPU về local và so tại đây; không cần chạy lại inference GPU chỉ để lập bảng:

```powershell
.venv/Scripts/python.exe scripts/benchmark_video_runtime.py --compare outputs/kaggle_video_20261107/openvino_cpu/report.json <thu-muc-ket-qua-Kaggle>/report.json --out outputs/kaggle_video_20261107/comparison
```

So sánh từ chối tính speedup nếu khác video, weights, tensor/config encoder, processor, code inference hoặc cấu hình sampling/batch/F2. Runtime/backend/threading và package versions được ghi riêng để diễn giải khác biệt CPU/GPU. Notebook giữ lại kết quả GPU nếu phép so sánh bị từ chối.

## Các chỉ số và cách đọc

- `model_load_seconds`: khởi tạo runtime; có thể gồm tải model nếu cache trống, không tính là steady-state inference.
- `warmup_seconds`: một frame có đầu/người, pipeline riêng; không thêm observation vào video benchmark.
- `processing_seconds`, `sampled_processing_fps`: vòng decode mẫu + detection/association/tracking/classification, không gồm xuất video.
- `inference_fps`: frame mẫu chia tổng thời gian detector/classifier; khác FPS phát video và throughput đầu-cuối.
- `video_wall_seconds`: thời gian toàn lượt video gồm probe/decode/inference/render/encode; model load, warmup, hash và bộ ảnh khó ở ngoài phạm vi này.
- `latency_p50_ms`, `latency_p95_ms`: latency frame mẫu trong vòng pipeline.
- VRAM process bằng NVML và Torch peak ghi riêng. CPU/không đo được ghi null/N/A, không thay bằng0. VRAM video lấy sau warmup, trước bộ ảnh khó.
- Vehicle tracks, tổng person detections lặp theo frame, UNKNOWN rate và F2 audit. Tổng person detections không phải số người duy nhất qua toàn video.
- Output frame/FPS/resolution/duration/audio được runner chung kiểm tra. 351frame inference không làm output chỉ còn351frame.

Lượt local chạy trong môi trường làm việc thực có web/annotation server và kiểm tra nhẹ đang hoạt động, chưa phải máy benchmark được cô lập. Đây là số đo chẩn đoán một lượt; cần lặp lại khi đánh giá cuối cùng, nhất là nếu CPU bị throttle/nhiệt hoặc workload nền thay đổi.

## Chất lượng mũ len, mũ bảo hiểm và trường hợp khó

Sáu ảnh hiện có gồm mũ lông/đầu không mũ, nhiều người/trẻ em cùng xe, xe/đầu nhỏ, đêm và mưa. **Mũ lông không tự được tính là mũ len**; chưa có đủ ảnh mũ len với nhãn xác nhận để báo accuracy riêng nhóm này. Có một nguồn ảnh ứng viên [người đi xe đội beanie trên Pexels](https://www.pexels.com/photo/a-woman-on-a-motorbike-7401383/), nhưng chưa tải/xác minh ảnh gốc hoặc đưa vào bộ đo; không suy nhãn từ mô tả tìm kiếm. [Pexels license](https://www.pexels.com/license/) là giấy phép riêng, không gọi là CC/open dataset.

Kết quả gồm `hard_cases/*_annotated.jpg`, `predictions.json` và `helmet_review.csv` với box, p_no_helmet, nhãn dự đoán. Các cột human_label/hat_type/reviewed để trống/chưa duyệt; không biến dự đoán thành GT. Người review có thể ghi mũ bảo hiểm/mũ len/mũ lông/cap/đầu trần/chưa rõ và xem cả đầu bị bỏ sót.

`head_box_metrics` đo box đầu được ghép vào xe trên nhãn thô của ảnh phát triển; không phải helmet P/R hoặc test độc lập. Nhãn video người dùng trước còn thiếu và có bản hỗ trợ chưa duyệt; không tự coi bản hỗ trợ là GT. Vì vậy helmet precision/recall/FP/FN và ID switches toàn video hiện N/A. UNKNOWN không được coi là đúng hoặc không có vi phạm. Muốn đo recall cần gán cả đầu bị model bỏ sót, không chỉ sửa các dòng prediction đã xuất.

## Trạng thái kiểm chứng

4kiểm thử mới kiểm tra so sánh từ chối mismatch, live provider/fallback guard, xuất video đủ frame bằng shared runner, và không tự gán nhãn cho bộ ảnh khó. Model trong unit test là fixture rõ ràng; không lấy timing của fixture làm FPS GPU. Notebook code cells đã kiểm tra cú pháp, chưa có lần chạy Kaggle thật. Không suy tăng tốc/accuracy từ số kiểm thử phần mềm.
