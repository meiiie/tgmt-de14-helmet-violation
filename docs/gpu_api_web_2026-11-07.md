# Web riêng với backend NVIDIA GPU

Người dùng thao tác trên web: chọn ảnh/video → Nhận diện → xem ảnh/video, số xe/người, tải CSV. Web gửi file và tham số qua HTTP, chờ job rồi tự tải và kiểm tra kết quả. ZIP chỉ là định dạng truyền nội bộ; không còn bước xuất/nhập ZIP bằng tay trên màn hình chính. Notebook không phải giao diện của hệ thống.

Màn hình chính chỉ giữ input, nút nhận diện và kết quả. Cài đặt thiết bị, endpoint, model, sampling, F2, CLIP nằm trong accordion đóng. Bảng chi tiết/evidence và thông tin xử lý cũng đóng. Tab chuyển ZIP Colab và camera thử nghiệm được ẩn, các helper cũ vẫn giữ để tương thích. Chỉ hiện loại kết quả ảnh hoặc video đã chạy.

Giữ các mặc định đang dùng: YOLO26s + DEIMv2 + DINOv3-S probe, CLIP tắt, imgsz1280, stride3, min_observations3, F2 bật cho video, phân tích toàn clip, full-length output. Không thay model, thresholds, classification, tracker hoặc training. Các giá trị này không được tuyên bố là tối ưu mới dựa trên kiểm thử HTTP.

## Triển khai backend một lần

Cần một máy NVIDIA cho phép chạy dịch vụ API và một môi trường Torch CUDA/onnxruntime-gpu đã hoạt động. Có thể dùng GPU phòng lab hoặc máy NVIDIA sẵn có; lựa chọn thuê GPU cần chủ tài khoản tự quyết định chi phí. Chưa có máy chủ/endpoint NVIDIA trong phiên làm việc này.

Người dùng đã chọn chuẩn bị phương án thuê GPU theo giờ, chi phí thấp. Xem [cấu hình và dự toán Runpod](gpu_rental_low_cost_2026-11-07.md); chưa tạo hay mua tài nguyên.

Clone repo hoặc đóng gói nguồn bằng `python scripts/package_cloud_workspace.py`. Chạy các lệnh từ thư mục dự án trong môi trường đã cài Torch CUDA và `onnxruntime-gpu`. File requirements dưới đây cài thư viện inference dùng chung và thư viện HTTP. Chỉ dùng `requirements-local.txt` cho máy CPU để tránh cài đồng thời hai bản ONNX Runtime.

```bash
python -m pip install -r requirements-gpu-api.txt
python -c "import torch; import onnxruntime as ort; print(torch.__version__,torch.cuda.is_available(),ort.get_available_providers())"
```

Chuẩn bị cùng model như bản hiện tại:

- `weights/yolo26s.pt`
- `weights/deimv2_wholebody49_boxes_only.onnx`
- `weights/dino_probe.joblib` cùng provenance; dùng đúng probe đã huấn luyện.
- DINOv3 ViT-S/16 và processor trong HF cache hoặc thư mục local mà encoder hiện có hỗ trợ. Khi tải gated model, host cần quyền truy cập hợp lệ và `HF_TOKEN` qua biến môi trường/secrets; không đưa token vào source, archive hoặc trình duyệt.
- CLIP không cần cho mặc định tắt đối chiếu. SAM là lựa chọn nâng cao, cần checkpoint/quyền riêng; không tự thay sang SAM.
- `ffmpeg` và `ffprobe` trên PATH để xuất video full-length và giữ audio.

Backend kiểm tra Torch CUDA và CUDAExecutionProvider lúc khởi động. Health xác nhận khả năng CUDA của môi trường; từng job còn kiểm tra device/provider thực của runtime đã nạp. Health không thay cho một lần inference thành công với đủ weights.

Linux trên GPU host (khóa chỉ tạo một lần; lưu bằng secret manager hoặc biến môi trường riêng của host):

```bash
export HELMET_GPU_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python scripts/run_gpu_server.py --host 0.0.0.0 --port 8765
```

Máy chủ cần một URL HTTPS mà máy chạy web truy cập được. Nếu dùng Runpod Pod, expose HTTP port8765 và lấy URL proxy được cung cấp; API bất đồng bộ trả jobID sớm, polling từng request riêng. Không cần kết nối HTTP chờ inference nhiều phút. Đây là hướng dẫn triển khai, chưa tạo Pod hoặc mua GPU. Xem [Runpod expose ports](https://docs.runpod.io/pods/configuration/expose-ports).

Khởi động **một process/worker**. Kernel file lock từ chối process thứ hai dùng chung jobs-dir, bảo vệ job đang chạy khỏi thao tác dọn lúc startup. Hàng đợi tối đa2 job gồm đang upload/chờ/chạy, một worker inference để tracker/model không bị gọi đồng thời. API từ chối CPU và trả429 khi đầy; web không tự fallback local.

## Nối web tới GPU

Trên máy Intel đang chạy web, cấu hình biến môi trường trong cùng terminal trước khi mở app:

```powershell
$env:HELMET_GPU_API_URL='https://<dia-chi-backend>'
$env:HELMET_GPU_API_TOKEN='<cung-khoa-API-da-luu-tren-GPU-host>'
.venv/Scripts/python.exe app.py --port 7861
```

Khi chạy local, có thể mở Cài đặt nâng cao, chọn **GPU qua API**, nhập URL/key trong **Kết nối máy chủ GPU** và bấm **Kiểm tra kết nối**. Key từ biến môi trường được giữ ở máy chủ. Khi dùng `--share`, web chỉ kết nối tới endpoint do máy chủ cấu hình. Dùng HTTPS ngoài mạng nội bộ.

Khi chỉ cấu hình `HELMET_GPU_API_URL`, web mặc định chọn **GPU qua API**. Các cấu hình còn lại mặc định chọn **Máy này** và dùng OpenVINO CPU nếu có. Sau khi chọn file và bấm Nhận diện, phần chi tiết hiển thị riêng thời gian inference và tổng thời gian trên web.

Bản cập nhật dùng cùng renderer `veryfast` và giữ box đầu có nhãn `UNCONFIRMED` trong cả luồng local và executor GPU. Với backend GPU đã chạy, cập nhật mã backend từ ZIP nguồn rồi khởi động lại dịch vụ để job mới dùng renderer và pipeline đã sửa.

## GPU Kaggle qua notebook batch

Trên máy chạy web, đăng nhập Kaggle CLI và đặt `HELMET_KAGGLE_NOTEBOOK` thành tên notebook riêng tư dạng `owner/notebook`. Đặt `HELMET_KAGGLE_MODELS` thành tên dataset riêng tư chứa bundle benchmark `helmet_kaggle_input.zip`. Nếu CLI nằm ngoài PATH, đặt `HELMET_KAGGLE_CLI` thành đường dẫn tới chương trình `kaggle`. Với một checkpoint DINOv3 riêng tư đã được cấp quyền, đặt `HELMET_KAGGLE_DINO` thành tên dataset chứa checkpoint, manifest SHA-256 và giấy phép. Nếu dùng Hugging Face để tải model, notebook cần đọc được secret `HF_TOKEN` trong lượt chạy batch. Khởi động app và mở địa chỉ localhost. Web mặc định chọn **Máy này**. Muốn gửi job lên Kaggle, chọn **GPU Kaggle** trong **Cài đặt nâng cao → Nơi xử lý**.

Web tạo dataset riêng tư cho file bạn chọn, gửi một version notebook T4, chờ GPU rồi tải kết quả của đúng version. Web xác minh job ID và hash đầu vào trước khi mở ảnh hoặc video. Kaggle có thể chờ cấp GPU nhiều phút cho mỗi lượt chạy. Khi job kết thúc, web yêu cầu xóa dataset đầu vào tạm. Kaggle giữ kết quả theo version notebook riêng tư. Nếu quá thời gian hoặc mất kết nối khi gửi notebook, hãy kiểm tra trạng thái version trước khi xử lý dataset tạm.

**Thông tin xử lý** hiển thị riêng thời gian tải lên Kaggle, chuẩn bị dataset, chờ GPU, chuẩn bị môi trường và tải kết quả. Số đo chi tiết của từng job được ghi trong `outputs/kaggle_web/timings/`. Thời gian chờ GPU đo từ lúc Kaggle nhận notebook đến lần web thấy trạng thái `RUNNING`. Đây là xấp xỉ theo chu kỳ kiểm tra 10 giây. Notebook ghi thời gian kiểm tra CUDA và cài thư viện bằng đồng hồ của phiên Kaggle. Phiên chỉ gọi pip cho các gói thiếu hoặc sai phiên bản. ONNX Runtime GPU được dùng lại nếu đúng phiên bản và một lần import mới xác nhận có CUDA provider.

Cấu hình Kaggle dùng tài khoản CLI của máy chủ và chỉ mở web ở localhost. Đường dẫn này dùng YOLO26s + DEIM + DINOv3-S probe với CLIP tắt. Thông tin về notebook, quyền truy cập secret và các lệnh batch có trong [tài liệu Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## API và vòng đời dữ liệu

Tất cả endpoint cần `Authorization: Bearer <token>`:

| Endpoint | Chức năng |
|---|---|
| GET `/health` | Protocol, CUDA/GPU, provider, số job đang chờ/chạy |
| POST `/jobs` | Streaming request ZIP; xác thực archive, trả202/jobID |
| GET `/jobs/{id}` | queued/running/completed/error |
| GET `/jobs/{id}/result` | Tải kết quả ZIP đã hoàn thành |

Mỗi job có UUID và SHA256 của input; web kiểm tra identity và checksum trước khi mở kết quả. Không gửi code, command, model hoặc secret trong job. Upload tối đa512MiB; archive expanded tối đa2GiB/256files theo protocol hiện có. Validation/hash/extract chạy ngoài event loop. Video vẫn dùng chung runner giữ FPS, duration, resolution và audio tương thích; frame không inference giữ nguyên, không thêm detection giả.

Sau job thành công chỉ giữ resultZIP; xóa request và execution intermediates. Khi health/submission được gọi, giữ tối đa3 job đã kết thúc hoặc1giờ, trừ download còn đang mở. Response giải phóng pin kể cả Range sai/disconnect. Khi dừng bình thường, xóa các thư mục job thuộc process. Lúc khởi động lại, xóa orphan UUID folder **có marker API chính xác**, giữ nguyên thư mục không thuộc API; vì vậy dùng jobs-dir riêng và một process. Registry không bền qua restart, job đang chạy phải gửi lại. File đã được web tải về nằm trong thư mục tạm của web để phục vụ xem/tải; chưa có cơ chế lưu trữ hồ sơ lâu dài.

Worker hiện nạp model theo mỗi job, giống executor đã kiểm chứng. Chưa thêm model cache hoặc thay inference scheduling; muốn tối ưu tiếp cần đo startup/steady-state trên GPU host thật. HTTP integration tests không chứng minh tốc độ hoặc accuracy tăng.

## Kiểm chứng và giới hạn

`tests/test_gpu_api.py` dùng HTTP TCP thật, xác thực, queue bound, kiểm tra input/output identity, từ chối CPU, bảo vệ key khi đổi endpoint, xử lý file lỗi và retention. Ảnh/video đi qua executor/Pipeline/video runner chung; chỉ model/runtime GPU được thay bằng fixture rõ ràng. Không dùng kết quả fixture để báo FPS/VRAM/precision/recall GPU.

Toàn bộ134 kiểm thử phần mềm qua bằng `python -m unittest discover -s tests -p "test_*.py" -v`; log: `outputs/gpu_api_20261107/full_tests_final.log`. Đây là134 test case của code, không phải134 ảnh/video đánh giá accuracy. Kiểm tra web đang phục vụ và lỗi thiếu endpoint thực qua Gradio API lưu ở `outputs/gpu_api_20261107/acceptance.json`. Không có kiểm tra visual bằng trình duyệt trong lần này.

Chưa có endpoint GPU thật được triển khai hoặc kết nối. ZIP benchmark T4 cũ là bằng chứng của lần benchmark trước, không phải API server đang chạy. Tài nguyên ColabT4 miễn phí không được coi là dịch vụ API cho web: [Colab FAQ](https://research.google.com/colaboratory/faq.html) hạn chế bypass notebook UI để tương tác chủ yếu qua web UI trên free runtime không có compute-unit balance dương. Không triển khai tunnel lách điều kiện này. Bước còn lại là cung cấp GPU host/endpoint cho phép dịch vụ rồi chạy nghiệm thu bằng video thật.
