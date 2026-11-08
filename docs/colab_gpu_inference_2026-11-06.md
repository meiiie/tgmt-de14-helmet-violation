# Dùng T4 đã benchmark để nhận diện ảnh/video mới

ZIP người dùng gửi `path/to/helmet_runtime_benchmark.zip` có SHA256 `209d97606244d7fc26de3eb90de2ef3bf99c2f512f38d97b8dc299935054d5ac`. Báo cáo xác nhận TeslaT4, TorchCUDA, DEIM CUDAExecutionProvider; baseline3.4357 và cloud_cuda4.8881 sampled FPS trên32lượt inference/16ảnh/2lần lặp. Dùng YOLO26s+DINO+DEIM+CLIP, không phải SAM. Đây là kết quả benchmark cũ, không phải GPU có thể gắn vào app Intel, không phải trọng số mới, không phải kết quả xử lý1102.mp4. Không lấy4.89FPS làm cam kết tốc độ cho mọi video/cấu hình.

Môi trường đã chạy thành công: Torch2.11.0+cu130, Ultralytics8.4.203, transformers5.22.0, onnxruntime-gpu1.30.0. **Không chạy lại ô cài dependency/CUDA hoặc benchmark chỉ để xử lý ảnh/video.** Ô setup cũ có guardCUDA12; không dùng guard cũ để kết luận phiênCUDA13 đang hoạt động là sai.

Người dùng xác nhận hiện chỉ có ColabT4, chưa có GPU API endpoint. Luồng hiện tại là notebook tương tác: app local tạo jobZIP → Colab chạy pipeline chung trênGPU → tải resultZIP → app local hiển thị. Không phải kết nối HTTP tự động. API server riêng vẫn là phương án tương lai khi có endpoint thực; không tạo tài nguyên trả phí hoặc tunnel trong bước này. [Colab FAQ](https://research.google.com/colaboratory/faq.html) phân biệt notebook tương tác với việc chủ yếu sử dụng webUI bên ngoài trên free runtime.

## Chạy ngay với phiên Colab trong ảnh

1. Giữ nguyên notebook/runtime T4 đã benchmark. Trong app7861 mở tab **GPU Colab (ảnh / video)**. Chọn một ảnh hoặc video, giữ YOLO26s+DINO, bật DEIM mặc định; CLIP mặc định tắt theo lựa chọn trước. Tạo và tải jobZIP. SourceZIP mới không chứa model/data/token và được cung cấp trong tab.
2. Thêm một ô mã ở cuối notebook Colab hiện tại. Dán toàn bộ nội dung `outputs/cloud/colab_gpu_inference_cell.py` (cũng có ở `notebooks/colab_gpu_inference_cell.py` trong repo). Bản notebook cập nhật có cùng ô ở mục8; không cần mở notebook mới nếu muốn giữ phiên hiện tại.
3. Chạy ô đó; chọn đúng2file: codeZIP mới và requestZIP. **Không chọn helmet_runtime_benchmark.zip.** Ô mã tìm PROJECT cũ hoặc /content/helmet_runtime, cập nhật src/scripts/configs, giữ mọi checkpoint/probe đã có, giữ HF_TOKEN và package GPU đã cài. Không chạy benchmark/training. Worker chạy bằng subprocess mới để không dùng module Python cũ trong kernel.
4. Chờ worker in GPU/model provider thực và tạo `helmet_gpu_result_<jobid>.zip`, trình duyệt tải về. Trong app local chọn ZIP đó, bấm **Mở kết quả GPU**. Nếu vẫn ở cùng phiên app, checksum và jobID được đối chiếu với job vừa xuất. Nếu mở lại app, vẫn kiểm tra integrity/protocol của ZIP nhưng không thể đối chiếu với trạng thái input cũ đã mất.

Đã chuẩn bị riêng `1102_gpu_request.zip` cho video1102.mp4: YOLO26s+DINO, DEIM bật, CLIPtắt, imgsz1280, stride3, n_min3, F2bật, full-length, không giới hạn thời gian phân tích. Cùng code inference của local, không có mô hình/kiến trúc classification riêng trên cloud. SAM chỉ chạy nếu phiên Colab có checkpoint và quyền truy cập SAM, benchmark được gửi chưa xác nhận SAM hoạt động.

## Triển khai và kiểm chứng

- `scripts/run_gpu_job.py` dùng remote_jobs allowlist/checksum và shared runtime/Pipeline/run_video, yêu cầu TorchCUDA và headCUDA provider thật; không fallbackCPU. Image bỏ temporalF2 giống local; video giữ sampling/render tách biệt, FPS/resolution/audio như runner hiện tại. Ghi báo cáo timing/model load, runtime/providers, counts, audit, CSV và media vào resultZIP. Không gửi token, source code hoặc checkpoint trong resultZIP.
- TabGPUColab export/import không gọi model local. ZIPbenchmark được giải thích rõ là chưa có media inference result. File sai job/input bị từ chối khi còn trạng thái job; archive paths/CRC/SHA được module đã có kiểm tra trước extraction.
- Notebook continuation không cài package hoặc thay model đang có. Code ZIP được kiểm tra đường dẫn/symlink/size, validate toàn bộ trước ghi; giữ weights có sẵn. Cell chứa đúng cùng bootstrap code như file download, không định nghĩa pipeline mới.
- Kiểm thử local dùng Pipeline/ByteTrack/render thực với detector/classifier fixture và CUDA metadata giả để kiểm tra transfer/control flow; **không phải benchmarkGPU mới**. Chặn CPU, image/job roundtrip, sourceFPS/framecount, no-local-model, F2image, preservation weights, bad ZIP đều có regression. EvidenceGPU thật hiện tại là ZIP người dùng gửi; việc chạy1102 mới trên T4 còn cần người dùng thực thi ô Colab.

Không hứa giữ GPU khi runtime hết hạn; source/model cache có thể mất khi Colab cấp VM mới. Nếu phiên đã mất, cần chạy lại setup/model cần thiết trong notebook, không cần benchmark lại. Sau inference nên tải kết quả về trước khi runtime bị thu hồi.

Kiểm chứng local:8/8 kiểm thử luồng Colab và124/124 toàn dự án đã qua, log outputs/colab_inference_20261106/focused_final.log và full_tests.log. Reviewer phát hiện temporalF2 bị áp vào image; đã sửa để đúng local và có regression failing-before/passing-after. Source-only ZIP đã rebuild và kiểm tra chứa scripts/run_gpu_job.py; source input1102 hash4239a39500fbccf4c9908099ff4cce1e570c7fcd91078da6bafaaee165d9e335, request jobidbe2847c4a62e4fc99d76f74cbb8ba27b. New1102 T4 run còn chờ người dùng chạy ô notebook; không gộp số FPS CPU fixture thành GPU benchmark.
