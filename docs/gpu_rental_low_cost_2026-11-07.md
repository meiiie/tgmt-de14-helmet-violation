# Thuê GPU theo giờ cho web riêng

Đề xuất ban đầu: **Runpod Pods, một RTX A5000 24GB**, giữ web trên máy Intel và gọi API GPU đã triển khai trong repository. Chọn Pod giúp dùng thẳng FastAPI hiện tại và URL HTTPS được nhà cung cấp cấp; không cần chuyển thuật toán sang một serverless handler khác. Đây là lựa chọn triển khai cho demo/đánh giá, chưa phải kết luận nhà cung cấp rẻ nhất trên mọi thời điểm.

Ngày07/11/2026, trang giá Runpod niêm yết A5000 ở **0,27USD/giờ**. A4000 16GB có thể là lựa chọn nếu tìm thấy offer rẻ hơn trong console, nhưng chưa xác nhận offer/giá A4000 hiện có. Không dùng giá serverless để dự toán Pod. [Giá GPU chính thức](https://www.runpod.io/pricing), [GPU types](https://docs.runpod.io/references/gpu-types).

## Ngân sách dự kiến

Tiền GPU tính theo toàn bộ thời gian Pod được giữ chạy, bao gồm cài môi trường, tải model và chờ thao tác; không chỉ thời gian inference. Với đơn giá tham khảo0,27USD/giờ:

| Thời gian Pod chạy | Tiền GPU, chưa gồm phí khác |
|---|---:|
| 2giờ | 0,54USD |
| 5giờ | 1,35USD |
| 10giờ | 2,70USD |

Bảng là phép tính dự toán, không phải báo giá hoặc số tiền đã chi. Kiểm tra giá **GPU và tổng chi phí** hiển thị trong console trước khi thuê. Nếu GPU vượt0,30USD/giờ, nên chọn offer khác hoặc xem lại ngân sách; đây là ngưỡng đề xuất, chưa có cơ chế chặn thanh toán tự động.

Container/volume khi chạy:0,10USD/GB/tháng; volume khi dừng:0,20USD/GB/tháng. Với20GB container và40GB volume, riêng storage lúc chạy xấp xỉ0,0083USD/giờ nếu quy đổi720giờ/tháng. Giữ40GB volume trong trạng thái dừng suốt tháng tương đương8USD; do đó cần quản lý cả storage sau demo. [Phí Pod/storage](https://docs.runpod.io/pods/pricing).

Tài liệu billing cho biết có thể bắt đầu với10USD credit, credit nạp không hoàn lại. Đây là khoản nạp tài khoản, khác với tiền GPU thực dùng; kiểm tra mức nạp và thuế thực tế tại thanh toán. Không cần bật auto-pay cho đợt thử nhỏ. [Runpod billing](https://docs.runpod.io/accounts-billing/billing).

## Cấu hình cụ thể

File đối chiếu: `configs/runpod_gpu_deployment.json`. File này là checklist cấu hình tay, không phải payload gửi API để tạo máy.

| Mục | Giá trị đề xuất |
|---|---|
| Dịch vụ | Pods, on-demand,1GPU |
| GPU | Ưu tiên A5000; A4000 nếu giá/availability tốt hơn |
| RAM/CPU tối thiểu đề xuất | 16GB RAM,4vCPU |
| Disk | 20GB container;40GB volume `/workspace` |
| Network volume | Chưa cần cho đợt demo ban đầu |
| Template | Official PyTorch có CUDA; kiểm tra stack thực tế trước khi chạy |
| HTTP port | 8765 |
| Worker inference | 1 |
| Secret của API | `HELMET_GPU_API_TOKEN` riêng, ngẫu nhiên |
| Model access | `HF_TOKEN` được cấp quyền DINOv3, đặt bằng Secrets |
| Cache | `HF_HOME=/workspace/hf_cache` |

Nhu cầu VRAM này là đề xuất dựa trên pipeline nhỏ và benchmark T4 trước, chưa đo A5000/A4000. Chưa hứa FPS, latency hay accuracy mới. Không cần thuê A100/H100 cho bước nghiệm thu ban đầu.

## Các bước khi bạn quyết định thuê

1. Mở console Runpod, chọn Pods và đúng GPU/giá theo bảng. Chọn official PyTorch template, cấu hình disk/port/secrets; đối chiếu tổng giá trước bước Deploy. Chỉ tạo một Pod. Chưa có Pod nào được tạo trong phiên làm việc này.
2. Sau khi Pod sẵn sàng, đưa **source một lần** từ `outputs/cloud/helmet_gpu_api_source.zip` vào `/workspace/helmet`. Đưa các weights riêng theo [hướng dẫn GPU API](gpu_api_web_2026-11-07.md); archive source không chứa weights hoặc dữ liệu cá nhân. Đây là bước cài backend, không phải thao tác cần lặp cho mỗi video.
3. Kiểm tra Torch CUDA/ORT/ffmpeg và model. Cài `requirements-gpu-api.txt` trong môi trường GPU. Với host mới chưa có thư viện inference, chuẩn bị các dependencies của dự án trong môi trường riêng; giữ GPU Torch của template và dùng **onnxruntime-gpu**, không cài requirements-local hoặc thay bằng ORT CPU. Các phiên bản đã chạy trên T4: Torch2.11.0+cu130, Ultralytics8.4.203, transformers5.22.0, ORTgpu1.30.0. Đây là hồ sơ tham khảo đã đo, không bảo đảm mọi template/driver chấp nhận CUDA13. Probe local ghi scikit-learn1.9.1 trong provenance; giữ phiên bản tương thích. Nếu stack khác, cần kiểm tra compatibility trước khi benchmark, không tự reinstall ngẫu nhiên.
4. Chạy `python scripts/run_gpu_server.py --host 0.0.0.0 --port 8765` trong thư mục source, bằng môi trường có secrets. Dùng terminal session bền như tmux nếu cần giữ process khi ngắt terminal. Một worker và storage lock ngăn hai process cùng sử dụng thư mục job.
5. Trong phần Connect lấy HTTP proxy của port8765; địa chỉ có dạng `https://<pod-id>-8765.proxy.runpod.net`. Nhà cung cấp đã có HTTPS proxy, không phải tạo tunnel từ Colab. [Expose HTTP ports](https://docs.runpod.io/pods/configuration/expose-ports).
6. Trên web7861: Cài đặt nâng cao → Kết nối máy chủ GPU → URL/key → Kiểm tra kết nối. Từ đó tải ảnh/video và bấm Nhận diện ngay trên web; không chuyển ZIP thủ công.
7. Chạy một ảnh và1102.mp4 trên endpoint thật; đối chiếu GPU/provider từ result, số frame/FPS/duration/resolution/audio, runtime và quyết định với bản hiện tại. Lưu report để đo startup/steady-state, precision/recall trên nhãn hợp lệ. Chưa có benchmark host thuê nên không suy tốc độ từ model GPU hoặc kiểm thử fixture.
8. Kết thúc demo, tải kết quả cần giữ về local rồi Stop hoặc Terminate Pod. Stop vẫn giữ/phí volume; Terminate mất dữ liệu của Pod. Chỉ xóa volume sau khi đã sao lưu probe và dữ liệu cần giữ. Dừng `app.py` trên laptop **không** dừng Pod cloud.

## Vì sao chưa chọn Vast.ai hoặc Serverless ngay

Vast.ai có giá marketplace thay đổi theo host, cộng phí disk và bandwidth; stopped instance vẫn tính storage. Có thể so lại tổng giá trước khi thuê nếu ưu tiên chi phí thấp hơn, nhưng cần xử lý networking/port mapping phù hợp. Chưa truy cập tài khoản/offer có thể thuê nên không hứa giá cụ thể. [Vast billing](https://docs.vast.ai/guides/reference/billing), [Vast networking](https://docs.vast.ai/guides/instances/connect/networking).

Serverless có thể tiết kiệm khi ít request, nhưng cần thiết kế vòng đời worker/model và benchmark cold start. Không đổi kiến trúc lần này chỉ vì kỳ vọng rẻ hơn. Tạm dùng một Pod theo buổi demo để nghiệm thu API thật trước.

Hiện code và cấu hình đã chuẩn bị; **chưa thuê GPU, chưa phát sinh thanh toán, chưa có endpoint hoạt động**. Bước cần tài khoản của bạn là tạo Pod theo cấu hình sau khi tự kiểm tra giá. Không cần gửi API key Runpod hoặc thông tin thanh toán vào chat; chỉ cấu hình URL và khóa ứng dụng trên máy chạy web.
