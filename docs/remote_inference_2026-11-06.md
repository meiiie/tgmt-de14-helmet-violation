# Kết nối inference local / GPU

**07/11 — cập nhật thay thế:** đã triển khai web → HTTP GPU API → kết quả tự động, theo lựa chọn web riêng của người dùng. Chi tiết hiện hành tại [GPU API web](gpu_api_web_2026-11-07.md). Phần bên dưới là lịch sử đề xuất trước triển khai; host/endpoint GPU thật vẫn chưa được cung cấp.

**Cập nhật theo trả lời mới nhất:** người dùng hiện chỉ có ColabT4, chưa có API endpoint. Đã bổ sung luồng notebook tương tác dùng cùng protocol/runtime tại `docs/colab_gpu_inference_2026-11-06.md`. Phần HTTP tự động bên dưới là đề xuất chưa được nối tới server thực, không phải GPU đã kết nối vào app local.

Yêu cầu: app local trên Intel có thể gửi ảnh/video sang môi trường NVIDIA, sử dụng đúng pipeline hiện tại và nhận kết quả về. Không chạy SAM CUDA bằng Torch CPU, không đổi model hoặc classification để giả lập GPU.

Ràng buộc đã xác nhận: người dùng hiện chỉ có Colab T4 miễn phí. Colab FAQ hạn chế dùng free runtime chủ yếu qua web UI bên ngoài notebook; không mặc định triển khai API/tunnel trên free Colab.

Hai cách triển khai:

1. Job ZIP cho Colab tương tác: app xuất input + options + SHA256; người dùng chạy notebook một lần cho mỗi job; notebook dùng runtime factory/Pipeline/video runner có sẵn, xuất result ZIP; app kiểm tra checksum và nạp ảnh/video/CSV/audit. Có thao tác thủ công, không phải API liên tục.
2. HTTP API có xác thực trên GPU server riêng: cùng protocol/job executor, app gửi job và lấy kết quả tự động. Cần endpoint GPU thực và môi trường cho phép cung cấp dịch vụ. Chưa có server này trong phiên hiện tại.

Người dùng đã chọn **HTTP API tự động** và sẽ cung cấp GPU server. Triển khai API có bearer token, một worker inference để không đụng tracker/model giữa các job, nhận job ZIP streaming và trả status/result ZIP. App có URL/key/test connection, không tải model local ở mode remote. Không triển khai tunnel/API trên Colab miễn phí; không tạo tài nguyên trả phí. Endpoint thực còn chờ người dùng cung cấp.

Protocol v1 dự kiến: JSON allowlist options (backend YOLO/SAM, classifier, head detector, CLIP, imgsz, stride, n_min, F2, giới hạn phân tích, full/preview); input media một file; job UUID; checksum và kích thước. Không gửi HF_TOKEN/API key/checkpoint trong ZIP. Không chấp nhận script, đường dẫn hoặc command từ job.

Executor GPU bắt buộc CUDA khả dụng, không fallback CPU. Ảnh dùng static pipeline; video dùng runner full-length hiện có. Result gồm runtime/provider thực, timing, tracks/observations/filter audit, ảnh/video/CSV/evidence. App không cần có weights hoặc CUDA khi xuất/nạp job.

Archive validation: từ chối absolute path, traversal, backslash, symlink, duplicate entries, giới hạn file/bytes, validate toàn bộ manifest và SHA trước khi sử dụng. Chỉ unpack vào thư mục job riêng; không ghi vào source/weights. Kiểm thử transfer và pipeline với model fixture nhẹ trên CPU; inference SAM/T4 cần chạy thật ở Colab, chưa thể xác nhận từ local.
