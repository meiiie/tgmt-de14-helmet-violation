# Chuyển Offline Gradio sang YOLO26s — 02/11/2026

## Cấu hình

`YOLO26s + DEIMv2 → ghép đầu–xe → ByteTrack → crop đầu → DINOv3/probe + CLIP → kết quả`

- `app.get_models()` truyền rõ `weights/yolo26s.pt` vào `YoloDetector`. Hai lựa chọn YOLO của Offline và các thông báo liên quan đã đổi tên thành YOLO26s.
- Khi đã có probe, mặc định chọn YOLO26s + DINOv3-S; SAM 3.1 vẫn là lựa chọn riêng trong menu. Bật **Ưu tiên độ chính xác** để dùng DEIMv2 + CLIP.
- Giữ nguyên DEIMv2 bản đầy đủ, probe, CLIP, ngưỡng phân loại, ghép đầu–xe và ByteTrack từ bản cải thiện trước. Không huấn luyện thêm bằng hai ảnh kiểm tra.
- Tên instance `YoloDetector` lấy từ tên checkpoint, tránh báo nhầm YOLO11s khi thực tế nạp YOLO26s. Constructor mặc định của lớp vẫn dùng YOLO11s cho các script huấn luyện/đánh giá lịch sử; nhánh Offline Gradio truyền checkpoint YOLO26s tường minh.

## Kiểm chứng mới

Chạy cả YOLO26s và YOLO11s trong cùng tiến trình, dùng chung DEIMv2 và bộ phân loại. Mỗi model được warm-up một lượt, sau đó chạy ba lần mỗi ảnh, đảo thứ tự model ở lượt đo thứ hai. Đo trên CPU, PyTorch 2.15.0+cpu, Ultralytics 8.4.193, `imgsz=1280`, 4 thread PyTorch.

| Trường hợp | YOLO26s | YOLO11s chạy đối chứng |
| --- | --- | --- |
| Ảnh 1 — số người | 2 | 2 |
| Ảnh 1 — nhãn mũ | Cả 2 không mũ | Cả 2 không mũ |
| Ảnh 2 — số xe/người | 3 xe, 7 người; từng xe 4 + 2 + 1 | 3 xe, 7 người; từng xe 4 + 2 + 1 |
| Ảnh 2 — nhãn mũ | 4 có mũ, 3 chưa xác định | 4 có mũ, 3 chưa xác định |
| Ảnh 2 — box xe thô trước lọc/ghép | 4 | 5 |

Ba đầu chưa xác định là em bé phía trước và hai người ở xa. Số box người thô không được coi là số người đúng: các box có thể trùng hoặc bỏ sót người khác. Đã kiểm tra trực quan ảnh chú thích để xác nhận 4 người ở xe trước, 2 người ở xe xa và 1 người bên phải.

| Thời gian trung vị ba lượt, giây/ảnh | YOLO26s | YOLO11s |
| --- | ---: | ---: |
| Riêng YOLO — ảnh 1 | 0,416 | 0,691 |
| Pipeline — ảnh 1 | 3,945 | 6,898 |
| Riêng YOLO — ảnh 2 | 0,828 | 0,851 |
| Pipeline — ảnh 2 | 7,245 | 7,924 |

`pipeline_seconds` lấy từ thời gian xử lý frame, vẽ và ghi video bên trong `Pipeline.run`; không gồm nạp model, đọc ảnh đầu vào, xuất CSV/evidence sau xử lý hoặc ghi PNG. Khoảng dao động pipeline: YOLO26s ảnh 1 là 3,785–6,523 s, ảnh 2 là 6,995–8,581 s; YOLO11s tương ứng 4,272–6,951 s và 7,515–7,985 s. Chưa thể suy ra YOLO26s luôn nhanh hơn từ ba lượt đo CPU này.

Các lượt kiểm tra YOLO26s đều đạt điều kiện về số người/xe và nhãn của ảnh 1. Đây là hai ảnh kiểm tra lỗi đã biết, không phải tập đánh giá độc lập và không có số liệu mAP/precision/recall mới. F2 không được áp dụng cho ảnh tĩnh; ngưỡng F2 hiện có chưa được tinh chỉnh lại trên tập video YOLO26s. Các kết quả benchmark YOLO11s/SAM cũ không được chuyển thành bằng chứng cho YOLO26s.

## Cách chạy lại

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_offline_detectors.py --unhelmeted outputs/user_review/case1_input.png --family outputs/user_review/case2_input.png --repeats 3 --out outputs/offline_yolo26
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Kết quả đo từng lượt, phiên bản thư viện và SHA-256 hai checkpoint nằm tại `outputs/offline_yolo26/comparison.json`. Ảnh và CSV nằm trong các thư mục `yolo26s/` và `yolo11s/` bên dưới.

Bộ unit test có 22 test: 20 test trước đó và 2 test mới kiểm tra nhánh Offline thực sự truyền checkpoint YOLO26s, đồng thời không tự chuyển sang SAM khi có GPU. Các test logic này tách biệt với việc chạy trọng số thật trên hai ảnh.
