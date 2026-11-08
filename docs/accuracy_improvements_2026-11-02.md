# Cải thiện nhận diện người và mũ — 02/11/2026

**Ghi nhận trước khi chuyển YOLO26s:** số đo 7–8 giây/ảnh và 20 unit test bên dưới thuộc bản YOLO11s. Xem [cấu hình mới, 22 test và kết quả đo lại YOLO26s](offline_yolo26_2026-11-02.md).

## Lỗi đã tái hiện

- Ảnh hai người trên xe: probe DINOv3 cũ cho điểm không mũ khoảng 0,39 và 0,20 trên crop đầu suy từ box toàn thân, nên cả hai bị gán có mũ. Crop chứa nhiều vai/nền; mũ lông nằm ngoài miền dữ liệu Myanmar dùng huấn luyện probe.
- Ảnh gia đình: YOLO không phát hiện toàn thân em bé phía trước; một số người/xe có nhiều box chồng nhau. Số box toàn thân không phản ánh đúng số người.
- Mã cũ biến mọi điểm dưới 0,5 thành phiếu có mũ, kể cả điểm sát 0,5. Xe không có người phát hiện vẫn được tạo một crop đầu giả định. Một người có mũ có thể khiến xe được báo có mũ trong khi người khác chưa phân loại được.
- Với YOLO helmet, phép `1 - confidence` có thể đảo nghĩa khi confidence của lớp có mũ dưới 0,5.

## Thay đổi

1. Thêm chế độ Offline **Ưu tiên độ chính xác**: DEIMv2 Wholebody49 bản đầy đủ tìm đầu độc lập với box toàn thân. Loại box xe/đầu trùng và gán mỗi đầu cho tối đa một xe, không giới hạn ba người/xe. Bản HGNetV2-N không được dùng cho chế độ này vì thử nghiệm còn bỏ sót người.
2. Crop theo đầu thật, vuông hóa và thêm lề 15%. DINOv3 + probe hiện tại vẫn là bộ phân loại chính. CLIP ViT-B/32 đối chiếu mũ bảo hiểm với đầu trần/mũ vải/mũ lông/mũ nắng.
3. Khi ngữ nghĩa mũ thường rõ ràng, cho phép sửa nhầm lẫn của probe; khi bằng chứng mâu thuẫn hoặc crop nhỏ hơn 32 px, trả chưa xác định. Điểm CLIP và các ngưỡng hợp nhất là heuristic, **chưa hiệu chỉnh xác suất trên tập độc lập**.
4. Phiếu có độ chắc chắn thấp bị loại khỏi quyết định track. Chỉ kết luận cả xe có mũ khi tất cả đầu phát hiện đều có bằng chứng có mũ; một người chưa phân loại không được tự coi là có mũ. Không suy ra có người đội mũ chỉ từ một box xe.
5. Bộ lọc F2 nhận đầu đã ghép với xe làm bằng chứng có người, để không bỏ nhầm người bị che toàn thân khi xe đứng yên. Trong chế độ này, cột `person_score` phản ánh độ phủ của bằng chứng người (toàn thân hoặc đầu) trong vùng rider.
6. Ghi số người phát hiện trên hình, bảng và CSV (`n_riders`). Với video, đây là số đầu lớn nhất quan sát được trên một track, không phải số người duy nhất trong toàn video.

DEIMv2 + CLIP chỉ được tích hợp vào tab Offline. Tab Realtime vẫn dùng detector/probe trước đó; quy tắc bỏ phiếu thận trọng hơn là phần lõi dùng chung.

## Kết quả chạy thật

Nguồn ảnh là ảnh gốc trong cache Gradio tương ứng hai ảnh chụp màn hình người dùng. Không huấn luyện lại trọng số bằng hai ảnh này. Chúng là các trường hợp kiểm tra chấp nhận, **không phải tập đánh giá độc lập**; lựa chọn pipeline đã được xem xét trên chúng.

| Trường hợp | Kết quả |
| --- | --- |
| Hai người không mũ | Đếm 2 người; cả 2 được gán không đội mũ |
| Xe gia đình phía trước | Đếm đủ 4 người, bao gồm em bé phía trước |
| Xe nhỏ phía sau | Đếm đủ 2 người; chưa xác định loại mũ do đầu rất nhỏ |
| Xe bên phải | 1 người, có mũ |
| Tổng ảnh gia đình | 7 người, 3 xe; 4 đầu có mũ, 3 đầu chưa xác định (em bé phía trước và 2 người phía xa) |

CPU PyTorch 2.15.0, Ultralytics 8.4.193, ONNX Runtime 1.31.1: khoảng 7–8 giây/ảnh sau khi nạp model trong lần kiểm tra này. Nạp lần đầu tốn thêm thời gian; đây không phải benchmark FPS video. 20 unit test đi qua, gồm kiểm tra worker tiếp tục hoạt động sau lỗi classifier. Test worker cũ phụ thuộc một thư mục video không có cục bộ đã được thay bằng kiểm tra hàng đợi thật, không cần dataset.

Chưa đo lại precision/recall trên HELMET test hoặc bộ giao thông Việt Nam độc lập. Các số liệu kernel cũ trong README thuộc pipeline/cấu hình cũ; không áp dụng trực tiếp cho chế độ mới. Gán đầu–xe vẫn là hình học nên có thể nhầm người đi bộ gần xe; người che khuất hoàn toàn vẫn có thể bị bỏ sót.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_accuracy_models.py
.\.venv\Scripts\python.exe app.py --port 7860
```

Trọng số bổ sung khoảng 810 MB; script giữ checkpoint DINO/probe hiện có. Sau khi khởi động lại app, chọn backend DINOv3 và bật **Ưu tiên độ chính xác** (tự bật nếu đã có trọng số bổ sung). Bỏ chọn để dùng đường chạy nhẹ hơn.

Kiểm tra lại hai ảnh gốc đã phục hồi cục bộ:

```powershell
.\.venv\Scripts\python.exe scripts/check_user_accuracy.py --unhelmeted outputs/user_review/case1_input.png --family outputs/user_review/case2_input.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Ảnh, CSV và báo cáo được lưu dưới `outputs/user_accuracy/`. Ảnh người dùng và trọng số nằm trong thư mục được Git bỏ qua.

## Nguồn trọng số

- DEIMv2 Wholebody49: [mô tả và giấy phép](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49), [release ONNX](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/tag/weights).
- CLIP: [model card](https://huggingface.co/openai/clip-vit-base-patch32), [mã nguồn và giấy phép MIT](https://github.com/openai/CLIP). Chỉ dùng để đối chiếu ngữ nghĩa, không giả định model card cung cấp độ chính xác cho bài toán mũ bảo hiểm.
