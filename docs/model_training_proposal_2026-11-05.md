# Đề xuất cải thiện accuracy và giảm model sau benchmark

## Vấn đề đã có bằng chứng

[Đợt đo hiện tại](runtime_performance_2026-11-05.md) cải thiện runtime nhưng không cải thiện accuracy: vi phạm TP 5, FP 3, FN 3 trên 8 quan sát positive; 37/55 dự đoán xe chưa xác định. Mẫu rất nhỏ và có tương quan. Chưa đủ cơ sở để chọn backbone mới hoặc tuyên bố training sẽ giải quyết mọi lỗi.

- Hai FN là quan sát GT-positive chưa khớp dự đoán xe. Audit có raw/union candidates tại các frame liên quan (ví dụ frame 59 Bago_rural_15, confidence khoảng 0.573, union IoU khoảng 0.936). Cần xác định NMS, ghép người–xe, ByteTrack startup hay metric-box mismatch trước khi quy lỗi cho YOLO.
- Một FN là xe GT-positive khớp nhưng không có đầu, nên kết quả unknown. Cần xem crop/head association và mức chi tiết thực tế của ảnh, chưa thể quy hoàn toàn cho classifier.
- Bỏ CLIP gây lỗi mũ lông trong ảnh người dùng dù aggregate P/R không đổi. Giảm model cần bộ hard negatives có nhãn head, không chỉ GT vi phạm cấp xe.
- Probe hiện có khoảng 1419 crop từ 39 training clips; cách lấy nhãn đồng nhất theo xe bỏ qua nhóm người có/không mũ trộn lẫn. Nhãn head thật cho trẻ em/mũ thời trang/đầu xa vẫn thiếu.

## Phương án và thứ tự thực nghiệm

| Phương án | Vấn đề muốn giải quyết | Điều kiện đánh giá trước khi dùng |
| --- | --- | --- |
| Audit trên validation, chạy thêm ngữ cảnh tracking | Miss dù raw candidate tồn tại | Xác định stage gây mất, so frame và full-track trên cùng clip, không giảm precision để tăng count giả |
| Nhãn head/crop thực + train lại DINO linear probe | Crop đúng nhưng nhầm mũ thời trang hoặc domain mới | Group split; đo no-helmet P/R/F1, coverage, hats/children, hai ảnh regression và latency |
| Fine-tune detector đầu nhỏ trên GPU | GT head không được phát hiện dù đủ chi tiết | Head recall/IoU theo pixel size/occlusion; kiểm tra false heads và chi phí inference |
| Dùng teacher để distill classifier nhỏ hoặc detector đa nhiệm | Thay DINO/CLIP hoặc giảm model | Chỉ sau baseline có nhãn; so cả crop, người/xe, vi phạm và latency end-to-end; không nhận candidate sai mũ lông |
| INT8/FP16 hoặc xuất thêm YOLO/DINO sang ONNX/OpenVINO | Những stage còn tốn thời gian | Tách từng thay đổi, parity/accuracy kiểm tra trước; calibration chỉ dùng training/validation, đo lại thiết bị đích |

Đây là đề xuất experiment, chưa phải kết quả. Giữ checkpoint baseline và ngưỡng hiện tại. Trước hết dùng validation và video đầy đủ để chẩn đoán, không chỉnh ngưỡng theo 16 official-test frames. Thay đổi association/tracking phải có fixture từ validation, có kết quả trước/sau và kiểm tra duplicate/rider ownership.

## Dữ liệu bổ sung và chống rò split

Thu video giao thông thực có nguồn/quyền sử dụng rõ ràng, ưu tiên cảnh Việt Nam: trẻ nhỏ phía trước, nhiều người/xe, đầu xa/bị che, mũ lông/cap/mũ vải, blur, tối và góc camera khác. Ghi nguồn, giấy phép, clip/location/time, camera, độ phân giải và hash. Lưu original-resolution frames khi có; ảnh resize từ mirror không tạo thêm chi tiết.

Gán box đầu, liên kết đầu–người–xe, nhãn `helmet/no_helmet/unknown` cho từng đầu. Không dùng nhãn xe để mặc định mọi người cùng loại mũ. Mẫu mơ hồ cần người thứ hai review; giữ unknown nếu không thấy rõ. Annotation đủ cả người có mũ để không làm lệch phân phối. Dùng teacher/pseudo labels chỉ trong training và sửa thủ công hard cases.

Chia train/validation/test theo clip **và location trước khi sinh crop**; không trộn frame kề nhau vào nhiều split. Giữ official test hiện tại và hai ảnh người dùng làm regression ngoài training. Không mine hard examples từ test để bổ sung training; thu case tương tự từ nguồn training độc lập. Kiểm tra video/crop trùng giữa các nguồn. Thống kê số head thật, size, occlusion, hats, trẻ em và số clip độc lập theo split trước khi chọn lượng training.

## Protocol và tiêu chí nhận candidate

Khóa manifest, hashes checkpoint, seed, package/model revision và ngưỡng trước khi chạy. Dùng validation để chọn candidate; test chỉ để so bản đã chốt. So baseline/candidate trên cùng GPU cho training/inference cloud và trên Intel đích cho deploy; GPU nhanh hơn không chứng minh code tối ưu hơn khi khác phần cứng.

Đo head detection precision/recall theo size; no-helmet crop precision/recall/F1/AUROC cùng coverage; rider count MAE/exact; vi phạm cấp xe/frame và cấp track riêng. Báo unknown và missed vehicles cùng P/R, phân tích hats/children/occluded. Báo p50/p95, compute FPS và video end-to-end, cold load, actual providers, peak VRAM (Torch/NVML rõ phạm vi). Giữ các lỗi định tính tiêu biểu bên cạnh bảng tổng hợp.

Gate ban đầu: hai ảnh regression phải qua; không giảm no-helmet recall hoặc precision trên validation; không tăng unknown để tạo precision đẹp; đạt latency tốt hơn trên phần cứng đích nếu mục tiêu là giảm model. Sau khi có bộ validation đủ lớn, chốt tolerance và confidence interval theo clip trước khi xem test. Không thể bảo đảm accuracy đại diện từ 16 frame hiện có.

Huấn luyện probe là thử nghiệm ít tốn kém đầu tiên **khi đã xác nhận crop đúng nhưng classifier yếu**. Dùng cùng `src/build_head_crops.py` / `src/train_probe.py` trên Colab/Kaggle, tham khảo [workflow chung](cloud_workflow_2026-11-05.md). Script training hiện có thể ghi đè `weights/dino_probe.joblib`: chạy trong workspace candidate riêng và lưu provenance; không ghi đè baseline đang phục vụ ứng dụng. Không chạy retraining tự động bằng Run all notebook. Trong phiên hiện tại chưa có training hoặc model candidate mới.
