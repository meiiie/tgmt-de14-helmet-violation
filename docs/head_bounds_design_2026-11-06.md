# Box đầu và đối chiếu mũ tách biệt — 06/11/2026

Người dùng đã chọn: **giữ box đầu sát bằng DEIM, tắt CLIP để giảm chi phí khi cần**. Checkbox hiện tại gộp hai việc: bật DEIM tìm đầu và CLIP kiểm tra mũ vải; khi tắt, box đầu suy từ 28% chiều cao person, dễ rộng/lệch do tay/thân. Màu xanh của nhánh này chưa chứng minh loại mũ đúng. Thiết kế giữ model/weights/fusion thresholds và default đối chiếu; không train.

Các phương án đã xét: thu nhỏ box heuristic (rẻ nhưng không định vị được đầu lệch khỏi tâm thân); dùng model đầu hiện có ở cả hai nhánh (được người dùng chọn); thay/train model mới (chưa có bằng chứng/ngoài phạm vi). Không thu nhỏ box chỉ để ảnh trông đẹp trong khi crop phân loại vẫn lệch.

Thay đổi: RuntimeConfig thêm lựa chọn head detection độc lập, mặc định None giữ semantics API/config cũ. Gradio yêu cầu head localization khi trọng số head đã có; checkbox chỉ điều khiển CLIP. Nếu thiếu head model, nhánh không đối chiếu vẫn chạy heuristic và thông báo là ước lượng. CLI có cờ giữ/ước lượng đầu tường minh. Một factory cho local/cloud; không sao chép pipeline.

Khi chỉ đổi lựa chọn CLIP với mọi cấu hình model/thiết bị/head localization còn lại giống nhau, factory có thể dùng lại detector/head/base classifier của runtime trước. Không dùng lại nếu head mode, weights, imgsz, backend, threads, device hoặc preprocessing khác. CLIP nạp khi cần; wrapper không thay đổi base model. Cache Gradio vẫn giới hạn hai bundle.

Ảnh tĩnh render ra last_vis mà không encode MP4. Box hiển thị dùng tọa độ đầu thực, crop có margin riêng. Không điều chỉnh DINO/CLIP threshold để ép unknown thành xanh. Summary phân biệt mode/heads, unknown vẫn được tính người.

Kiểm chứng: failing unit tests cho head localization khi CLIP tắt, chia sẻ model đúng điều kiện, không encode video khi render ảnh; suite đầy đủ. Chạy raw DINO probabilities/CLIP margins trên crop thật để giải thích unknown. So legacy fast, tight+DINO, tight+DINO+CLIP trên 6 development regressions + official-test smoke subset; ghi box/count/head P/R, no-helmet P/R, unknown, latency, model IDs/shared identity. Đo inference thật sau warmup, tách load/export và không benchmark ảnh annotation. Giữ ảnh mũ lông: việc tắt CLIP có thể làm sai ca này nên không promote DINO-only thành mode chính xác hơn.
