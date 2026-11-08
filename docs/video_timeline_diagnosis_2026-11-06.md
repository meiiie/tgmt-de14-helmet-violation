# Chẩn đoán video 35 giây → 8 giây — 06/11/2026

**Trạng thái: chỉ điều tra và lập kế hoạch, chưa sửa code production, chưa chạy lại neural inference.** Các kiểm tra thực hiện: ffprobe metadata + đếm decoded frames, đối chiếu SHA256, đọc CSV, tái tính sampling/voting từ dữ liệu đã có. Thời gian 366.9s do người dùng cung cấp, không phải lần đo mới.

## Đúng file đã tìm được

Upload `1102.mp4` trong Gradio cache có cùng SHA256 với `path/to/1102.mp4`. Output cache có cùng SHA256 với `de14_dip18yet/video_annotated_h264.mp4`. Bằng chứng đầy đủ: `outputs/video_timeline_audit_20261106/audit.json`; [bản metadata lưu trong repo](results/video_timeline_audit_2026-11-06.json).

| Thuộc tính ffprobe | Input | Output |
| --- | ---: | ---: |
| Decoded video frames | **1052** | **80** |
| FPS (`r_frame_rate`, `avg_frame_rate`) | **30/1** | **10/1** |
| Video stream duration | **35.066667s** | **8.000000s** |
| Container duration | 35.085011s | 8.000000s |
| Resolution | 1920×1080 | 1920×1080 |
| Video codec | HEVC | H.264 |
| Audio | AAC, 44.1 kHz, stereo, 35.085011s | **Không có audio stream** |

Chênh 18ms giữa video duration và container input đến từ audio, không phải lỗi mất gần 27 giây. Chưa có GT được xác minh cho file này; không suy precision/recall hoặc ID switches từ prediction CSV.

## Nguyên nhân chính, theo file/function

1. `app.py:analyze()` đặt `maxf = int(max_sec * fps)` rồi dừng generator ở `i < maxf`. Slider “Tối đa (giây video)” mặc định **8**, nên default chỉ đọc 240 frame đầu ở nguồn 30 FPS. Đây là giới hạn thời gian nội dung, không phải timeout xử lý.
2. Generator chỉ yield frame `i % stride == 0`, với mặc định stride **3**. Dãy nguồn `0,3,…,237` có đúng **80 frame**. Không frame nào từ đoạn còn lại được đưa vào pipeline theo đường chạy này.
3. `analyze()` truyền `fps=source_fps/stride=10` vào `Pipeline`. `Pipeline.run()` chỉ viết các frame nó được yield, `cv2.VideoWriter(..., self.fps, ...)` vì thế viết 80 frame ở 10 FPS: **80/10 = 8s**.
4. CSV kết thúc ở **t=7.9s**, khớp sample cuối source frame 237. Output không chứa toàn bộ 35 giây bị tăng tốc; nó là sampled preview của prefix 8 giây.
5. `i // stride` làm mất source frame index trong CSV, nhưng hiện `fi/(source_fps/stride) = i/source_fps`: với CFR và sampling đều, timestamp này vẫn đúng cho prefix. Không quy nhầm lỗi duration cho phép chia index. Cần lưu cả source index và timestamp riêng để render đầy đủ và hỗ trợ VFR.
6. `_to_h264()` chỉ nhận video vừa tạo, không map audio từ input; OpenCV writer cũng không mang audio qua. Chuyển H.264 không thể khôi phục phần nguồn chưa đọc hay audio đã bỏ.
7. CLI `src/run_video.py:frames()/main()` không có default cap 8 giây (`max_frames=0`), nhưng cũng chỉ ghi sampled frames với FPS nguồn/stride. Nó vẫn là sampled rendering, chưa đáp ứng original FPS/full frame content.

Effective settings stride3/cap8 khớp chính xác code default, metadata và CSV; không có snapshot toàn bộ request gốc để khẳng định mọi field UI. CSV cũng khớp `n_min=3`: track #25 có 3 known observations và nhãn VIOL; track #1 có 2 nhưng UNKNOWN. CLIP/imgsz/runtime của request cần xác nhận từ summary để khóa benchmark, không tự đoán từ duration.

**Không sửa bằng cách chỉ đặt writer FPS=30:** 80 frame/30 sẽ còn 2.67s. Cũng không kéo 80 frame thành 35s bằng FPS thấp hoặc lặp một frame; cần xuất lại toàn bộ frame nguồn đúng timeline.

## Observation và sampling/tracking

`Pipeline.run()` chỉ append `obs[tid]` khi `vehicle_probability()` trả xác suất đủ quyết định: một head rõ không mũ, hoặc tất cả head đều rõ có mũ. Đầu unknown/thiếu crop có thể khiến cả quan sát xe unknown. Một sampled frame cho tối đa một vehicle observation, dù xe có nhiều người. `core.decide()` lại lọc None/NaN và khoảng mơ hồ rồi mới so `n_min`.

`n_frames` là số lần track được trả về; `n_obs` là số known vehicle observations. Track ít observation **không bị n_min xóa**, mà thành UNKNOWN; F2 là bước loại riêng. ByteTrack còn có admission/confirmation riêng, không đồng nghĩa n_min.

Replay trên 9 track đã được giữ trong CSV hiện tại, chỉ đổi gate n_min, không rerun model:

| n_min | Vi phạm | Có mũ | UNKNOWN | UNKNOWN rate |
| --- | ---: | ---: | ---: | ---: |
| 3 hiện tại | 1 | 4 | 4 | 44.4% |
| 5 | 0 | 3 | 6 | 66.7% |
| 10 | 0 | 1 | 8 | 88.9% |

Đây là replay **8 giây/stride3**, dùng probability CSV đã làm tròn; không phải benchmark full clip hoặc precision/recall. N_min không đổi lượng inference hiện tại nên không tạo tăng tốc đáng kể.

Nguồn 30 FPS, toàn clip 1052 frame:

| Stride | Khoảng cách hai inference | Inference cadence | Sample count | Span tối thiểu first→last của 5 / 10 known observations |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.0667s | 15 Hz | 526 | 0.267 / 0.600s |
| 5 | 0.1667s | 6 Hz | 211 | 0.667 / 1.500s |
| 10 | 0.3333s | 3 Hz | 106 | 1.333 / 3.000s |

Span giả định mọi sample đều được track/known; phase xuất hiện, confirmation, occlusion và unknown làm yêu cầu thực tế dài hơn. Với stride10/n_min10, xe hiện dưới 3s không thể có 10 known observations trong thời gian đó.

`tracking.py:ByteTrackWrapper` đặt `args.frame_rate` nhưng phiên bản BYTETracker đang cài không đọc field này. `max_frames_lost = args.track_buffer = 30`; counter tăng theo mỗi `update()`, không theo source frame. 30 lost updates tương ứng **2/5/10 giây** ở stride2/5/10 trên nguồn 30 FPS. KF dùng dt=1 theo sampled step; đây không tự động là lỗi khi stride đều, nhưng không phải đồng hồ giây. Không đổi buffer/KF tùy tiện trước benchmark. Frames chỉ render không được gọi tracker bằng empty detections để giả như đó là detection miss.

## F2: quan sát được và phần chưa thể biết

`core.track_stats()/keep_track()` loại nếu `disp_norm < 0.75 AND person_score < 0.7`. `person_score` là trung bình mức phủ rider evidence, không phải probability person detector. Accuracy head path đã thêm associated head vào evidence nên person miss đơn lẻ không nhất thiết làm F2 loại; nếu person/head đều miss trong nhiều frame, mean có thể giảm.

CSV cuối có **9 track: 1 VIOL, 4 OK, 4 UNKNOWN**. Observations CSV có 39 known rows, trong đó 36 thuộc track giữ lại. **Track #9** có ba row t=3.8/3.9/4.0s, p_no_helmet=0.829/0.951/0.900 nhưng vắng trong final tracks. Theo đường code hiện tại, F2 là gate bỏ track ở bước này. Do đó chỉ chứng minh được **ít nhất 10 track trước F2, ít nhất 1 bị loại, 9 được giữ**. Có thể còn unknown-only tracks bị loại mà observations CSV không ghi; không dùng max ID=32 làm tổng track.

Frame output t=3.9s vẫn vẽ #9 và head NO HELMET (`outputs/video_timeline_audit_20261106/output_t3_9.png`), bởi `Pipeline.run()` render trước khi áp F2 vào final result/CSV. Đây là sự không đồng bộ cần sửa. Chưa có geo stats của #9 hoặc GT nên chưa kết luận nó là rider thật bị bỏ hay FP đúng phải loại. Cảnh là camera di chuyển: displacement trong ảnh cũng không chứng minh xe chuyển động ngoài đời; F2 cần audit cả ego-motion và xe dừng có rider.

## Chi phí và phạm vi lượt này

80/366.9 = **0.218 sampled-processing FPS**, không phải pure neural inference FPS. Summary hiện loại model loading và H.264 conversion sau `el`, nhưng gồm decode/inference/tracking/ghi output ban đầu. Pipeline ước khoảng 4.586s/sample trong lần người dùng báo; t_det/t_cls gốc không được lưu nên không thể bóc chính xác thời gian model.

Nếu tuyến tính theo chi phí cũ: toàn clip stride2/5/10 khoảng 40.2/16.1/8.1 phút; **chỉ là dự toán workload**, không phải benchmark hay cam kết ETA. Render đủ 35s không tự làm inference nhanh hơn. Ma trận 6 cấu hình phải báo thời gian thật theo từng phase và không đánh tráo cached replay FPS thành inference FPS.

Kế hoạch thực hiện: [video full-length và benchmark](superpowers/plans/2026-11-06-full-length-video.md). Chưa sửa production hoặc khởi chạy benchmark model trong lượt lập kế hoạch này.
