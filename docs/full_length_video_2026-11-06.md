# Full-length video output và kiểm tra lấy mẫu

Trước sửa, app mặc định chỉ đọc 8 giây đầu. Video 1052 frame/30 FPS/35.0667 giây bị lấy 80 mẫu (stride3, giới hạn8s), rồi ghi 80 frame ở10 FPS thành8 giây. Đây là cắt phạm vi phân tích, không phải mô hình đã xử lý đủ35 giây.

## Hành vi mới

- App mặc định **giới hạn phân tích =0**, nghĩa là phân tích hết video theo stride đang chọn. Không đổi mặc định stride3, n_min3 hoặc ngưỡng F2.
- **Full-length Annotated Video**: runner đọc lại toàn bộ nguồn, ghi mỗi frame nguồn đúng một lần, giữ FPS hữu tỷ, kích thước và audio nguồn khi codec tương thích MP4. Không dùng `-shortest` để cắt video.
- Frame không inference giữ nguyên, không annotation. Chưa có nội suy/hold box: vì vậy box chỉ xuất hiện trên frame thực sự phân tích. Renderer không gọi model hoặc tracker, không cộng observation.
- Nếu chọn giới hạn phân tích8s, video full vẫn giữ đủ phần sau8s; summary/audit ghi rõ phạm vi phân tích. **Sampled Preview** là lựa chọn riêng, chỉ xuất mẫu, FPS nguồn/stride và không mux audio.
- Box/nhãn xe dùng quyết định tích lũy **tại frame đó**, không lấy kết luận tương lai gán ngược về đầu clip. Quyết định F2 cuối cùng được áp dụng nhất quán cho video và CSV; track bị loại vẫn có trong audit.
- Video VFR dùng mốc thời gian của từng frame để phân tích và xuất MP4. Bản đầy đủ giữ số frame, độ phân giải và thời lượng hiển thị của frame cuối. Audio AAC hoặc MP3 được sao chép. Các codec audio khác trong video VFR, như Vorbis, được chuyển sang AAC cho MP4. Video có metadata xoay vẫn cần chuẩn hóa hướng trước khi phân tích.

## Code và dữ liệu kiểm tra

`src/de14/video_io.py` dùng chung cho `app.py`, `src/run_video.py` và benchmark; không rẽ nhánh CPU/GPU. `pipeline.py` vẫn giữ detection/association/classification/ByteTrack cũ, chỉ bổ sung source metadata, quyết định theo thời gian và telemetry.

Môi trường cần có binary `ffmpeg` và `ffprobe` trên PATH, ngoài các Python dependencies. Trên cloud cần kiểm tra hai công cụ này trước khi chạy export video; chạy notebook/model thành công chưa xác nhận khả năng xuất video/audio.

Mỗi lượt xuất:

- `video_annotated.mp4` hoặc `video_sampled_preview.mp4`.
- `video_tracks.csv`: track được giữ; `video_observations.csv`: observation known, có `source_index`, `inference_index`, `t_sec`.
- `video_filter_audit.json`: tất cả track trước lọc, lý do giữ/loại, `person_score`, `disp_norm`, số frame có body/head, known/unknown và thresholds.
- `video_stages.jsonl`: detection boxes/confidence, association, crop hợp lệ/bị loại và xác suất từng đầu theo source frame/time.
- `video_audit.json`: toàn bộ measurements, timeline, trạng thái tracker, metadata nguồn/đầu ra, số frame ghi thật, decode/inference/render timing; app có thêm `video_runtime.json`.

Các counter là **lượt theo sampled frame**, không phải số người độc lập: `head_detections` là đầu trả về từ detector; `associated_heads` là đầu của các xe đã được tracker xuất ra, trước lọc crop/classification; `missing_head_frames` đếm vehicle-frame không có đầu. Stage log cho phép phân biệt association trước tracking và crop sau tracking. Không dùng các tỷ lệ này thay recall có GT.

`detector_output`/các count `raw_*` chỉ đầu ra wrapper detector đang dùng, đã qua ngưỡng/NMS nội bộ của wrapper; không phải toàn bộ logits chưa lọc của mạng. NMS/raw tensor sweep chi tiết tiếp tục dùng công cụ `scripts/debug_rider_pipeline.py` sẵn có; ma trận video này giữ nguyên cấu hình detector.

`sec`/`processing_seconds` đo vòng lặp lấy mẫu + inference/tracking, không gồm nạp model, probe PTS nguồn và render/encode lượt hai. `total_seconds` của runner gồm probe, vòng inference, export và kiểm tra media; tải model được ghi riêng. Trong cặp n_min dùng chung NN, thời gian render và replay được ghi riêng, không biến FPS replay thành FPS inference.

`n_obs` là số **known vehicle observations**, tối đa1/xe/frame lấy mẫu. Bốn đầu không thành bốn votes. UNKNOWN không tăng n_obs. N_min10 không xóa track nhưng có thể làm nhiều track ngắn thành UNKNOWN. Ở nguồn30 FPS, stride10 chỉ có3 lần cập nhật/giây, cần ít nhất3 giây giữa observation đầu/cuối để đạt10 votes, chưa kể bỏ sót/UNKNOWN/xác nhận track.

ByteTrack hiện giữ lost buffer30 **updates**: tương ứng2/5/10 giây ở stride2/5/10. Audit ghi cả hai đơn vị; chưa sửa buffer hay Kalman dt khi chưa có GT chứng minh lợi ích. Frame chỉ render không làm tracker già đi.

F2 giữ nguyên điều kiện loại: displacement<.75 và mean rider evidence<.7. Head độc lập vẫn là rider evidence khi body bị che; cả body/head bỏ sót vẫn có thể dẫn đến false drop. Camera chuyển động cũng khiến displacement ảnh khác chuyển động vật lý. Không tự nhận mọi track có dự đoán vi phạm đều đúng.

## Chạy local/cloud

```powershell
.venv/Scripts/python.exe src/run_video.py "path/to/1102.mp4" --profile openvino_cpu --no-accuracy --stride 3 --min-observations 3 --f2 --out outputs/full_video
```

`--max-seconds 8` hoặc `--max_frames 80` chỉ giới hạn inference. `--output-mode sampled_preview` chọn preview tường minh. CUDA dùng cùng lệnh với `--profile cloud_cuda` sau khi môi trường GPU/trọng số đã sẵn sàng; chưa đo CUDA trong lần kiểm tra này.

```powershell
.venv/Scripts/python.exe scripts/benchmark_video_sampling.py "path/to/1102.mp4" --profile openvino_cpu --imgsz 1280 --out outputs/benchmarks/video_sampling_new
```

Benchmark chạy NN thật cho mỗi stride2/5/10, rồi replay n_min5/10 bằng xác suất chưa làm tròn; không lấy FPS replay làm FPS inference. Mỗi tổ hợp có video riêng, causal labels và F2 shadow on/off. Không cache detection/classification trong phép đo này. Provenance mới khóa cả tensor DINO đã nạp và cấu hình processor; resume sai cấu hình bị từ chối trước khi đè manifest.

Video1102.mp4 chưa có temporal GT được xác minh. Precision/recall/FP/FN/ID switches/GT fragmentation phải N/A. UNKNOWN rate, số track/detection, thời gian và tính toàn vẹn media đo được mà không cần GT. Không dùng16 frame smoke cũ để tuyên bố chất lượng tracking dài hoặc tuning n_min10.

Ma trận6cấu hình đã hoàn tất: [báo cáo thực nghiệm](video_sampling_benchmark_2026-11-06.md), `outputs/benchmarks/video_sampling_20261106/report.json` và `docs/results/video_sampling_2026-11-06.json`. Nguồn1102chưa cóGT; người dùng đã chọn [công cụ gán nhãn local](video_labeling_guide_2026-11-06.md).
