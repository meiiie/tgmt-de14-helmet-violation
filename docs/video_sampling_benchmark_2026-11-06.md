# Video full-length và benchmark lấy mẫu — 06/11/2026

Đã sửa lỗi xuất video8giây từ nguồn35giây. Bản sửa tách lấy mẫu inference khỏi render toàn bộ nguồn. Sáu tổ hợp benchmark đều giữ **1052frame /30FPS /1920×1080 /35.0667s video /35.0850s container, AAC nguồn**. File cuối bằng renderer hiện tại ở `outputs/video_fix_20261106/final_full_length_annotated.mp4`; audio sau giải mã trùng SHA256 với nguồn. Không chạy lại NN khi render file này.

## Cấu hình và phạm vi

- Máy Intel i7-1355U, Windows; `openvino_cpu`, YOLO26s, imgsz1280, DEIMv2, DINOv3/probe, **CLIP tắt** theo lựa chọn gần nhất. FP32, ngưỡng và logic association/classification/F2 giữ nguyên.
- Ba lượt NN thật không dùng prediction cache: stride10,5,2. Mỗi lượt replay cùng xác suất chưa làm tròn với n_min5/10; cặp cùng stride dùng chung thời gian NN, không phải hai phép đo độc lập.
- `1102.mp4` SHA256: `4239a39500fbccf4c9908099ff4cce1e570c7fcd91078da6bafaaee165d9e335`.
- Một lượt/stride trên máy đang sử dụng, không cô lập CPU. App có lúc được nạp/thử cùng thời gian. Chưa đủ repeat/điều kiện để xếp hạng latency ổn định hoặc suy khác biệt model throughput chỉ từ stride. Run366.9s lịch sử không lưu cấu hình đầy đủ; không dùng nó làm đối chứng accuracy/speed có kiểm soát.
- Tải runtime49.04s được ghi riêng. Các con số “xử lý” bên dưới là vòng lấy mẫu + inference/tracking, không gồm tải model, probe PTS và render/export. Báo cáo JSON giữ các phase riêng và tổng wall của lượt đầu.

## Kết quả thực

| Stride | n_min | Mẫu NN | Xử lý mẫu (s) | Inference FPS¹ | Track trước / loại F2 / giữ | Person detections² | UNKNOWN / giữ | Dự đoán vi phạm³ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|2|5|526|1629.84|0.325|88 /26 /62|2125|48/62 =77.4%|3|
|2|10|526|1629.84|0.325|88 /26 /62|2125|54/62 =87.1%|0|
|5|5|211|917.28|0.233|41 /9 /32|852|27/32 =84.4%|0|
|5|10|211|917.28|0.233|41 /9 /32|852|30/32 =93.8%|0|
|10|5|106|386.07|0.282|10 /1 /9|446|7/9 =77.8%|0|
|10|10|106|386.07|0.282|10 /1 /9|446|9/9 =100%|0|

¹ Mẫu/(wall detector calls + classifier calls), không phải FPS playback và không phải FPS replay cache. Sampled processing FPS bao gồm decode/tracking lần lượt0.323/0.230/0.275 ởstride2/5/10.

² Tổng lượt person theo các frame lấy mẫu, không phải2125 người độc lập.

³ Chưa phải TP/vi phạm thực tế. Video chưa có nhãn chuẩn: **precision, recall, FP, FN, ID switches và GT fragmentation = N/A**. Ba dự đoán ởS2-M5 không được tự coi là3vi phạm thật; giảm3→0 khi tăngn_min cũng không tự chứng minh3FN. UNKNOWN rate có mẫu số thay đổi theo số track giữ, không dùng riêng nó để chọn stride.

Tất cả6output:1052frame,30FPS,35.0667s,1080p và audio. Mốc giữa hai inference ởstride2/5/10 là0.0667/0.1667/0.3333s. ByteTrack buffer hiện30updates, tương ứng2/5/10s; chưa đổi buffer/Kalman.

## Nút thắt và vị trí giảm coverage

Ởstride10, detector(YOLO+DEIM)370.94s, classifier5.08s; ởstride5:888.46s/17.09s; stride2:1580.69s/39.19s. Khối phát hiện chiếm phần lớn thời gian gọi model. Không suy toàn bộ phần này là DEIM nếu chưa đo riêng YOLO/DEIM.

Ví dụstride10:411lượt motorcycle từ wrapper detector →319qua ngưỡng pipeline →48vehicle-frame được tracker xuất;205đầu từ wrapper →118đầu được association ghép trước tracking →37đầu tới bước crop sau tracking. Đây là **dòng dữ liệu theo frame**, không phải recall. Một phần có thể là background/box yếu, cần GT để xác định false reject. Các count `raw_*` đã qua ngưỡng/NMS bên trong wrapper, không phải mọi logit trước NMS.

Ởstride10 không track nào đạt10known observations; n_min10 làm toàn9track được giữ thànhUNKNOWN. Bốn người trên một xe vẫn chỉ cho tối đa1phiếu/xe/mẫu. UNKNOWN và frame chỉ render không tăngn_obs. N_min không giảm số NN calls.

F2 có audit từng track, gồm cả track đã bị loại. Ởstride5, track10 tại source frame120/t4.0s cóp_nohelmet0.89954, nhưng chỉ có head/body evidence ở1/2vehicle-frame: person_score0.5, displacement0.1504 →F2loại. Overlay trước lọc: `outputs/video_fix_20261106/f2_s5_t4_before.jpg`. Tình huống tương ứng về thời gian/vị trí/xác suất với#9bị loại trong run cũ. Chưa kết luận false drop hoặc vi phạm thực; camera đang di chuyển. Video cuối và CSV cùng áp F2, không còn vẽ track đã bị F2loại như thể nó được giữ.

## Tối ưu không đổi kết quả nhận diện

Probe PTS nguồn bằngFFprobe4threads: **45.80→6.55s (6.99×)**, parsed metadata/tất cả1052PTS giống hệt. Đã áp dụng cho reader. Đây là cải thiện bước chuẩn bị nguồn, **không phải tăng7×FPS model**.

Đã thử riêngx264 trên cùng8s/240frame1080p, CRF20,4threads sau NN:

| Preset | Encode(s) | Bytes | SSIM so với video tham chiếu | PSNR(dB) |
| --- | --- | --- | --- | --- |
|fast|18.90|11039591|0.994284|49.17|
|veryfast|9.70|10222071|0.992414|47.39|
|ultrafast|1.70|17894856|0.981908|44.96|

Giữpresetfast vì các preset nhanh hơn giảm fidelity của ảnh so với tham chiếu. Phân loại đã thực hiện trước encoder nên NN không đổi; không đưa thay đổi visual vào mặc định chỉ dựa trên tốc độ. Lượt thử đầu có chạy song song bước trích ảnh annotation, đã lưu riêng `report_contended.json`; chỉ bảng repeat sau khi trích ảnh xong được dùng ở đây.

## Kiểm chứng và artifact

- **90/90 kiểm thử Python**:65baseline +25mới, gồm timeline/audio/VFR/rotation/source-path guard, voting/F2, benchmark provenance/replay, public app/CLI và7schema tests annotation. Log `outputs/video_fix_20261106/unit_tests_final.log`.
- **13/13 kiểm thử logic JavaScript** của công cụ gán nhãn: mapping pixel, nhiều đầu, review/drag cancellation, import/export. DOM/canvas được mô phỏng ở biên; browser thật chưa QA vì môi trường chặn browser automation. Đây không phải13mẫu accuracy.
- App/CLI regression trên caller trước sửa tái hiện output10/60và2/60frame; caller mới giữ60/60ở30FPS. Synthetic35s/1050frame chỉ inference80mẫu vẫn xuất1050frame.
- Replay106mẫu NN thật qua snapshot pipeline trước sửa: per-frame predictions giống hệt;9track giữ và mọi field track giống, trừ roundingtimestamp<1µs. Không gọi NN lần hai.
- Source/probe/head weights và code hashes gốc trong `outputs/benchmarks/video_sampling_20261106/provenance.json`; prediction và trace từng phase trong các thư mụcstride. Runtime metadata DINO trảrevision=null: các file snapshotHF/refmain đã hash giữa run và kiểm lại không đổi; cùng runtime dùng suốt3stride. Script benchmark mới khóa tensor encoder đã nạp +processor +library versions trước khi cho resume.
- Báo cáo gốc: `outputs/benchmarks/video_sampling_20261106/report.json`, `report.csv`; từngS2/S5/S10-M5/M10cóvideo/predictions/tracksCSV. Bản gọn có version: [video_sampling_2026-11-06.json](results/video_sampling_2026-11-06.json).

Không đổi defaultstride3/n_min3/F2threshold, không thêm model/train, không có kết quả CUDA/VRAM mới. CPU không có VRAMNVIDIA; shared memoryIrisXe chưa đo. Local/cloud tiếp tục chung runtime/runner.

Người dùng đã chọn [trang gán nhãn local và hướng dẫn](video_labeling_guide_2026-11-06.md). Sau khi có nhãn đã duyệt, so P/R/tracking trên cùng các mốc source đã gán trước khi đổi gate ByteTrack/F2 hoặc huấn luyện lại.
