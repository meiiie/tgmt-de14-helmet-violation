# Box đầu sát ở cả hai chế độ, CLIP tùy chọn — 06/11/2026

Đã thực hiện lựa chọn của người dùng: **giữ DEIM định vị từng đầu khi tắt CLIP**. Checkbox đổi tên thành “Đối chiếu mũ vải bằng CLIP (thận trọng hơn, chậm hơn)”. Không có model/trọng số mới, không training và không đổi ngưỡng fusion. Tùy chọn đối chiếu vẫn được giữ; nó có thể tránh nhầm mũ vải nhưng cũng có thể khiến một mũ bảo hiểm thật thành unknown.

## Vì sao không thể coi ảnh xanh là đúng cả bốn người

Hai file người dùng gửi là ảnh đã vẽ annotation. Tái hiện trên **ảnh gốc 653×434** trong `outputs/debug_crowded_20261106/inputs/four_riders.jpg`, không đưa box/chữ đã vẽ trở lại model.

Nhánh cũ khi tắt ưu tiên dùng YOLO person box rồi ước lượng đầu từ 28% chiều cao body. Tay và tư thế ngồi làm tâm box lệch; `head_box` còn vuông hóa/mở rộng nên crop chứa nhiều thân hoặc đầu người khác. Tái hiện cho **2 nhóm, tổng 5 đầu** (2+3), gồm body box trùng. Cả 5 đều được DINO gắn helmet dù trong ảnh có mũ lưỡi trai. Vì vậy ảnh nhiều box xanh không chứng minh đếm/loại mũ đúng.

Nhánh mới dùng cùng DEIM ở cả hai lựa chọn: **1 nhóm, 4 đầu**, box thực khoảng 71–81 pixel thay cho các vùng 124–150 pixel. Box vẽ là head box detector; crop phân loại vẫn có margin riêng 0.15, không thu nhỏ box cho đẹp trong khi crop vẫn sai. Đây là box đầu có phần đội mũ, không phải segmentation riêng vỏ mũ.

## Unknown đến từ đâu

Raw evidence tại `outputs/head_bounds_20261106/evidence.json`, các crop `head_<id>_margin_<m>.png`. Xác suất dưới đây là output probe trong crop hiện tại, không phải độ tin cậy đã hiệu chỉnh cho mọi cảnh. CLIP margin là chênh điểm tương đồng, không phải probability.

| Đầu trên xe chính | DINO p(không mũ) | CLIP margin | Tắt CLIP | Bật CLIP |
| --- | ---: | ---: | --- | --- |
| Trẻ phía trước, mũ vải sáng | 0.523 | +0.105 | Unknown | Unknown |
| Người đội mũ đen | 0.071 | +0.584 | Helmet | Unknown |
| Người đội mũ lưỡi trai | 0.340 | +0.558 | Helmet, vẫn nhầm mũ vải | Unknown |
| Người đội mũ đỏ/cam phía sau | 0.009 | −2.705 | Helmet | Helmet |

CLIP đang khiến đầu đội mũ đen abstain, còn DINO đơn lẻ chưa phân biệt đúng mũ lưỡi trai. Thay margin crop 0/0.15/0.3 được ghi riêng để chẩn đoán; không chọn threshold/crop mới chỉ để sửa đúng một ảnh. Giữ unknown khi bằng chứng không rõ, không ép thành helmet/no-helmet.

`src/build_head_crops.py` hiện tạo training crops bằng `head_box(person)` heuristic. Đây là khác biệt cần kiểm tra giữa crop training và crop DEIM mới; chưa chứng minh nó là nguyên nhân duy nhất làm sai mũ. Bước tiếp theo hợp lý là chuẩn hóa/kiểm định crop và nhãn head trên training/validation độc lập trước khi cân nhắc fit lại probe. Không đưa 6 ảnh development hoặc official test vào training.

## Code và chi phí

- `RuntimeConfig.detect_heads` tách head localization khỏi `accurate`/`verify_with_clip`. None giữ semantics cấu hình cũ. Gradio dùng head model đã có ngay cả khi checkbox CLIP tắt. Thiếu head weights thì nhánh không đối chiếu giữ fallback và summary ghi “ước lượng từ người”.
- Factory `build_runtime(config, reuse=...)` chỉ chia sẻ detector/head/base classifier nếu toàn bộ cấu hình còn lại và head mode/device giống nhau. Không reuse khác weights, imgsz, backend, threads hay preprocessing. Cache app vẫn tối đa hai bundle. CLIP có thể còn resident trong cache khi tắt, nhưng không được chạy ở nhánh đó; không tuyên bố VRAM tự giảm khi tắt.
- Ảnh tĩnh dùng `render=True, write_video=False`: giữ ảnh/CSV/evidence, không encode MP4 một frame. Video vẫn ghi MP4 như trước.
- Nhánh cũ chỉ ước lượng có 2 model hoạt động; box sát không CLIP có 3; box sát + CLIP có 4. Giữ DEIM có chi phí thật, không hứa cùng tốc độ nhánh ước lượng.
- Thời gian `11.5s` trong summary cũ được tính sau `get_runtime`, nên không thể quy toàn bộ nó cho thời gian tải trọng số. Cần tách inference, first-use/warmup và export. Việc chia sẻ model tránh nạp trùng khi đổi checkbox; bỏ MP4 tránh công việc không cần cho ảnh.

CLI dùng cùng factory/pipeline local/cloud:

```powershell
# Box sát, không CLIP: lựa chọn người dùng đã chọn
python src/run_video.py path/to/video.mp4 --profile openvino_cpu --no-accuracy
# Box sát + CLIP
python src/run_video.py path/to/video.mp4 --profile openvino_cpu
# Nhánh ước lượng cũ, tường minh
python src/run_video.py path/to/video.mp4 --profile openvino_cpu --no-accuracy --estimated-heads
```

## Nguồn benchmark và kiểm tra

`scripts/benchmark_head_modes.py` chạy ba nhánh trên cùng model instances/trọng số, cùng 6 ảnh development/27 head labels và official-test smoke 16 frame. Inference được chạy lại thật; không dùng cache detection để công bố FPS. Hai repeat ảnh đảo thứ tự mode; video một repeat và có thể drift. Tách load/switch khỏi compute/decode/export. VRAM trong phép đo là shared four-model resident process, không phải so standalone footprint 2/3/4 model.

```powershell
python scripts/benchmark_head_modes.py --manifest data/benchmark/crowded_20261106.json --official-manifest data/benchmark/manifest.json --profile openvino_cpu --repeats 2 --out outputs/benchmarks/head_modes_20261106
```

Report gốc: `outputs/benchmarks/head_modes_20261106/report.json`; ảnh đối chiếu từng case cùng thư mục. Các benchmark trước giữ nguyên lịch sử, không gán latency phiên trước cho bản này. Mẫu 6 ảnh được dùng để phát triển, không phải independent accuracy. Head P/R ở IoU 0.3 đánh giá box/đầu sau association, không phải độ đúng loại mũ hay group AP.

[Số liệu đầy đủ, config và hashes](results/head_modes_2026-11-06.json). Kết quả chạy cuối trên Intel/OpenVINO CPU:

| Nhánh | Model hoạt động | P50 ảnh / P95 (giây) | FPS compute | Head precision / recall trên 27 GT |
| --- | ---: | ---: | ---: | ---: |
| Ước lượng cũ, không CLIP | 2 | 0.85 / 1.60 | 1.012 | 55.6% / 55.6% |
| **Box sát + DINO, không CLIP** | **3** | **4.77 / 5.03** | **0.229** | **100% / 88.9%** |
| Box sát + DINO + CLIP | 4 | 5.10 / 5.42 | 0.196 | 100% / 88.9% |

Hai repeat, 6 ảnh duy nhất/27 rider heads. TP/FP/FN: nhánh cũ 15/12/12; hai nhánh box sát 24/0/3. Có một prediction ở vùng mép ảnh mơ hồ được ignore như manifest cũ. Box sát ở hai chế độ giống nhau trên cả 6 ảnh, tolerance 0.001 pixel. Đây là cải thiện localization trên development set, không phải chứng minh tất cả nhãn mũ đúng.

Riêng ảnh người dùng: nhánh cũ p50 0.78s và 2 nhóm/5 đầu; box sát không CLIP 4.87s và 1 nhóm/4 đầu; box sát có CLIP 5.11s và cùng 4 đầu. **Chọn box sát có chi phí lớn so với ước lượng.** Tắt CLIP chỉ bỏ phần đối chiếu, không bỏ chi phí DEIM. Trung bình phân loại giảm 0.769→0.393s/frame; detection vẫn khoảng 4s và là phần lớn thời gian trong run này. Không hứa tắt CLIP sẽ làm 11.5s giảm nhiều lần, hoặc mọi lần chạy cùng latency.

Factory kiểm chứng cùng detector/head và cùng base classifier (`is` true). Chuyển bundle từ có CLIP sang không CLIP mất khoảng **0.000089s**, không load lại model chung; đây là thời gian tạo bundle trong process đã có model, không phải latency ảnh hoặc khởi động app. Nạp bốn model đầu tiên trong run mất 17.85s. Cache có thể giữ CLIP resident; VRAM CPU null và không đo Iris shared memory.

| Official test smoke: 16 frame, 32 GT vehicles, 8 positive observations | Ước lượng cũ | Box sát không CLIP | Box sát + CLIP |
| --- | ---: | ---: | ---: |
| No-helmet TP / FP / FN | 6 / 2 / 2 | 5 / 1 / 3 | 5 / 1 / 3 |
| No-helmet precision / recall | 75% / 75% | 83.3% / 62.5% | 83.3% / 62.5% |
| Unknown vehicles | 32/55 | 29/55 | 38/55 |
| Rider count MAE, 26 matched vehicles | 0.423 | 0.500 | 0.500 |
| Video compute P50 / P95 (giây) | 0.72 / 1.30 | 4.40 / 4.83 | 4.62 / 5.29 |
| FPS compute | 1.245 | 0.261 | 0.212 |

**Không kết luận box sát cải thiện toàn bộ end-to-end metrics**: trên mẫu video nhỏ, precision tăng nhưng recall giảm và rider MAE chưa tốt hơn. Default verified hiện tại không đổi geometry/classifier/fusion; tắt CLIP là lựa chọn giảm đối chiếu, không được promote thành chính xác hơn. Các frame tương quan/GT có thể thiếu xe nền; video timing chỉ một repeat, không chọn threshold theo kết quả này.

Giữ tùy chọn CLIP vì hard case ngoài bảng tổng hợp: mũ lông vẫn bị DINO-only nhận helmet, còn CLIP chuyển về no-helmet đúng; đồng thời CLIP làm mũ đen của ảnh mới unknown. Vì vậy cần đánh giá per-head/hard-case cùng aggregate metrics, không nhìn số box xanh. Unknown không phải bị mất detection.

**65 unit tests đã qua**, log `outputs/head_bounds_20261106/unit_tests_final.log`. Bốn test mới có red/green logs cho giữ đầu trẻ khi CLIP tắt, reuse tránh nạp lại, từ chối reuse khác imgsz và render ảnh không VideoWriter; thêm một test chạy hết metric/export bằng fixture nhỏ, không tải model. Review độc lập không có Critical/Important; đã sửa summary để báo CLIP theo cấu hình hiệu lực. Raw classifier evidence không phải training/validation để chỉnh threshold.

Lượt benchmark đầu bị lỗi xuất metric do thiếu import, log `outputs/head_bounds_20261106/benchmark.log` được giữ và không dùng làm report hoàn chỉnh. Sau red test tái hiện lỗi và sửa import, suite đã qua. Lượt cuối ở `benchmark_final.log`; tool lưu `image_progress.json` và `official_progress.json` có `complete:false` sau từng phần, để một lỗi export không làm mất inference đã chạy. Chỉ `report.json` mới là kết quả đầy đủ.

## Kiểm tra ứng dụng thực tế

App đã khởi động bằng `python app.py` ở cổng mặc định **7861**, profile `openvino_cpu`. `outputs/head_bounds_20261106/app_acceptance.json` ghi 5 request HTTP thật: ảnh mới CLIP off/on/off đều 1 xe/4 người; ảnh gia đình CLIP off vẫn 1+2+4; ảnh mũ lông CLIP on vẫn nhận xe có người không mũ. Hai ảnh trả về được lưu `app_tight_no_clip.webp` và `app_tight_clip.webp`.

Trên các request này, thời gian xử lý summary ảnh mới: off 3.5s, on 3.8s, off lặp lại 3.3s. Đây là vài quan sát API, không phải median benchmark hoặc bảo đảm tốc độ. Request đầu có tổng wall khoảng 40.5s do khởi tạo model/upload và xử lý; summary 3.5s không bao gồm thời gian nạp model. Lần đầu bật CLIP có tổng wall khoảng 7.0s, lần off lặp lại khoảng 4.5s; cả hai đã dùng lại YOLO/head/DINO. Không quy mọi khác biệt so với 11.5s của người dùng cho một thay đổi code vì thời điểm, warmup và tải máy khác nhau.

Checkbox vẫn mặc định bật khi đủ weights để giữ hành vi thận trọng trước đây. Với lựa chọn người dùng đã chọn, **tắt checkbox CLIP rồi Phân tích**: DEIM vẫn tìm đầu, chỉ bỏ đối chiếu CLIP. Khung xanh là dự đoán của model, không phải chứng nhận nhãn đúng. Các số “chưa xác định” trong summary hiện đếm xe/track, còn nhãn từng đầu hiển thị trên box.
