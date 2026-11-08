# Debug bỏ sót nhiều người trên một xe — 06/11/2026

## Tái hiện và bằng chứng

Ảnh gốc được phục hồi nguyên file 653×434 từ upload cache của Gradio, không inference trên screenshot có cả UI. Input, hash và các tensor gốc được lưu tại `outputs/debug_crowded_20261106/`. Không thay model/trọng số, không training và không đổi confidence classifier.

Trên ảnh lỗi, DEIM phát hiện **6 đầu**: 4 đầu của xe trước có scores 0.9031, 0.9021, 0.9000, 0.8939; 2 đầu xe phía sau có scores 0.7880, 0.6286. Tất cả vượt head threshold 0.3 và sống qua head NMS IoU 0.5. Model đầu không phải bước làm mất 3 người trên xe trước.

YOLO trả các motorcycle boxes từng phần:

| Box | XYXY (pixel, làm tròn) | Confidence | Sau lọc |
| --- | --- | ---: | --- |
| Phần trước của xe chính | 359,171,586,433 | 0.5256 | Qua conf 0.3, qua ByteTrack |
| Phần sau của xe chính | 146,148,380,434 | 0.3932 | Qua conf 0.3; không qua high 0.4 / new-track 0.45 |
| Box giữa | 244,160,532,430 | 0.2470 | Bị loại bởi conf_moto 0.3 |
| Vùng mép trái trên | 0,5,34,84 | 0.4209 | Không tạo track mới ở ngưỡng 0.45 |

Trước tracking, association cũ gán **1 đầu** vào phần trước và **5 đầu** vào phần sau. Trong 5 có 3 đầu đúng của xe trước và 2 đầu của xe sau bị ghép nhầm. ByteTrack bỏ phần sau, để lại 1 đầu. Đây là tương tác của fragmentation, association và xác nhận track, không phải classifier bỏ crop.

NMS thật của YOLO trong runtime này nhận tensor `[1,84,22680]`, `end2end=False`: có NMS bình thường. Thử class-aware IoU 0.3/0.5/0.7/0.9 và agnostic không phục hồi đúng nhóm 4 người; IoU cao giữ thêm duplicates. `max_det=300` so với 1000 không bão hòa vì chỉ 11 person/motorcycle rows ở conf 0.2. DEIM xuất `[1,1240,6]`, không có giới hạn 1 head/xe trong code.

| Đổi một tham số của baseline | Kết quả trên ảnh lỗi |
| --- | --- |
| YOLO IoU 0.3 / 0.5 / 0.7 / 0.9 | 5/6/7/12 person và 3/4/4/7 motorcycle boxes; kết quả cuối vẫn 1 đầu |
| Class-agnostic thay class-aware | Cùng 7 person + 4 motorcycle, cuối vẫn 1 đầu |
| YOLO confidence 0.05 / 0.1 / 0.2 / 0.3 | Cuối vẫn 1 đầu; hạ confidence có thêm boxes yếu |
| Head confidence 0.1 / 0.3 / 0.5 | Cùng 6 đầu; hạ đến 0.05 giữ thêm 2 head candidates trùng/yếu |
| Head NMS IoU 0.3 / 0.5 / 0.7 / 0.9 | Cùng 6 đầu |
| `max_det` 300 → 1000 | Cùng kết quả; cố tình đặt 1 hoặc 3 mới làm mất motorcycle |
| Pipeline conf_moto 0.3 → 0.4 | Có vẻ được 4 đầu nhưng **sai danh tính**: gồm 2 đầu xe sau, thiếu 2 đầu xe trước |
| ByteTrack chỉ hạ high hoặc chỉ hạ new | Không có nhóm 4 đúng; hạ cả hai về 0.3 giữ nhóm 1+5, vẫn ghép nhầm |

Không chọn threshold bằng tổng số box. Ảnh `sweeps/pipeline_conf_0.4_legacy_final_heads.png` và các trường `legacy_final_heads` cho phép đối chiếu chính xác đầu nào còn/mất.

## Sửa pipeline

1. Hợp nhất hai motorcycle fragments chỉ khi có bằng chứng chung từ một person box confidence ≥0.5, mỗi phần che ít nhất 25% body, gần rời nhau (0 < IoU <0.1), thẳng hàng chiều dọc (>0.8). Chỉ nhập phần yếu dưới ngưỡng new-track; không merge bắc cầu qua nhiều box. Giữ confidence cao nhất gốc, không nâng score.
2. Ưu tiên ghép head với phần trên body rồi body–xe. Body ở phía sau có đáy quá cao so với xe trước không phải passenger của xe đó. Head không có full body vẫn được xét trong envelope xe + rider hỗ trợ, để giữ trẻ nhỏ/che khuất. Mỗi head được xử lý riêng, không giới hạn một head cho mỗi body hoặc xe.
3. Ảnh tĩnh dùng `tracking=False` tường minh: không áp high/new-track gate vốn cần chuỗi video. Box xe yếu dưới 0.45 phải có head/body hỗ trợ để được nhận; nhờ đó không đưa box rỗng ở mép tường vào kết quả. Video tiếp tục ByteTrack với ngưỡng cũ. Không hạ global confidence để chữa ảnh tĩnh.

Trên replay, chỉ merge thì vẫn gán nhầm 6 đầu vào xe trước; chỉ sửa association vẫn chia nhóm thành hai phần. Cần so từng sửa đổi và bản kết hợp. Hai ảnh regression trước đây vẫn giữ counts 2 và 1+2+4.

## Artifact và tái chạy

- Manifest manual: `data/benchmark/crowded_20261106.json`, 6 ảnh, 27 head boxes; file ảnh dưới `outputs/debug_crowded_20261106/inputs/`. Đây là **development regression**, không phải independent test. Nhãn coarse do một người review; IoU matching 0.3, không có helmet-class GT. Người mép trái ảnh `three_riders` mơ hồ được ignore rõ ràng.
- Tensor trước lọc: `<case>/raw.npz`; output model sau lọc: `<case>/detections.json`.
- Mỗi ablation có `trace/<case>/<mode>/trace.json` và PNG cho detector output, confidence, hợp nhất xe, person/head association, tracking, crop, classification; từng crop hợp lệ cũng được lưu PNG riêng.
- `trace/<case>/sweeps.json`: từng IoU/confidence/class-agnostic/max_det, rows/confidence trước và sau, head threshold/NMS, motorcycle NMS bổ sung, ngưỡng pipeline và ByteTrack. Box raw đầy đủ trong NPZ; ảnh pre-NMS chỉ vẽ top 80 để đọc được, không phải inference cap.
- `trace/<case>/comparison.png`: trái pipeline trước, phải bản sửa cho ảnh tĩnh. Replay dùng classifier abstain có chủ ý để tách lỗi geometry khỏi nhận diện mũ; không dùng thời gian replay để công bố FPS.
- `baseline_src/de14/`: snapshot numerical pipeline trước sửa, chỉ thêm trace để quan sát, cùng hash trong report. Snapshot và ảnh local không nằm trong package source/cloud.

```powershell
python scripts/debug_rider_pipeline.py --manifest data/benchmark/crowded_20261106.json --baseline-source outputs/debug_crowded_20261106/baseline_src/de14 --cache outputs/debug_crowded_20261106 --out outputs/debug_crowded_20261106/trace
python scripts/benchmark_crowded_riders.py --manifest data/benchmark/crowded_20261106.json --official-manifest data/benchmark/manifest.json --baseline-source outputs/debug_crowded_20261106/baseline_src/de14 --profile openvino_cpu --repeats 2 --official-repeats 1 --out outputs/benchmarks/crowded_20261106_final
```

Muốn tái hiện trên máy khác cần mang theo đúng input/manifest và snapshot baseline, hoặc tạo snapshot trước khi sửa. Tool kiểm tra hash ảnh trước khi dùng cache. Không lấy file cache từ ảnh khác để so sánh.

## Đánh giá và giới hạn

Kết quả real-model paired benchmark cuối được ghi riêng tại `outputs/benchmarks/crowded_20261106_final/report.json`: cùng các model đã nạp, cùng input, xen kẽ thứ tự before/after qua hai repeat; đo compute sau warmup, bỏ decode/load/export. Official HELMET test 16 frame được đánh giá riêng ở mức xe/frame theo metric contract cũ, không dùng để chọn threshold. Bản đo trung gian trước guard loại box xe yếu rỗng được giữ ở `outputs/benchmarks/crowded_20261106/report.json`; không ghi đè hoặc gán nó thành phép đo bản cuối.

[Bản tổng hợp có hashes và số liệu](results/crowded_riders_2026-11-06.json) giữ kết quả ngoài thư mục outputs được ignore.

| 6 ảnh development, 27 rider heads có nhãn | Trước | Sau |
| --- | ---: | ---: |
| Head TP / FP / FN sau toàn pipeline | 19 / 1 / 8 | 24 / 0 / 3 |
| Precision / recall giữ đúng head | 95.0% / 70.4% | 100% / 88.9% |
| Latency compute p50 / p95 (ms/ảnh) | 2722 / 3278 | 2642 / 2781 |
| FPS compute | 0.374 | 0.382 |

Hai repeat tạo 12 timing samples mỗi phiên bản, accuracy chỉ tính 6 ảnh duy nhất. Một prediction ở mép trái ảnh `three_riders` nằm trong vùng ignore đã khai báo, không tính TP/FP. Sau sửa: ảnh mới có 1 xe chính với 4 người; ảnh gia đình cũ giữ 1+2+4; ảnh chở 3 người phục hồi nhóm 3 (ngoài ra có người/xe cắt mép thuộc vùng ignore); ảnh ban đêm bỏ người cạnh đường, 4→3; ảnh mưa bỏ rider nền khỏi xe trước, nhóm 4→3 nhưng rider nền chưa được ghép xe riêng.

Riêng ảnh người dùng mới: latency p50 **2261→2586 ms** vì phải phân loại 4 crop thay cho 1; không nên suy từ median 6 ảnh rằng mọi ảnh đều nhanh hơn. 4 đầu đã được xử lý độc lập, nhưng classifier hiện giữ **3 unknown + 1 helmet**. Đợt sửa này không chứng minh đã phân loại chính xác mũ vải cho 3 đầu đó; confidence/threshold classifier không đổi.

| Official HELMET test, 16 frame / 32 GT xe / 8 positive observations | Trước | Sau |
| --- | ---: | ---: |
| Vi phạm TP / FP / FN | 5 / 3 / 3 | 5 / 1 / 3 |
| Precision / recall vi phạm | 62.5% / 62.5% | 83.3% / 62.5% |
| Unknown | 37/55 = 67.3% | 38/55 = 69.1% |
| Vehicle matching precision / recall | 47.3% / 81.3% | 47.3% / 81.3% |
| Rider-count MAE, chỉ trên 26 matched vehicles | 0.50 | 0.50 |
| Latency compute p50 / p95 (ms/frame) | 2949 / 3572 | 3339 / 6806 |
| FPS compute | 0.335 | 0.230 |

Accuracy tính một lần trên 16 frame tương quan; timing video chỉ có một repeat, có jitter lớn ở cuối run sau. **Không kết luận bản sửa làm video nhanh hơn.** Lần đo trung gian cũng được giữ, không chọn run đẹp hơn để thay số liệu cuối. Precision tăng đi kèm thêm một unknown; số TP và recall không tăng trên tập video nhỏ này. Không đánh đồng với track voting toàn clip hoặc accuracy tổng quát.

Không thay checkpoint; SHA256 được lưu cả trước/sau. Torch/DEIM đều CPU trong profile OpenVINO CPU; VRAM CUDA null, không có số GPU NVIDIA mới. **60 kiểm thử phần mềm qua**, log `outputs/debug_crowded_20261106/unit_tests_final.log`; 3 test diagnostic chạy lại sau thay đổi script cũng qua. Các fixture kiểm tra root cause, head độc lập/unknown, box xe rỗng, không merge chỉ vì overlap, khác độ sâu và trace từng lần lọc. Red logs giữ riêng. Read-only review không còn phát hiện actionable trong phạm vi sửa; compile/API/package checks có artifact riêng.

Ứng dụng đã restart trên `http://127.0.0.1:7861/`, profile `openvino_cpu`. Ba request HTTP `/analyze` thật đã qua: ảnh mới 1 xe chính/4 người, ảnh gia đình cũ 3 xe/7 người, ảnh mũ lông 2 người không mũ. Evidence `outputs/runtime/crowded_app_acceptance_20261106.json`, ảnh xuất `outputs/debug_crowded_20261106/app_after.webp`, CSV `app_after_tracks.csv`. Ảnh mới mất 7.9s ở lần gọi đầu có first-use/export overhead; không thay thế warm compute trong bảng. CLI help, compileall và diff whitespace checks cũng đã kiểm tra. ZIP cloud được cập nhật cùng source nhưng không chứa ảnh, snapshot baseline hay dữ liệu local.

Bộ 6 ảnh có 27 đầu thật, gồm cả người ở xe phía sau. Bản sửa không giả nhận các đầu nền thành passenger xe trước để tăng recall. Những head có thật nhưng chưa tìm được motorcycle phù hợp vẫn phải ghi FN, không tự đổi thành TP. Precision/recall head-preservation không phải accuracy phân loại mũ, precision nhận diện xe hoặc độ đúng ghép từng xe. Bộ này được dùng để phát triển bản sửa nên không thể coi kết quả là accuracy triển khai.

Chưa xử lý hoàn toàn xe xa: trong ảnh mới, hai đầu phía sau đã được model đầu tìm thấy nhưng motorcycle ở đó không có confidence đủ qua detector. Tương tự ảnh mưa có một rider phía sau. Giữ lỗi này trong báo cáo; không hạ toàn bộ confidence hoặc đổi/training model trong đợt sửa này.

Rule hợp nhất fragments và body geometry vẫn là heuristic. Các kiểm thử hai xe riêng/khác độ sâu và bộ ảnh hiện có không bảo đảm đúng mọi cảnh đông xe. Cần mở rộng independent validation có nhãn head–vehicle trước khi kết luận khả năng tổng quát.
