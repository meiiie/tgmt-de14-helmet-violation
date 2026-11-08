# Kết quả tối ưu xuất video và giữ detection đầu trước crop

Đối chứng là bản đã chạy trên `1102.mp4`. Giữ YOLO26s, DEIMv2, DINOv3-S probe, CLIP tắt, imgsz 1280, stride 3, min observations 3, F2 bật. Ba phép đo dưới đây có phạm vi riêng; không cộng thời gian replay vào thời gian inference.

| Phép đo trên máy Intel | Trước | Sau |
|---|---:|---:|
| Xuất video từ cùng 351 kết quả đã lưu | 62,14 s | 35,74 s |
| Box đầu khớp nhãn ở 18 mốc giao với các frame đã duyệt | 8/22 | 14/22 |
| Inference mới, cùng 12 mẫu trong đoạn 3,9–5,0 s, không xuất video | 52,00 s | 52,69 s |
| Track đã xác nhận còn lại sau F2, replay toàn clip | 52 | 52 |

- `src/de14/video_io.py`: giữ decode/pipe BGR và H.264 CRF20, đổi preset `fast` → `veryfast`. Render giảm 42,5%, SSIM 0,989728 so với bản `fast`. Cả hai có 1.052 frame, 30 FPS, 1920 × 1080, video 35,0667 s và cùng hash gói AAC với nguồn. Hai cấu hình PNG overlay mất 116,5–119,6 s nên không dùng.
- `src/de14/pipeline.py`: đầu DEIM đã ghép được vào một detection xe hợp lệ được crop ngay cả khi ByteTrack chưa xác nhận xe. Kết quả từng frame có nhãn `UNCONFIRMED`. Nếu track được xác nhận sau đó, dùng lại dự đoán trên đúng crop và bỏ overlay tạm trùng. Các quan sát tạm còn lại không có phiếu trong quyết định track, không vào F2 hay CSV vi phạm.
- Replay 351 mẫu cho kết quả đối chứng trùng bản cũ; 89 lượt đầu giữ lại, 30 dự đoán được dùng lại khi track xác nhận. DINO xử lý 89 crop mới trong 13,13 s trên CPU. Tại 18 mốc đã duyệt, so khớp box đầu một đối một ở IoU ≥ 0,5: cả hai bản đều chỉ có 8/22 box trên track xác nhận; 6 box thêm nằm ở phần quan sát tạm. Còn 8 box nhãn thiếu từ bước phát hiện đầu. Chưa đủ nhãn để đo precision/recall phân loại mũ toàn clip hay ID switch. Lần inference mới trên 12 mẫu giữ thêm 3 lượt đầu và có 2 lượt dùng lại.
- Mũ lông: hành khách có `p_no_helmet` 0,160–0,279 khi đổi viền crop 0%, 5%, 15%, 30%, 50%; cả năm lần probe đều gán nhầm có mũ bảo hiểm. Mở rộng viền đến 50% còn làm người lái đầu trần chuyển sang chưa rõ. Chưa có căn cứ đổi crop/ngưỡng; cần bộ crop mũ vải, mũ lông và mũ bảo hiểm đã xác minh để đánh giá riêng trước khi chỉnh probe.

Dữ liệu đo và video: `outputs/render_tracking_20261107/`. `render_trial2/report.json` chứa thử nghiệm render; `head_benchmark/report.json` và `head_benchmark/hat_audit/crop_margin_predictions.json` chứa replay và rà soát mũ; `live_window/report.json` là đo inference mới. `1102_after.mp4` là video xuất từ replay, với đầu mới qua DINO thật và dự đoán cũ lấy từ cache khớp crop. Lần xuất này mất 40,27 s, vẫn giữ đủ 1.052 frame, 30 FPS, 1080p và hash AAC gốc. Đây là lần xuất riêng có thêm box tạm, nằm ngoài phép đo hai preset trên cùng dự đoán.

Đo lại trên Kaggle T4: dùng private Dataset cũ và notebook benchmark cũ để chạy phần setup đến hết mục 4. Đính kèm `helmet_render_tracking_source.zip` dưới dạng private input, rồi thêm cell sau (đổi đường dẫn ZIP thành đường dẫn thực tế). Giữ bản project cũ để notebook gốc vẫn kiểm tra được checksum:

```python
import runpy, shutil, subprocess, sys
from pathlib import Path

source_zip = Path('/kaggle/input/<private-dataset>/helmet_render_tracking_source.zip')
assert source_zip.is_file(), source_zip
updated = BASE / 'helmet_kaggle_video_updated'
update_project_sources = runpy.run_path(str(PROJECT/'notebooks/colab_gpu_inference_cell.py'))['update_project_sources']
shutil.copytree(PROJECT, updated)
update_project_sources(source_zip, updated)
video = updated / 'data/kaggle_video/1102.mp4'
result = BASE / 'helmet_kaggle_measured_updated'
subprocess.run([sys.executable, 'scripts/benchmark_video_runtime.py', str(video),
    '--profile', 'cloud_cuda', '--stride', '3', '--min-observations', '3',
    '--hard-cases', str(updated/'data/benchmark/crowded_20261106.json'),
    '--out', str(result)], cwd=updated, check=True)
archive = shutil.make_archive(str(BASE/'helmet_kaggle_results_updated'), 'zip', root_dir=result)
from IPython.display import FileLink, display
display(FileLink(archive))
```

Nếu còn audit của bản GPU cũ, chạy thêm `scripts/benchmark_render_export.py VIDEO OLD_AUDIT --out OUTPUT --repeats 1` từ project mới để so hai preset trên cùng host và cùng dự đoán. So sánh phiên bản qua cùng video, weights, encoder, GPU và cấu hình; script `--compare` cũ kiểm tra đồng nhất mã cho phép so CPU/GPU cùng phiên bản, nên không dùng nó để đối chiếu hai phiên bản. Các số T4 cũ (63,83 s xử lý, 103,69 s xuất) chưa đại diện cho bản sửa.
