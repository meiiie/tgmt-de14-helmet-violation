# Đề 14 – Phát hiện người đi xe máy không đội mũ bảo hiểm (SAM 3.1 + DINOv3)

Pipeline: **SAM 3.1** (prompt văn bản `motorcycle` / `person` / `head`) → ghép rider–xe (người lái + người ngồi sau) → **ByteTrack** → crop đầu → **DINOv3 ViT-S/16 + linear probe** (hoặc YOLO helmet) → bỏ phiếu theo track → `không đội mũ` / `có đội mũ` / **`chưa xác định`**. Đầu ra gồm CSV track (kèm mốc thời gian), CSV các quan sát, ảnh crop bằng chứng `evidence/<video>_track<ID>_t<giây>s_p<prob>.jpg` và video có chú thích.

## Kết quả đo (CPU box dùng chung 8 vCPU, load avg 25–80; đo ngày 23/10/2026)
Dữ liệu: HELMET (Myanmar, Lin & Siebert 2020, CC-BY 4.0 theo OSF). Annotation lấy từ Kaggle mirror `hozngvan/helmet-detection`; frame 640×640 (ảnh gốc 1920×1080 đã bị co về 640×640) lấy từ HF `eiglesiasr/nohelmetnet`. Train và test tách **theo clip**, dùng `data_split.csv` chính thức.

### A. Phân loại crop đầu (618 crop từ 10 clip *training*, 249 không mũ; GroupKFold 5 theo clip)
| Phương pháp | Precision (không mũ) | Recall | F1 | AUC | Coverage |
|---|---|---|---|---|---|
| **DINOv3 ViT-S/16 + linear probe** | **0.579** | **0.558** | **0.569** | **0.692** | 100% |
| DINOv3 ViT-S/16 + kNN (k=7) | 0.433 | 0.349 | 0.387 | 0.549 | 100% |
| YOLO11s helmet (nnsohamnn), chạy trên crop | 0.396 | 0.492 | 0.439 | 0.579 | 55.2% |
Tốc độ embedding DINOv3-S trên CPU (box quá tải): 1.9 crop/s.

### B. SAM 3.1 trên CPU (đo thật)
- Đã tải checkpoint gated `facebook/sam3.1/sam3.1_multiplex.pt` (cần HF_TOKEN). Phần detector nạp vào image model: thiếu 4 key, thừa 36 key (lớp `convs.3`/`interactive_convs`, không thuộc nhánh detector).
- Mỗi frame chạy 3 prompt mất **91–173 s** (mã hoá ảnh ≈ 12.6 s, mỗi prompt ≈ 30 s), tức **≈ 0.007–0.011 FPS**. Trên CPU có rò bộ nhớ (lên tới ≈ 7 GB RSS rồi bị OOM-kill), nên phải chạy 1 tiến trình cho mỗi frame (`src/run_sam31_cpu_sample.sh`).
- Đã cache 20 frame (Bago_urban_3, 2 fps). Ví dụ frame 001: 14 motorcycle, 11 person, 8 head.

### C. Đánh giá theo track trên GPU (Kaggle kernel v2, 2× Tesla T4, chạy xong 24/10/2026)
Kernel [`holilihu/de14-helmet-sam31-dinov3`](https://www.kaggle.com/code/holilihu/de14-helmet-sam31-dinov3). 7 clip test HELMET, stride 2, 50 frame/clip (350 frame), 51 GT track, trong đó 19 track vi phạm. Số liệu gốc ở `docs/results/kaggle_v2/*.json`.

| Phương pháp | Precision (track khớp GT) | Precision (tính cả track không khớp GT) | Recall (trên 19 GT vi phạm) | Recall (track đã quyết định) | Tỷ lệ chưa xác định | Coverage | FPS |
|---|---|---|---|---|---|---|---|
| B0: YOLO helmet theo từng frame (chạy lại ở kernel v3 sau khi sửa `.cpu()`; gán GT bằng oracle) | 0.667 | – | 0.526 | 0.667 | 0.235 | 1.000 | 17.1 |
| B1: YOLO11s + ByteTrack + YOLO helmet | 0.455 | 0.313 | 0.263 | 0.556 | 0.429 | 0.686 | 2.27 |
| B1': YOLO11s + ByteTrack + DINOv3-B probe | 0.786 | 0.268 | 0.579 | 0.917 | 0.200 | 0.686 | 1.98 |
| Core: SAM 3.1 + ByteTrack + YOLO helmet | 0.250 | 0.083 | 0.053 | 0.125 | 0.500 | 0.745 | 9.50* |
| **Core: SAM 3.1 + ByteTrack + DINOv3-B probe** | **1.000** (11 TP, 0 FP) | 0.212 (41 track không khớp GT) | **0.579** | 0.846 | **0.184** | **0.745** | 6.09* |

\* FPS của Core chỉ tính bước tracking + phân loại, vì SAM 3.1 đã cache trước. Bản thân SAM 3.1 (4 prompt `motorcycle,person,head,helmet`, fp32, T4, không Flash Attention) mất khoảng 259 s cho mỗi 50 frame, tức **≈ 0.19 FPS**, nên FPS đầu-cuối thật của Core ≈ 0.19.

Điểm yếu chính: precision giảm mạnh khi tính các track dự đoán không khớp GT (xe đỗ, xe ở xa, hoặc xe mà HELMET không gán nhãn). Cần kiểm tra thủ công một mẫu track này trước khi coi là FP thật.

Linear probe theo crop (2809 crop, 725 không mũ, 39 clip, CV theo clip):

| Backbone | Precision | Recall | F1 | AUC | Tốc độ embed (T4) |
|---|---|---|---|---|---|
| DINOv3 ViT-S/16 | 0.648 | 0.806 | 0.718 | 0.903 | 205.7 crop/s |
| DINOv3 ViT-B/16 | 0.652 | 0.801 | 0.719 | 0.911 | 78.5 crop/s |
| DINOv3 ViT-L/16 (kernel v3, trọng số trong dataset riêng, sha256 OK) | 0.709 | 0.850 | 0.773 | 0.937 | |
| YOLO helmet trên crop | 0.301 | 0.508 | 0.378 | 0.658 | coverage 50% |

### D. Kernel v3 (24/10/2026, 1× Tesla T4): B0, SAM 3.1 fp16, ablation prompt, probe ViT-L
Cùng 7 clip, 350 frame, 51 GT track (19 vi phạm). SAM 3.1 chạy autocast **fp16** (T4 không hỗ trợ bf16 thật). Model nạp một lần cho mọi clip. Thời gian `set_image` và từng prompt được đo riêng.
- FPS end-to-end = frame / (thời gian SAM của các prompt dùng + pipeline).
- 4p = motorcycle, person, head, helmet; 2p = motorcycle, person, khi đó vùng đầu suy ra từ box người/xe.
- Số liệu gốc ở `docs/results/kaggle_v3/`.

| Phương pháp | Precision (track khớp) | Precision (gồm track không khớp) | Recall | Chưa xác định | FPS end-to-end |
|---|---|---|---|---|---|
| B0: YOLO helmet toàn khung, từng frame | 0.667 | 0.667* | 0.526 | 0.235 | 17.1 |
| Core 4p + DINOv3-S (mặc định app) | 0.800 | 0.185 | 0.632 | 0.184 | 0.57 |
| Core 2p + DINOv3-S | 0.722 | 0.188 | 0.684 | 0.158 | 0.66 |
| Core 4p + DINOv3-B | 1.000 | 0.160 | 0.421 | 0.237 | 0.56 |
| Core 2p + DINOv3-B | 0.750 | 0.194 | 0.632 | 0.158 | 0.64 |
| Core 4p + DINOv3-L | 0.917 | 0.234 | 0.579 | 0.158 | 0.47 |
| Core 2p + DINOv3-L | 0.765 | 0.245 | 0.684 | 0.158 | 0.52 |

\*B0 gán box helmet vào GT bằng oracle nên không có track không khớp. Ở mức frame, B0 có P 0.607, R 0.145 và 73.6% frame chưa xác định.

**Thời gian SAM 3.1 fp16 trên T4** (bỏ frame khởi động):
- `set_image` mất 1.16 s, mỗi prompt khoảng 0.11–0.12 s.
- 4 prompt: 1.62 s/frame (0.62 FPS). 2 prompt: 1.40 s/frame (0.72 FPS).
- Một lần chạy 2 prompt riêng (19 frame) đo được 1.45 s/frame (0.69 FPS), khớp với số trên.
- So với v2 (~0.19 FPS) nhanh hơn khoảng 3×, chủ yếu nhờ nạp model một lần và fp16.

**Kiểm tra tay 20 track không khớp** (`docs/results/unmatched_review.md`):
- 15 là FP thật: 13 xe đỗ, 2 vùng nền.
- 2 là box trùng của xe đã có nhãn.
- 0 xe đang chạy mà thiếu nhãn.
- 3 chưa xác định.
- Vì vậy cột "gồm track không khớp" sát với precision thật hơn.

## Chạy
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # + torch CPU/GPU
git clone https://github.com/facebookresearch/sam3 third_party_sam3 && (cd third_party_sam3 && git apply ../patches/sam3_cpu.patch && pip install --no-deps -e .)
set -a; source /workspace/tgmt/.env; set +a        # HF_TOKEN, KAGGLE_API_TOKEN — không commit
bash scripts/get_data.sh
python src/build_head_crops.py && python src/train_probe.py
python src/eval_helmet.py --method b0
python src/eval_helmet.py --method b1 --detector sam31 --classifier dino
python src/run_video.py <video_or_dir> --detector sam31 --classifier dino --stride 3 --out outputs/demo
```
GPU: `notebooks/de14_colab_kaggle.ipynb` (Colab/Kaggle T4, secrets qua Colab/Kaggle Secrets).

## Ước lượng trên T4 (CHƯA ĐO)
- SAM 3.1 có 848M tham số. Chạy fp16 cần khoảng 2–3 GB VRAM trọng số, tổng khoảng 6–8 GB ở độ phân giải 1008; vừa T4 16 GB nếu batch = 1.
- T4 không có FA3 và không hỗ trợ bf16 phần cứng, nên dùng fp16 hoặc `use_fa3=False`. Ước lượng thô: 1–3 FPS với 3 prompt.
- DINOv3-S: khoảng 200–500 crop/s.

## Công nghệ
| Thành phần | Model / phiên bản | Thư viện | License | Lý do | Link |
|---|---|---|---|---|---|
| Phát hiện + phân đoạn theo text (lõi) | SAM 3.1 (`sam3.1_multiplex.pt`, phát hành 27/03/2026) | `sam3` (repo Meta) + patch CPU | SAM License (gated) | Open-vocabulary, có head/person/motorcycle, tracking multiplex | https://github.com/facebookresearch/sam3 · https://huggingface.co/facebook/sam3.1 |
| Embedding đầu (lõi) | DINOv3 ViT-S/16 LVD-1689M | transformers 5.20.2 | DINOv3 License (gated) | Đặc trưng tự giám sát mạnh, probe tuyến tính rẻ | https://github.com/facebookresearch/dinov3 · https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m |
| Probe | Logistic regression / kNN | scikit-learn | BSD | Không cần fine-tune GPU | |
| Tracking | ByteTrack | ultralytics 8.4.184 | AGPL-3.0 | Chuẩn MOT, dùng cho cả SAM và YOLO | https://github.com/ifzhang/ByteTrack |
| Baseline detector | YOLO11s COCO | ultralytics 8.4.184 | AGPL-3.0 | Baseline nhanh | https://github.com/ultralytics/ultralytics |
| Baseline helmet | YOLO11s helmet (nnsohamnn) | ultralytics | MIT | Pretrained helmet/không mũ | https://huggingface.co/nnsohamnn/helmet-detection-yolo11 |
| Dữ liệu | HELMET (Myanmar) | Kaggle API 2.2.5 | CC-BY 4.0 (OSF) / Kaggle ghi "Unknown" | Video + track, Đông Nam Á | https://osf.io/4pwj8/ · https://www.kaggle.com/datasets/hozngvan/helmet-detection · https://huggingface.co/datasets/eiglesiasr/nohelmetnet |
| Video demo | "Stratkruciĝo en Hanojo" | — | CC BY-SA 4.0 | Giao thông Hà Nội | https://commons.wikimedia.org/wiki/File:Stratkruci%C4%9Do_en_Hanojo.webm |
| Runtime | Python 3.13.5 (đề yêu cầu 3.12), torch 2.14.1+cpu | | | Box chỉ có CPU | |

### Giấy phép & trích dẫn dữ liệu HELMET
- Nguồn gốc: **HELMET dataset**, OSF https://osf.io/4pwj8/, giấy phép **CC-BY 4.0**.
- Bài báo gốc: H. Lin, J. D. Deng, D. Albers, F. W. Siebert, "Helmet Use Detection of Tracked Motorcycles Using CNN-Based Multi-Task Learning", *IEEE Access* 8, 2020. doi:10.1109/ACCESS.2020.3021357
- Kaggle `hozngvan/helmet-detection` và HF `eiglesiasr/nohelmetnet` (ảnh co về 640×640) **chỉ là bản sao (mirror)** do bên thứ ba tải lên. Trang Kaggle ghi license "Unknown", nên báo cáo dẫn license gốc CC-BY 4.0 từ OSF.

## Demo Gradio (`app.py`)
Có thể tải lên **ảnh hoặc video**. App phát hiện xe máy và rider (cả người lái lẫn người ngồi sau) rồi vẽ box: box đầu màu đỏ là không mũ, xanh là có mũ, vàng là chưa xác định. Với video, mỗi track được bỏ phiếu bằng ByteTrack. Kết quả gồm **bảng theo track** (track id, khoảng thời gian, nhãn, tỉ lệ không mũ, số quan sát), **gallery ảnh bằng chứng**, và **CSV** để tải về.

Backend:
- `YOLO11s + YOLO helmet (CPU)`: mặc định.
- `YOLO11s + DINOv3-S probe (CPU)`: cần `weights/dino_probe.joblib`, tạo bằng `src/train_probe.py`, và quyền truy cập DINOv3.
- `SAM 3.1 + DINOv3-S probe (GPU)`: cần checkpoint gated, lấy qua `HF_TOKEN` hoặc `SAM31_CKPT`.

```bash
# local (CPU)
. .venv/bin/activate && python app.py --port 7860          # mở http://localhost:7860
# Colab / Kaggle (GPU): sau khi cài theo notebook
python app.py --share                                       # in ra link public *.gradio.live
python scripts/screenshots.py http://127.0.0.1:7860         # chụp màn hình bằng Playwright (Chrome headless)
```
Kết quả chạy thật trên CPU của box ngày 24/10/2026:
- Ảnh HELMET Bago_urban_3 frame 001, phóng lại về 1920×1080, backend DINOv3-S: phát hiện 6 xe, gán 3 "không mũ" và 3 "có mũ", mất 3.4 s. GT của frame này chỉ có 1 xe với người lái có mũ, nên đây là **false positive**: phần lớn là xe đỗ, và đầu chỉ khoảng 10 px.
- Clip Hà Nội 6 s (640×480, CC BY-SA 4.0), backend YOLO helmet, lấy mỗi 3 frame (30 frame): phát hiện 9–10 track, 1 "không mũ", 7–9 "chưa xác định", tốc độ 2.3–6.1 FPS.

![image](docs/screenshots/01_image_helmet_dinov3.jpg)
![video](docs/screenshots/02_video_hanoi_yolohelmet.jpg)

### E. Kernel v4 (24/10/2026, T4): bộ lọc track, dùng lại cache SAM 3.1 fp16 của v3
SAM 3.1 không chạy lại: cache v3 nằm trong dataset riêng `holilihu/de14-sam31-cache-v3`.

**Đặc trưng theo track** (`src/de14/core.py`):
- `person_score` = trung bình theo frame của max IoA(box person đã ghép, vùng rider của xe). Vùng rider là box xe mở rộng ngang 25% và kéo lên trên 1 chiều cao xe.
- `disp_norm` = đường chéo hộp bao quỹ đạo tâm xe chia cho trung vị đường chéo box xe.

**Bộ lọc:**
- F1: giữ track nếu `person_score ≥ τp`.
- F2: bỏ track nếu `disp_norm < τd` **và** `person_score < τs`. Nhờ vậy rider dừng đèn đỏ vẫn được giữ.
- F1+F2: áp cả hai.

**Chọn ngưỡng** (`scripts/select_filters.py`, không gán cứng):
- Sắp xếp 7 clip test theo tên. Fold tinh chỉnh = clip ở chỉ số 0, 3, 6 (Bago_highway_10, Bago_rural_49, Mandalay_1_105).
- Grid search tối đa F1 tính với P gồm cả track không khớp, rồi đánh giá trên 4 clip còn lại (Bago_highway_20, Bago_rural_3, Bago_urban_3, Bago_urban_50; 10 GT track vi phạm).
- Mỗi cấu hình chọn ngưỡng riêng.

Kết quả trên 4 clip đánh giá. Pu = P gồm track không khớp; F1u dùng Pu; FPS end-to-end tính cả thời gian SAM.

| Cấu hình | Bộ lọc | Ngưỡng | P | Pu | R | Chưa xác định | F1u | FPS |
|---|---|---|---|---|---|---|---|---|
| 4p + ViT-S | không | – | 0.700 | 0.179 | 0.700 | 0.190 | 0.286 | 0.58 |
| 4p + ViT-S | F1 | τp=0.2 | 0.700 | 0.318 | 0.700 | 0.190 | 0.438 | 0.58 |
| **4p + ViT-S** | **F2** | **τd=0.75, τs=0.7** | 0.700 | **0.333** | 0.700 | 0.190 | **0.452** | 0.58 |
| 4p + ViT-S | F1+F2 | τp=0.2, τd=0.1, τs=0.7 | 0.700 | 0.318 | 0.700 | 0.190 | 0.438 | 0.58 |
| 2p + ViT-S | không | – | 0.636 | 0.179 | 0.700 | 0.190 | 0.286 | 0.67 |
| 2p + ViT-S | F1 | τp=0.2 | 0.636 | 0.304 | 0.700 | 0.190 | 0.424 | 0.67 |
| 2p + ViT-S | F2 | τd=0.75, τs=0.7 | 0.636 | 0.318 | 0.700 | 0.190 | 0.438 | 0.67 |
| 2p + ViT-S | F1+F2 | τp=0.2, τd=0.1, τs=0.7 | 0.636 | 0.304 | 0.700 | 0.190 | 0.424 | 0.67 |
| 4p + ViT-L | không | – | 0.857 | 0.200 | 0.600 | 0.143 | 0.300 | 0.49 |
| 4p + ViT-L | F1 / F2 / F1+F2 | τp=0.1 / τd=0.5, τs=0.1 / τp=0.1, τd=0.05, τs=0.1 | 0.857 | 0.300 | 0.600 | 0.143 | 0.400 | 0.49 |
| 2p + ViT-L | không | – | 0.700 | 0.206 | 0.700 | 0.190 | 0.318 | 0.55 |
| 2p + ViT-L | F1 / F2 / F1+F2 | τp=0.1 / τd=0.5, τs=0.1 / τp=0.1, τd=0.05, τs=0.1 | 0.700 | 0.304 | 0.700 | 0.190 | 0.424 | 0.55 |

Bộ lọc không làm mất TP nào (R giữ nguyên). Chúng chỉ loại 10–18 track không khớp GT, đa số là xe đỗ, đúng như kết quả kiểm tra tay ở `docs/results/unmatched_review.md`.

**Cấu hình mặc định của app: SAM 3.1 (4 prompt, fp16) + DINOv3 ViT-S + F2 (τd=0.75, τs=0.7).**
- Lý do: F1u cao nhất trên clip đánh giá (0.452), nhanh hơn ViT-L, và trùng với chỉ đạo giữ ViT-S làm mặc định.
- `app.py` đọc `configs/track_filter.json` và có checkbox bật/tắt F2.
- Không có GPU hoặc checkpoint SAM thì app dùng YOLO11s + DINOv3-S + F2. Ngưỡng F2 đã được **kiểm chứng lại cho YOLO** ở kernel v5: chọn lại trên cùng fold vẫn ra τd=0.75, τs=0.7 (xem mục F).

**Lưu ý:** mẫu đánh giá nhỏ (10 GT vi phạm), nên chênh lệch 0.01–0.03 F1 giữa các bộ lọc chưa có ý nghĩa thống kê.

### F. Kernel v5 (26/10/2026, T4): kiểm chứng F2 cho backend YOLO11s + ByteTrack + DINOv3-S
Dùng cùng fold tinh chỉnh (Bago_highway_10, Bago_rural_49, Mandalay_1_105) và cùng 4 clip đánh giá như mục E. Chọn lại trên YOLO vẫn ra **τd=0.75, τs=0.7**, giống hệt ngưỡng chọn trên SAM, nên `configs/track_filter.json` dùng một ngưỡng cho cả hai backend.

| YOLO11s + DINOv3-S, 4 clip đánh giá | P | Pu | R | Chưa xác định | F1u | FPS (T4) |
|---|---|---|---|---|---|---|
| Không lọc | 0.625 | 0.200 | 0.500 | 0.235 | 0.286 | 2.03 |
| **F2** (τd 0.75, τs 0.7) | 0.625 | **0.385** | 0.500 | 0.235 | **0.435** | 2.03 |
| F1 (τp 0.3), để tham khảo | 0.625 | 0.357 | 0.500 | 0.235 | 0.417 | 2.03 |

FPS của app trên CPU của box (YOLO11s + DINOv3-S + F2, stride 3): 4.6 FPS với Hanoi 6 s và 2.6 FPS với HELMET Bago_urban_3. Ảnh chụp ở `docs/screenshots/03…06_*_f2.png`.

## Bảng kết quả cuối cùng
**Phạm vi dữ liệu:**
- **T7** = 7 clip HELMET *test* (Bago_highway_10/20, Bago_rural_3/49, Bago_urban_3/50, Mandalay_1_105), stride 2, 350 frame, 51 GT track (19 vi phạm).
- **E4** = 4 clip đánh giá giữ lại của T7 (Bago_highway_20, Bago_rural_3, Bago_urban_3, Bago_urban_50; 10 GT track vi phạm). Ngưỡng bộ lọc chọn trên 3 clip còn lại.
- **CV** = GroupKFold 5 theo clip trên crop đầu từ clip HELMET *training* (kernel v3: 2809 crop, 39 clip).

**Chỉ số:** P = precision trên track khớp GT; Pu = precision tính cả track không khớp GT; R = recall trên mọi GT track vi phạm. FPS là end-to-end, gồm cả SAM, đo trên Kaggle T4.

| Phương pháp | Kernel | Tập | P | Pu | R | Chưa xác định | FPS |
|---|---|---|---|---|---|---|---|
| B0: YOLO helmet toàn khung, từng frame (gán GT bằng oracle) | v3 | T7 | 0.667 | – | 0.526 | 0.235 | 17.1 |
| B1: YOLO11s + ByteTrack + YOLO helmet | v2 | T7 | 0.455 | 0.312 | 0.263 | 0.429 | 2.27 |
| B1′: YOLO11s + ByteTrack + DINOv3-B | v2 | T7 | 0.786 | 0.268 | 0.579 | 0.200 | 1.98 |
| B1′: YOLO11s + ByteTrack + DINOv3-S | v5 | T7 | 0.733 | 0.289 | 0.579 | 0.200 | 2.02 |
| Core: SAM 3.1 fp16 4p + DINOv3-S | v4 | T7 | 0.800 | 0.185 | 0.632 | 0.184 | 0.57 |
| Core: SAM 3.1 fp16 4p + DINOv3-B | v3 | T7 | 1.000 | 0.160 | 0.421 | 0.237 | 0.56 |
| Core: SAM 3.1 fp16 4p + DINOv3-L | v4 | T7 | 0.917 | 0.234 | 0.579 | 0.158 | 0.47 |
| Core: SAM 3.1 fp16 2p + DINOv3-S | v3 | T7 | 0.722 | 0.188 | 0.684 | 0.158 | 0.66 |
| **App mặc định, GPU**: SAM 3.1 4p + DINOv3-S + F2 | v4 | E4 | 0.700 | 0.333 | 0.700 | 0.190 | 0.58 |
| (cùng cấu hình, không lọc) | v4 | E4 | 0.700 | 0.179 | 0.700 | 0.190 | 0.58 |
| **App mặc định, CPU**: YOLO11s + DINOv3-S + F2 | v5 | E4 | 0.625 | 0.385 | 0.500 | 0.235 | 2.03 |
| (cùng cấu hình, không lọc) | v5 | E4 | 0.625 | 0.200 | 0.500 | 0.235 | 2.03 |

| Probe crop đầu (CV, kernel v3) | P | R | F1 | AUC |
|---|---|---|---|---|
| DINOv3 ViT-S/16 + logistic | 0.648 | 0.806 | 0.718 | 0.903 |
| DINOv3 ViT-B/16 + logistic | 0.652 | 0.801 | 0.719 | 0.911 |
| DINOv3 ViT-L/16 + logistic | 0.709 | 0.850 | 0.773 | 0.937 |

**Lưu ý:**
- Mẫu E4 nhỏ (10 GT vi phạm), nên chênh lệch ±0.03 chưa có ý nghĩa thống kê.
- Kiểm tra tay 20 track không khớp: 15 FP thật (13 xe đỗ, 2 nền), 2 box trùng, 0 xe thiếu nhãn, 3 chưa xác định (`docs/results/unmatched_review.md`).

## Danh sách công nghệ (bản cuối)
| Thành phần | Model / phiên bản | Thư viện | Dataset | License | Lý do | Nguồn |
|---|---|---|---|---|---|---|
| Detector lõi | SAM 3.1 `sam3.1_multiplex.pt`, fp16 autocast trên T4 | `sam3` (repo Meta, commit main) + torch | – | SAM License (gated) | Phát hiện open-vocabulary bằng prompt văn bản motorcycle/person/head/helmet | https://github.com/facebookresearch/sam3 · https://huggingface.co/facebook/sam3.1 |
| Embedding đầu | DINOv3 ViT-S/16 (mặc định), B/16 và L/16 (so sánh), LVD-1689M | transformers ≥ 4.56 | – | DINOv3 License (gated) | Đặc trưng tự giám sát, probe tuyến tính rẻ; L tốt nhất trên CV nhưng chậm hơn | https://github.com/facebookresearch/dinov3 · https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m · https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m |
| Probe | Logistic regression (class_weight balanced), kNN k=7 | scikit-learn | crop đầu HELMET train | BSD-3 | Không cần fine-tune | https://scikit-learn.org |
| Tracking | ByteTrack | ultralytics 8.4.184 | – | AGPL-3.0 | MOT chuẩn, dùng cho cả hai backend | https://github.com/ifzhang/ByteTrack |
| Detector CPU / baseline | YOLO11s COCO | ultralytics 8.4.184 | COCO | AGPL-3.0 | Nhanh trên CPU, baseline B1/B1′ | https://github.com/ultralytics/ultralytics |
| Helmet baseline | YOLO11s helmet `nnsohamnn/helmet-detection-yolo11` | ultralytics | – | MIT | Baseline B0/B1 | https://huggingface.co/nnsohamnn/helmet-detection-yolo11 |
| Bộ lọc track F2 | person_score + disp_norm (tự viết), τd=0.75, τs=0.7 | numpy | chọn trên fold HELMET test | – | Loại xe đỗ không người, giữ rider dừng đèn đỏ | `src/de14/core.py`, `scripts/select_filters.py` |
| Dữ liệu đánh giá | HELMET (Myanmar) | Kaggle API 2.2.5 | mirror Kaggle `hozngvan/helmet-detection` | CC-BY 4.0 (OSF) | Video có track + nhãn từng rider | https://osf.io/4pwj8/ · doi:10.1109/ACCESS.2020.3021357 |
| Trọng số trên Kaggle | dataset riêng `holilihu/de7-weights` (SAM 3.1 chia part, DINOv3 S/B/L, sha256) và `holilihu/de14-sam31-cache-v3` (cache detection) | – | – | theo license model; chỉ dùng nội bộ, private | Không cần Kaggle Secret | (private) |
| Video demo | "Stratkruciĝo en Hanojo" | – | Wikimedia Commons | CC BY-SA 4.0 | Giao thông Việt Nam | https://commons.wikimedia.org/wiki/File:Stratkruci%C4%9Do_en_Hanojo.webm |
| Demo | Gradio app (`app.py`), chụp màn hình bằng Playwright | gradio, playwright | – | Apache-2.0 | Demo ảnh và video end-to-end | https://www.gradio.app |
| Hạ tầng | Kaggle T4 (GPU), box CPU (Python 3.13.5, torch 2.14.1+cpu) | – | – | – | Chỉ dùng tài nguyên miễn phí | https://www.kaggle.com |

### G. Chế độ REALTIME (`src/de14/realtime.py`, tab "Realtime" trong app)
**Pipeline:**
- YOLO26s (hoặc YOLO11s) chạy trên **mọi frame**, fp16 khi có GPU, rồi ghép rider–xe và ByteTrack.
- DINOv3-S + probe chạy **bất đồng bộ** trong worker thread. Worker chỉ nhận track mới hoặc track đến lượt sau mỗi N frame (mặc định 5), gom batch tối đa 32 crop, fp16.
- Bỏ phiếu theo track như chế độ offline.
- F2 được tính **online**: trung bình person_score và span tâm xe cập nhật tăng dần. Khác offline ở một điểm: đường chéo xe dùng trung bình chạy thay vì trung vị.
- SAM 3.1 vẫn là chế độ offline độ chính xác cao.
- Worker bắt lỗi classifier và `drain()` có hạn chót, nên lỗi không làm treo vòng lặp (`scripts/test_realtime_worker.py`: PASS).

**FPS end-to-end đo trên CPU của box** (đọc ảnh + YOLO + ByteTrack + DINOv3 async + chờ worker xong; 6 clip HELMET, mọi frame, imgsz 640, N=5; `docs/results/realtime_boxcpu/`):

| Detector | FPS CPU |
|---|---|
| YOLO26s | 14.5 |
| YOLO11s | 17.2 |

Frame HELMET trên box là bản 640×640 phóng lại, nên P/R trên box không so được với offline. P/Pu/R/F1u và FPS trên T4 **đang chờ kernel v7**.

**App, tab Realtime** (stream file, có lớp phủ FPS; CPU box, YOLO26s 640, F2 online):

| Video | FPS trung bình | Screenshot |
|---|---|---|
| Hanoi 6 s | 12.3 | `docs/screenshots/07_realtime_hanoi_file.png` |
| HELMET Bago_urban_3 (1920×1080) | 4.4 | `docs/screenshots/08_realtime_helmet_bago_urban_3_file.png` |

**App, tab Offline** (YOLO11s + DINOv3-S + F2, CPU, stride 3):

| Video | Frame | FPS | Track | Không mũ | Có mũ | Chưa xác định | Screenshot |
|---|---|---|---|---|---|---|---|
| Hanoi 6 s | 30 | 4.63 | 9 | 0 | 3 | 6 | `05_video_hanoi_dinov3_f2.png` |
| HELMET Bago_urban_3 | 27 | 2.57 | 4 | 0 | 2 | 2 | `06_video_helmet_bago_urban_3_dinov3_f2.png` |

Screenshot tab ảnh: `03_image_helmet_dinov3_f2.png`, `04_image_hanoi_dinov3_f2.png`.

**Công nghệ mới (thay đổi so với trước):**

| Thành phần | Lý do | Nguồn |
|---|---|---|
| **YOLO26s** (ultralytics 8.4.x, release assets v8.4.0, AGPL-3.0) cho detector realtime | Phiên bản YOLO mới nhất có sẵn trong ultralytics đã cài | https://docs.ultralytics.com/models/yolo26/ · https://github.com/ultralytics/assets/releases/tag/v8.4.0 |
| Gradio streaming (`gr.Image(streaming=True)`, generator) cho webcam/RTSP/file | Demo realtime trực tiếp trong app | https://www.gradio.app/guides/streaming-inputs |
| TensorRT/ONNX | Sẽ thử trong v7 (`pip install tensorrt-cu12` + `YOLO.export(format="engine", half=True)`). Chưa có kết quả | https://docs.ultralytics.com/integrations/tensorrt/ |
