# Đề 14 – Khảo sát: Phát hiện người đi xe máy không đội mũ bảo hiểm (video giao thông VN)
Ngày: 23/10/2026 (UTC+7). Quy ước: [ĐÃ KIỂM CHỨNG] = đã mở URL thật; [chưa kiểm chứng] = chỉ thấy qua tìm kiếm/chưa tải thử.

## 1. Bộ dữ liệu
| # | Nguồn | Nội dung | Giấy phép / tải được? | Trạng thái |
|---|---|---|---|---|
| D1 | AI City Challenge 2023/2024 Track 5 – https://www.aicitychallenge.org/2024-data-and-evaluation/ ; bài báo https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Naphade_The_7th_AI_City_Challenge_CVPRW_2023_paper.pdf | 100 video train, 20s, 10fps, 1080p, thành phố Ấn Độ; 7 lớp (motorbike, DHelmet, DNoHelmet, P1Helmet, P1NoHelmet, P2Helmet, P2NoHelmet); ~65k đối tượng | Trang https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ ghi: chỉ các track liệt kê còn tải được — **Track 5 KHÔNG có trong danh sách → không còn tải chính thức** | [ĐÃ KIỂM CHỨNG] |
| D2 | Roboflow "no-helmet-no-ride" (4.267 ảnh, có ảnh camera VN/châu Á), dùng trong bài Nguyễn Hữu Tuân & Vũ Thanh Hương, Tạp chí KHCN Hàng hải số 83 (08/2025): https://jmst.vimaru.edu.vn/index.php/jmst/article/download/638/578 | helmet/no-helmet, YOLO11n mAP@0.5 = 89,8% | Bài báo [ĐÃ KIỂM CHỨNG qua trích đoạn]; link Roboflow & license dataset [chưa kiểm chứng] | Cần xác minh |
| D3 | Dashcam Hanoi – https://platform.ultralytics.com/lamvh/datasets/dashcam-hanoi | 300 ảnh dashcam Hà Nội, 1 lớp, KHÔNG có nhãn mũ | License [chưa kiểm chứng] | Chỉ dùng làm ảnh test định tính |
| D4 | vivekvar/cctv-datasets (HF) | CCTV Ấn Độ, tham chiếu trong model card helmet-v5 | [chưa kiểm chứng] | — |
| D5 | Video test VN tự quay / YouTube giao thông VN | Bộ test miền mục tiêu, cần tự gán nhãn (CVAT) | Bản quyền YouTube cần cân nhắc | **Cần Darren quyết định** |

Kết luận: Không có dataset helmet công khai của VN có license rõ ràng đã kiểm chứng. Đề xuất: dùng trọng số pretrained (Mục 2) + tự gán nhãn ~10–20 clip ngắn giao thông VN (≈1–2k track-frame) làm **tập test**.

## 2. Mô hình / trọng số pretrained (đã gọi HF API, kiểm tra file + license)
| # | Repo | Mô tả | License | Ghi chú |
|---|---|---|---|---|
| M1 | YOLO11n/s COCO (Ultralytics) – https://github.com/ultralytics/ultralytics | Lớp `person`, `motorcycle` | AGPL-3.0 [chưa kiểm chứng trang license, kiến thức chung] | Detector giai đoạn 1 |
| M2 | https://huggingface.co/iam-tsr/yolov8n-helmet-detection | YOLOv8n, `helmet`/`no_helmet`, P=0.855 R=0.808 mAP50=0.881 (theo card), file `best.pt` | MIT | [ĐÃ KIỂM CHỨNG] — **baseline** |
| M3 | https://huggingface.co/nnsohamnn/helmet-detection-yolo11 | YOLO11m/s, `With Helmet`/`Without Helmet` cho người đi xe máy, mAP50=0.899 (val 126 ảnh) | MIT | [ĐÃ KIỂM CHỨNG] — ứng viên tốt cho crop |
| M4 | https://huggingface.co/vivekvar/helmet-v5 | Pipeline YOLO11l + pose → crop đầu → EfficientNet-B0 (F1=0.864), voting MIN_VOTES=2 | Apache-2.0 | [ĐÃ KIỂM CHỨNG]; HF cảnh báo "1 file scanned as unsafe" → **không load pickle trực tiếp**, chỉ tham khảo kiến trúc |
| M5 | https://huggingface.co/jashwanthpeddisetty0712/violation-detection-yolov8 | 5 lớp gồm WITH/WITHOUT_HELMET | [chưa kiểm chứng] | Dự phòng |
| R1 | https://github.com/vnptai/AICITY2024_Track5 | Lời giải top-3 Track 5 (VNPT AI), detector helmet + head | Repo tồn tại [ĐÃ KIỂM CHỨNG]; nội dung chi tiết chưa đọc | Tham khảo |
| R2 | ByteTrack – https://github.com/ifzhang/ByteTrack (có sẵn trong Ultralytics `tracker=bytetrack.yaml`) | MOT | URL tồn tại [ĐÃ KIỂM CHỨNG] | Tracking |
| R3 | VisionGuard – https://arxiv.org/html/2506.21005v1 | Framework helmet violation trên AIC 2023+2024, tự gán lại nhãn test | [chưa đọc toàn văn] | Related work |

## 3. Pipeline đề xuất
1. Đọc video theo lô (stride khung, vd. 2–3 → ~10 fps hiệu dụng).
2. YOLO11 COCO: detect `motorcycle` + `person`; ghép rider = person có IoU/độ chồng lấn với xe ≥ ngưỡng và tâm dưới nằm trong bbox xe mở rộng.
3. ByteTrack theo dõi rider (track_id).
4. Crop vùng đầu: 1/3 trên của bbox person (improved: keypoint pose YOLO11-pose).
5. Phân loại helmet/no_helmet trên crop (M2 hoặc M3 chạy trên crop đầu/thân trên).
6. Voting theo track: tích lũy xác suất; quyết định khi ≥ N_min khung hợp lệ (crop ≥ 24 px, conf ≥ τ):
   - tỉ lệ no_helmet ≥ 0.6 → **không đội mũ**; ≤ 0.4 → **có đội mũ**; còn lại hoặc thiếu bằng chứng → **chưa xác định**.
7. Xuất CSV theo track + video annotate.

## 4. Baseline vs cải tiến
- **Baseline (B0):** chạy trực tiếp M2 trên từng khung toàn ảnh, quyết định theo khung, không tracking, không "chưa xác định".
- **B1:** M1 rider + crop đầu heuristic + M3 + ByteTrack + voting có ngưỡng "chưa xác định".
- **Cải tiến (B2):** pose-based head crop, SAHI/tiling cho đối tượng nhỏ, fine-tune nhẹ trên Colab/Kaggle T4 (≤1–2 giờ) bằng D2 + vài trăm crop VN tự gán, hiệu chỉnh nhiệt độ (temperature scaling) cho ngưỡng.

## 5. Kế hoạch đánh giá
- Đơn vị đánh giá: **track rider** (chính) và khung (phụ). GT: nhãn track {helmet, no_helmet, ambiguous}.
- Precision/Recall/F1 lớp `no_helmet` trên các track đã quyết định; **Undetermined rate** = #track "chưa xác định"/#track; báo cáo đường cong P–R theo ngưỡng & coverage.
- FPS end-to-end (CPU box 8 nhân, và GPU T4 Colab), tách thời gian detect/classify/track.
- Ablation: B0 vs B1 vs B2; với/không voting; ngưỡng N_min.
- Phân tích lỗi: ban đêm, mưa, nón lưỡi trai/mũ vải, người ngồi sau, che khuất.

## 6. Cần Darren quyết định
1. Nguồn video test VN (tự quay / YouTube / camera công khai) và quy mô gán nhãn.
2. Có chấp nhận dataset không phải VN (Ấn Độ, Roboflow) để fine-tune không; license Roboflow cần xác minh.
3. Có tính người ngồi sau (passenger) hay chỉ người lái.
4. License AGPL của Ultralytics có ổn cho bài nộp không.
