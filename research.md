# Đề 14 – Nghiên cứu SOTA: Phát hiện người đi xe máy không đội mũ bảo hiểm
Cập nhật: 23/10/2026 (UTC+7). [KC] = đã mở/fetch URL thật (trang web, GitHub API, HF API); [chưa kiểm chứng] = chưa mở được nguồn gốc. Bổ sung cho `notes/01_survey.md`.

## 0. Yêu cầu từ đề bài (`/workspace/tgmt/de_bai.pdf`)
- Đề 14: "Xây dựng hệ thống phát hiện người đi xe máy không đội mũ bảo hiểm." Thời gian: 25/05/2026 – 21/06/2026.
- Nộp: **báo cáo ≥ 20 trang** (không tính bìa, phụ lục, tài liệu tham khảo, code) về xuất xứ, chi tiết thuật toán, ứng dụng, cài đặt; **≥ 20 slide, trình bày 20 phút**.
- Môi trường gợi ý: **CUDA 13.2, PyTorch 2.11, Python 3.12, fine-tune bằng Unsloth**; "Ultralytics chưa hỗ trợ attention tốt"; xu hướng **SLM/VLM nhỏ (~1–2 GB)**.
- Công cụ: OpenCV/Matlab/Python/EMGU/dlib. "Chương trình phải chạy đúng với yêu cầu của đề bài."
→ Hệ quả: nên có 1 nhánh VLM nhỏ (có thể fine-tune bằng Unsloth) và 1 detector không chỉ dựa vào Ultralytics (vd. RF-DETR, dạng DETR có attention).

## 1. AI City Challenge Track 5 (helmet) – kết quả & lời giải
- Dữ liệu: video từ một thành phố Ấn Độ, 100 train + 100 test, 20s, 10fps, 1080p, 7 lớp (motorbike, D/P1/P2 × Helmet/NoHelmet); ~65k đối tượng; 2024 mở rộng thêm P0 (9 lớp). [KC] https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Naphade_The_7th_AI_City_Challenge_CVPRW_2023_paper.pdf ; https://www.aicitychallenge.org/2024-data-and-evaluation/
- **Không còn tải chính thức**: trang truy cập dữ liệu không liệt kê Track 5. [KC] https://www.aicitychallenge.org/ai-city-challenge-dataset-access/
- **2023** (mAP): 1. SKKU 0.4163; 2. CTC-AI 0.4107; 3. GRAPH@FIT 0.3875; … 6. HCMUS 0.3664; 10. Helios 0.3342. [KC] https://www.aicitychallenge.org/2023-challenge-winners/ . Bài tổng kết 2023: hạng 1 dùng DETA + Swin-L, SKKU dùng YOLOv8, VNPT dùng 2 detector (helmet + head), phần lớn dùng SORT. [KC] (Naphade 2023, link trên)
- **2024**: 1. UIT (Helios) 0.4860; 2. China Mobile 0.4824; 3. **VNPT AI 0.4792**; 7. BUPT 0.394. [KC] https://www.aicitychallenge.org/2024-challenge-winners/
- **Hạng 1 2024 – UIT/Helios (Việt Nam)**: Co-DETR (Co-DINO, Swin-L), 16 epoch, đánh giá multi-scale (640/1280/2048×1280), ensemble bằng WBF IoU 0.7; Minority Optimizer + Virtual Expander cho lớp hiếm (không VE 0.4830 → có VE 0.4860). [KC] https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Vo_Robust_Motorcycle_Helmet_Detection_in_Real-World_Scenarios_Using_Co-DETR_and_CVPRW_2024_paper.pdf ; repo https://github.com/aiclub-uit/AICITY2024_Track5 (12 sao, không khai license) [KC]
- **Hạng 3 2024 – VNPT AI (Việt Nam)**, Van Luong et al., CVPRW 2024 pp. 7027–7036 [KC] https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Van_Luong_Motorcyclist_Helmet_Violation_Detection_Framework_by_Leveraging_Robust_Ensemble_and_CVPRW_2024_paper.html
  - 4 module: (1) detection gồm phát hiện **đầu** và phát hiện người/xe, ensemble YOLOv7 + YOLOv8 + Co-DETR + EfficientDet bằng **WBF**; (2) object association ghép người–xe, người–đầu chung track ID; (3) post-processing tracking bằng **SORT + Kalman chỉnh sửa** để suy hướng xe → suy ra P0/P2; (4) **score correction** cộng offset cho lớp hiếm. Copy-paste augmentation (thủ công & tự động) cho P2/P0 NoHelmet. +7.43% mAP so với baseline Co-DETR.
  - Repo (đã đọc README) [KC] https://github.com/vnptai/AICITY2024_Track5 : 3 bước — huấn luyện baseline (YOLOv8, YOLOv7-D6 export ONNX 1280, Co-DETR train_v1/v2), huấn luyện head (Co-DETR head + EfficientDet), `full_pipeline.py`. Phần cứng: **8× RTX 2080Ti** (YOLO) và **4× A100-40GB** (Co-DETR) → **không tái lập được trên Colab/Kaggle miễn phí**. Checkpoint trên OneDrive (link trong README, chưa tải thử – [chưa kiểm chứng tính khả dụng]). Repo không khai license.
- Related: VisionGuard (arXiv 2506.21005) tự gán nhãn lại test AIC 2023+2024 theo 9 lớp. [KC qua trích đoạn] https://arxiv.org/html/2506.21005v1
- Bài học: (a) tách **head detection** khỏi người/xe; (b) tracking + hiệu chỉnh theo track; (c) xử lý mất cân bằng lớp. Ta giữ ý tưởng nhưng thay ensemble nặng bằng mô hình nhẹ + VLM xác minh.

## 2. Phát hiện open-vocabulary (zero-shot)
| Mô hình | Link [KC] | License | Ghi chú dùng cho đề |
|---|---|---|---|
| Grounding DINO tiny/base | https://huggingface.co/IDEA-Research/grounding-dino-tiny , repo https://github.com/IDEA-Research/GroundingDINO | Apache-2.0 | Prompt "motorcycle. person. helmet. head." – chậm trên CPU; dùng để **auto-label** |
| YOLO-World | https://github.com/AILab-CVC/YOLO-World | **GPL-3.0** | Nhanh, open-vocab; license GPL |
| YOLOE (THU-MIG) | https://github.com/THU-MIG/yoloe | [chưa kiểm chứng license] | Open-vocab real-time |
| OWLv2 | https://huggingface.co/google/owlv2-base-patch16-ensemble | Apache-2.0 | Zero-shot, chậm |
| OmDet-Turbo | https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf | Apache-2.0 | Open-vocab real-time hơn |
| RF-DETR (Roboflow) | https://github.com/roboflow/rf-detr | Apache-2.0, cập nhật 22/10/2026 | Detector DETR (attention) đóng, fine-tune được – đáp ứng ghi chú "Ultralytics chưa hỗ trợ attention tốt" |
| **SAM 3** (Meta, 11/2025) | https://github.com/facebookresearch/sam3 , https://huggingface.co/facebook/sam3 , paper https://arxiv.org/html/2511.16719 | "SAM License" (other), **gated – cần HF token + chấp thuận** | Detect+segment+**track** theo khái niệm văn bản ("helmet", "motorcyclist"); ~0.9B tham số; SA-Co 270K khái niệm, đạt 75–80% hiệu năng người. Hợp làm **auto-labeler/teacher** trên GPU, không hợp chạy realtime CPU |
| YOLO26 (Ultralytics, được nhắc trong đề 2) | [chưa kiểm chứng] | AGPL-3.0 (Ultralytics) [chưa kiểm chứng] | Có thể làm detector baseline |

Lưu ý: không prompt "no helmet" được tin cậy với detector open-vocab (phủ định yếu) → dùng prompt "helmet"/"head" rồi suy luận, hoặc để VLM trả lời.

## 3. VLM nhỏ làm "verifier" cho track không chắc chắn
| Mô hình | Link [KC] | License | Gated |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | Apache-2.0 | Không |
| Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | Không |
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | không khai trong card (Qwen research license) [chưa kiểm chứng] | Không |
| Florence-2 base/large | https://huggingface.co/microsoft/Florence-2-base | MIT | Không |
| SmolVLM2-2.2B | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | Apache-2.0 | Không |
| PaliGemma 2 3B | https://huggingface.co/google/paligemma2-3b-pt-224 | Gemma | **Có – cần HF token** |
Đề xuất: **Qwen3-VL-2B** (Apache-2.0, không gated, ~2B phù hợp "1–2 GB" khi lượng tử hoá 4-bit, Unsloth hỗ trợ fine-tune [chưa kiểm chứng trang Unsloth]). Chỉ gọi VLM cho track có tỉ lệ phiếu trong vùng mơ hồ (0.4–0.6) hoặc thiếu phiếu → giữ FPS. Prompt dạng câu hỏi đóng: "Người lái xe máy trong ảnh có đội mũ bảo hiểm không? Trả lời: có / không / không rõ." Tham khảo thêm: dataset VLM-instruct helmet `revitotan/helmet-threerider-vlm-instruct-dataset` (MIT, HF API) [KC] có thể dùng thử fine-tune.

## 4. Dataset công khai có license rõ ràng (ưu tiên VN/châu Á)
| Dataset | Link | Quy mô | License | Ghi chú |
|---|---|---|---|---|
| **HELMET (Myanmar)** – Lin, Siebert et al., IEEE Access 2020 | https://osf.io/4pwj8/overview [KC]; mirror Kaggle https://www.kaggle.com/datasets/hozngvan/helmet-detection [KC] | 910 clip 10s, 10fps, 1080p; 91.000 frame; 10.006 xe có track; nhãn helmet theo vị trí người ngồi | OSF: **CC-BY 4.0**; Kaggle mirror ghi "Unknown"; yêu cầu truy cập OSF bị tắt từ 9/2021 (theo trích đoạn) | **Gần VN nhất (Đông Nam Á), dạng video + track** → rất phù hợp đánh giá theo track. Bài báo: weighted F-measure 67.3%, >8 FPS. https://doi.org/10.1109/access.2020.3021357 |
| Roboflow "helmet-detection-20241116" (Quoc Anh Dev – có vẻ VN) | https://universe.roboflow.com/quoc-anh-dev/helmet-detection-20241116-autvs [KC qua trích đoạn] | 4.7k ảnh; lớp bicyclist/driver/helmet/no-helmet | **CC BY 4.0** | Tải cần tài khoản Roboflow (free) |
| Roboflow "motorcycle helmet object detection" (projectt) | https://universe.roboflow.com/projectt/motorcycle-helmet-object-detection [KC qua trích đoạn] | 3.6k ảnh; With/Without Helmet | CC BY 4.0 | |
| Kaggle HelmetViolations | https://www.kaggle.com/datasets/pkdarabi/helmet [KC qua trích đoạn] | 1.004 ảnh (có augment), WithHelmet/WithoutHelmet/Plate | CC BY 4.0 | Cần Kaggle API |
| Roboflow "Motorcycle Helmet" (tung-pham-thanh) | https://universe.roboflow.com/tung-pham-thanh/motorcycle-helmet-7d2gr-pabje | 11k ảnh | [chưa kiểm chứng license] | |
| no-helmet-no-ride (bài ĐH Hàng hải 2025, 4.267 ảnh, YOLO11n mAP50 89,8%) | bài báo https://jmst.vimaru.edu.vn/index.php/jmst/article/download/638/578 | | [chưa kiểm chứng link/license] | |
| AIC Track 5 | xem mục 1 | | không tải được | |
| UniqueData/helmet_detection (HF) | HF API | | CC-BY-NC-ND-4.0 – **không cho phép phái sinh** | Loại |

## 5. Trọng số helmet pretrained (HF API [KC])
- iam-tsr/yolov8n-helmet-detection – MIT – P 0.855 / R 0.808 / mAP50 0.881 (theo card) https://huggingface.co/iam-tsr/yolov8n-helmet-detection
- nnsohamnn/helmet-detection-yolo11 – MIT – YOLO11m/s, mAP50 0.899 (val 126 ảnh) https://huggingface.co/nnsohamnn/helmet-detection-yolo11
- vivekvar/helmet-v5 – Apache-2.0 – YOLO11l + pose + EfficientNet-B0 head classifier F1 0.864, voting MIN_VOTES=2; **HF đánh dấu 1 file unsafe → không load pickle** https://huggingface.co/vivekvar/helmet-v5
Lưu ý: số liệu trong model card là tự báo cáo, chưa kiểm định độc lập; cần đo lại trên test VN.

## 6. Thiết kế chọn
**Baseline (B0/B1)** – nhẹ, chạy CPU:
YOLO COCO (person+motorcycle) → ByteTrack → crop 1/3 trên bbox người → YOLO11s helmet (nnsohamnn, MIT) → voting theo track: ≥3 phiếu, tỉ lệ no_helmet ≥0.6 = "không đội mũ", ≤0.4 = "có đội mũ", còn lại = **"chưa xác định"**. (B0 = mô hình helmet theo từng frame, không tracking.)

**SOTA/cải tiến (B2)** – lấy cảm hứng từ VNPT/UIT nhưng vừa tài nguyên miễn phí:
1. Detector: RF-DETR (Apache-2.0, attention) fine-tune trên Colab/Kaggle T4 với HELMET (Myanmar) + Roboflow CC-BY; lớp: motorcycle, rider, head.
2. Head detection riêng (ý VNPT) hoặc pose keypoints → crop đầu chính xác.
3. ByteTrack + gán rider–xe (driver/passenger theo vị trí trên xe).
4. Classifier đầu nhỏ (EfficientNet-B0/ConvNeXt-T) + temperature scaling; copy-paste augmentation cho lớp no-helmet / passenger hiếm (ý VNPT/UIT).
5. **VLM verifier Qwen3-VL-2B** (có thể fine-tune LoRA bằng Unsloth theo đề) chỉ cho track mơ hồ; nếu VLM trả "không rõ" → giữ "chưa xác định".
6. (Tuỳ chọn offline) SAM 3 / Grounding DINO làm auto-labeler tạo pseudo-label cho video VN, người kiểm tra lại.

**Đánh giá**: theo track trên (a) HELMET test split (Myanmar, có nhãn track) và (b) bộ clip VN tự gán nhãn. Precision/Recall/F1 lớp không-mũ, undetermined rate, coverage-vs-precision, FPS CPU/T4, ablation B0/B1/B2 ± VLM.

## 7. Token / API cần
- **HF token**: bắt buộc nếu dùng SAM 3 hoặc PaliGemma 2 (gated, cần bấm chấp thuận). Không cần cho Qwen3-VL, Florence-2, Grounding DINO, helmet YOLO.
- **Kaggle API** (kaggle.json): cần để tải mirror HELMET / HelmetViolations; hoặc tải thủ công.
- **Roboflow API key (free)**: cần để tải dataset Roboflow Universe.
- GPU: Colab/Kaggle T4 miễn phí cho fine-tune và VLM; box chỉ CPU 8 nhân.
