# Kiểm tra tay track Core v2 không khớp GT (SAM 3.1 + DINOv3-B, HELMET test)

**Cách làm.** Tái lập đúng tracking của kernel v2 trên box (`scripts/replay_unmatched.py`: cùng cache SAM 3.1 v2 `results/sam31/*.json`, ByteTrack, stride 2, conf 0.4). Số track và `n_obs` của từng track khớp 100% với `outputs/kaggle_v2/eval_core/*_tracks.csv`. Track "không khớp" = IoU < 0.3 với mọi box GT (luật đánh giá v2).
- Có 6/7 clip trên box; Mandalay_1_105 không có frame cục bộ.
- 39 track bị gán "không đội mũ" mà không khớp GT (v2 báo 41 cho cả 7 clip).
- Chọn ngẫu nhiên 20 track (seed 0). Với mỗi track lấy frame giữa: box dự đoán màu đỏ, box GT màu xanh, kèm ảnh phóng to.
- Ảnh nằm ở `unmatched_review/s0..s4.jpg`, danh sách track ở `sel.json`. Tôi đã xem từng ảnh.

| # | clip | track | Nhận định | Loại |
|---|---|---|---|---|
| 1 | Bago_highway_10 | 3 | xe đỗ ven đường, không người lái | FP thật (xe đỗ) |
| 2 | Bago_highway_10 | 5 | xe đỗ ven đường, không người lái | FP thật (xe đỗ) |
| 3 | Bago_highway_10 | 6 | xe đỗ / nền phía sau | FP thật (xe đỗ) |
| 4 | Bago_highway_10 | 12 | box mảnh chồng lên 2 xe có nhãn (IoU < 0.3) | box trùng/mảnh của xe có nhãn |
| 5 | Bago_highway_20 | 1 | xe nhỏ đứng yên ở nền | FP thật (xe đỗ) |
| 6 | Bago_highway_20 | 3 | xe đỗ chở bao, không người lái | FP thật (xe đỗ) |
| 7 | Bago_highway_20 | 6 | vùng nền lộn xộn, không rõ là xe | FP thật (nền) |
| 8 | Bago_highway_20 | 7 | xe đỗ nhỏ ở nền | FP thật (xe đỗ) |
| 9 | Bago_rural_3 | 1 | xe rất nhỏ ở xa, cạnh xe có nhãn, không rõ có người lái | chưa xác định |
| 10 | Bago_rural_3 | 5 | xe đỗ dưới mái hiên | FP thật (xe đỗ) |
| 11 | Bago_rural_3 | 7 | vật nhỏ giống bánh xe, không phải xe máy | FP thật (nền) |
| 12 | Bago_rural_49 | 5 | box trùng lên rider đã có nhãn | box trùng/mảnh của xe có nhãn |
| 13 | Bago_rural_49 | 57 | xe đỗ ở nền | FP thật (xe đỗ) |
| 14 | Bago_urban_3 | 1 | xe bị cắt ở mép phải khung hình, không rõ có người lái | chưa xác định |
| 15 | Bago_urban_3 | 4 | xe đỗ ven đường | FP thật (xe đỗ) |
| 16 | Bago_urban_3 | 7 | xe đỗ, có người đứng cạnh (không ngồi lái) | FP thật (xe đỗ) |
| 17 | Bago_urban_50 | 2 | xe bị cắt ở mép phải, không rõ có người lái | chưa xác định |
| 18 | Bago_urban_50 | 4 | dãy xe đỗ ở tiền cảnh trái | FP thật (xe đỗ) |
| 19 | Bago_urban_50 | 9 | xe đỗ, có người đứng cạnh | FP thật (xe đỗ) |
| 20 | Bago_urban_50 | 37 | xe đỗ, có người đứng cạnh | FP thật (xe đỗ) |

## Tổng hợp (20 track)
| Loại | Số |
|---|---|
| FP thật: xe đỗ / không người lái | 13 |
| FP thật: nền, không phải xe máy | 2 |
| Box trùng/mảnh của xe đã có nhãn (lỗi định vị/trùng track, cũng là cảnh báo sai) | 2 |
| Xe đang chạy có người lái nhưng không có nhãn GT | **0** (chưa thấy trường hợp chắc chắn) |
| chưa xác định (xe rất nhỏ hoặc bị cắt ở mép) | 3 |

**Kết luận.**
- Ít nhất 17/20 (85%) là cảnh báo sai thật, không phải do thiếu nhãn. HELMET chỉ gán nhãn xe có người lái, nên `precision_incl_unmatched` (0.21 ở v2) gần với precision thật hơn là con số 1.0 tính trên track khớp.
- Nguyên nhân chính là pipeline suy vùng rider từ box xe khi SAM không thấy người, nên xe đỗ vẫn bị phân loại.
- Hướng sửa (chưa chạy, cần quyết định): bắt buộc có box `person` chồng lên xe, hoặc lọc track gần như đứng yên (độ dịch chuyển tâm < ngưỡng).
