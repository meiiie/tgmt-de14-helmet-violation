# Gán nhãn video cho người mới

Trang local: **http://127.0.0.1:7862/**. Đây là công cụ riêng, không đổi app nhận diện ở7861. Dữ liệu chuẩn bị tại `outputs/annotations/1102/`:106 ảnh sạch, kích thước1920×1080, lấy từ frame nguồn0,10,…,1050 của1102.mp4. Không dùng box/dự đoán AI để điền sẵn GT.

## Bắt đầu với3 frame

1. **Vẽ xe**: kéo khung lớn ôm trọn xe máy và những người trên xe. Một xe là một mã, ví dụ `V001`.
2. Chọn trạng thái xe: **có người ngồi/đi xe**, **xe không người**, hoặc **chưa xác định**. Nhập tổng số người, kể cả người có đầu bị che nhưng bạn chắc chắn có mặt.
3. **Vẽ đầu**: mỗi đầu nhìn thấy là một khung nhỏ riêng. Chọn **Có mũ bảo hiểm / Không có mũ bảo hiểm / Chưa rõ**. Mũ vải, mũ lông, mũ lưỡi trai không phải mũ bảo hiểm. Nếu chi tiết quá nhỏ/mờ, chọn chưa rõ; không đoán.
4. Nếu xe có4 người, vẽ4 đầu khi nhìn thấy đủ. Nếu chỉ thấy1 đầu nhưng chắc có2 người, nhập tổng người2 và chỉ vẽ1 đầu thật; không vẽ đầu giả cho người bị che hoàn toàn.
5. Bấm **Đã kiểm tra đầy đủ frame này** sau khi đã đánh dấu mọi xe cần đánh giá. Frame đường trống cũng có thể duyệt; khi ấy nó là ví dụ không có xe, không phải dữ liệu bị quên.
6. Sang frame sau. Nếu vẫn là xe cũ, chọn `V001` trong **ID khi vẽ xe** rồi vẽ lại vị trí. Chỉ tạo ID mới cho xe khác. ID của mô hình không được dùng làm sự thật về danh tính.
7. Làm thử3 frame, bấm **Lưu file JSON**. File tải xuống thường tên `1102_labels.json` trong Downloads. Đưa lại file hoặc đường dẫn; tôi có thể kiểm tra nhãn cùng bạn trước khi làm phần còn lại.

Không cần hiểu JSON hay tự nhập tọa độ. Trang chuyển vị trí chuột thành pixel nguồn tự động, kể cả khi phóng to2×/3×. **Chọn/sửa** cho phép chọn và kéo khung; nếu sai kích thước, xóa rồi vẽ lại. Escape/hủy kéo trả lại tọa độ cũ. Sửa nhãn đã duyệt sẽ đưa frame về chưa duyệt.

Trang tự lưu trong trình duyệt, nhưng **JSON là bản sao chính cần tải định kỳ**. Có thể dùng **Mở file nhãn** để tiếp tục ở máy/trình duyệt khác. CSV là bảng xem nhanh; CSV chưa chứa đầy đủ thông tin các frame trống/nguồn như JSON, nên dùng JSON để đánh giá. Không gửi dữ liệu ra dịch vụ cloud.

## Vì sao phải giữ ID?

| Mốc | Xe nhìn thấy | ID đúng |
| --- | --- | --- |
| Frame0 | Một xe đang đi phía trước | V001 |
| Frame10 | Vẫn xe đó, vị trí thay đổi | V001 |
| Frame20 | Xe khác đi vào ảnh | V002 cho xe mới; xe cũ vẫnV001 |

Giữ ID giúp phân biệt mô hình mất track, tách một xe thành nhiều track hoặc đổi nhầm ID. Chỉ khoanh ảnh độc lập mà thay ID mỗi frame thì chưa đủ để đánh giá tracking.

Nếu đã gán nhầm ID, **không cần vẽ lại**: chọn xe hoặc một đầu của xe đó, sửa ô **ID của xe đang chọn**, bấm **Đổi ID, giữ nguyên các khung**. Tất cả box/nhãn đầu được giữ; frame sẽ về chưa duyệt để bạn xác nhận lại. Không cho hai xe khác nhau cùng ID trong một frame.

Ô **Tới frame** nhận chỉ số frame nguồn, ví dụ30là mốc1,0s của video30FPS (mốc thứ4 trong bộ ảnh lấy mẫu10frame/lần). Nó không phải số thứ tự1–106của các ảnh.

Sau khi người dùng xuất nhãn lần đầu, có [báo cáo rà soát và độ phủ trên box đã gán](label_review_2026-11-06.md) cùng trang **http://127.0.0.1:7862/review.html**. File gốc được giữ nguyên; bản nháp gợi ý trạng thái không tự thành GT đã duyệt.

## Phạm vi nhãn và kiểm tra

106 mốc cách nhau1/3giây là **GT tại các mốc đã duyệt**, không tự thành GT của toàn bộ1052 frame. Bản nháp có0 frame duyệt không phải negative GT. Khi chỉ duyệt3 frame, chỉ đánh giá3 frame đó; không suy accuracy toàn35giây. Không nội suy nhãn tự động. Nếu chỉnh mô hình bằng clip này, cần clip test khác để kiểm chứng độc lập.

```powershell
.venv/Scripts/python.exe scripts/validate_video_labels.py "path/to/1102_labels.json" --video "path/to/1102.mp4"
```

Validator kiểm tra source SHA256, frame/timestamp, tọa độ, ID trùng, số người/đầu và trạng thái chưa rõ. Chỉ số chưa rõ không bị coi thành người có mũ.

Tạo lại bộ gán nhãn cho video khác (không chạy model):

```powershell
.venv/Scripts/python.exe scripts/prepare_video_annotation.py "path/to/video.mp4" --out outputs/annotations/my_video --stride 10 --serve --port 7862
```

Nếu cổng7862 đang phục vụ bộ1102, dùng cổng khác hoặc kết thúc đúng server cũ trước khi mở bộ mới. Không cần dừng app7861.

## Video đã có nhãn chuẩn

| Nguồn | Nhãn phù hợp | Giới hạn cần phân biệt |
| --- | --- | --- |
| [HELMET / OSF](https://osf.io/4pwj8/) — ưu tiên | 910clip, mỗi clip100frame/10giây/10FPS/1920×1080 theo[Readme gốc](https://osf.io/download/pvecy/). CSV cótrack_id/frame_id/box/nhãn mũ theo người. OSF API xác nhận CC-BY4.0 ngày06/11/2026. | Phân phối bằng chuỗi ảnh video; không phải GT box riêng cho từng đầu. Kho ảnh gốc chia7ZIP khoảng1.09–4.76GB/phần, chưa tải toàn bộ. Mirror ảnh640×640 không phục hồi chi tiết1080p dù resize lại. Bộ16 frame hiện có chỉ là smoke. |
| [AI City2023 Track5](https://www.aicitychallenge.org/2023-data-and-evaluation/) | Video20giây,10FPS,1080p; schemaGT cótrack_id, box và lớp người đội/không mũ. | Tối đa3 người theo taxonomy được công bố; dữ liệu có quy trình yêu cầu/điều khoản riêng, không phải bản CC-BY củaHELMET. |
| [AI City2024 Track5](https://www.aicitychallenge.org/2024-data-and-evaluation/) | Video20giây,10FPS,1080p; có lớpP0 cho trẻ ngồi trước, tới4 người. | Schema công bố **không cótrack_id**, nên không tự đủ đoID switch. Box rider không đồng nghĩa box đầu. Có yêu cầu truy cập và[license riêng](https://www.aicitychallenge.org/wp-content/uploads/2024/01/DATASET_LICENSE_AGREEMENT_2024_Track5.pdf). |

NguồnHELMET: [repository tác giả](https://github.com/LinHanhe/Helmet_use_detection), [paper gốc](https://doi.org/10.1109/ACCESS.2020.3021357), [OSF metadata](https://api.osf.io/v2/nodes/4pwj8/), [license được node trỏ tới](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). Có thể đọc metadata/license bằng API dù trangOSF trực tiếp trả403 cho công cụ web trong lần kiểm tra này.

Không gửi yêu cầu truy cập AI City, đăng ký tài khoản hoặc tải toàn bộ kho lớn tự động trong đợt này. Dữ liệu training/probe hiện tại được giữ nguyên. Video1102 vẫn chưa có GT trước khi bạn gán/duyệt nhãn.

## Kiểm chứng công cụ

7 kiểm thử Python về schema/uncertainty đã qua; hiện có16 kiểm thử logic JavaScript của chính trang đã qua (mapping pixel, nhiều đầu, import/export, hủy kéo, review/autosave, đổiIDvàjump). Đây là kiểm thử với lớp DOM/canvas mô phỏng ở biên, **không phải QA browser thật**. Môi trường chặn mở Edge headless và browser client không được cấp native bridge, nên phần trình bày và thao tác thiết bị thật cần bạn thử trên trang local. Log mới ở`outputs/label_review_20261106/`; log đợt đầu ở`outputs/video_fix_20261106/annotation_*.log`.
