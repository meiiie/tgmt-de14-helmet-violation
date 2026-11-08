# Rà soát nhãn người dùng — 06/11/2026

Đã nhận `path/to/1102_labels.json` và bản mới hơn `path/to/1102_labels (1).json` (xuất20:59:26giờViệtNam). Dùng bản mới hơn để rà, không gộp reviewed từ bản cũ vào bản mới. Source SHA256 đúng1102.mp4. Hai file gốc không bị chỉnh sửa.

## Trạng thái file mới nhất

- 106 mốc có 256 lượt box xe, 249 ID khác nhau; chỉ V001 lặp lại qua 8 frame.
- 49 mốc có `reviewed=true`, 57 mốc chưa duyệt. Trong 49 mốc có 93 lượt xe: 43 occupied / 3 empty / 47 unknown; 68 box đầu: 37 helmet / 31 unknown, **0 no_helmet**.
- 29 lượt xe có thể suy kết luận helmet-safe theo schema; 64 lượt còn unknown. Người dùng xác nhận đã gán nội dung nhưng có thể quên chọn trạng thái/bấm duyệt. Không tự chuyển toàn bộ thành reviewed vì có lỗi nội dung nhìn thấy được.

Đối chiếu ảnh xác nhận:

1. Source frame30/t1.0s được duyệt với danh sách xe trống dù xe phía trước vẫn nhìn thấy. Frame40/50 cũng đang làreviewed vớivehiclesrỗng, cần rà lại.
2. V001ởframe0–20 là xe có người mặc áo tím; frame60 lại dùngV001cho xe đỗ khác; frame90/100 dùngV001cho xe người áo sáng. Không thể coi đó là một danh tính xe.
3. Trongchuỗi100–140, các xe tiếp tục di chuyển nhưng đổi nhiềuIDmới. Ví dụ xe bên trái100/110/120/130 mangV002/V003/V008/V010. Cần đối chiếu ảnh để thống nhất đúng xe, không tự ghép ID bằng khoảng cáchbox.
4. Frame110/120có xe/phần người cạnh xe được gánempty. Cần phân biệt đang ngồi xe với đứng bên xe; không dùng“empty”để chỉxe đứngyên. Chưa tự gán lại nhãn mũ hoặc kết luận đây là vi phạm thật.

Ảnh đối chiếu và hướng sửa có tại **http://127.0.0.1:7862/review.html**. Snapshot nhận được và báo cáo kỹ thuật nằm ở`outputs/label_review_20261106/`.

## So sánh độ phủ, chưa phải benchmark GT hoàn chỉnh

Chỉ lấy49mốc được người dùng đánh dấu duyệt; cả3stride đều có chính xác các source-frame này. Ghépbox một-một theoIoU. Không dùng57framechưa duyệt làmnegative, không chạy lạiNN, không đổi model/config. Vì ảnh có missing annotations vàIDkhông nhất quán, không báo precision/FP/FN/IDSW/fragmentation.

| Stride | Head detector khớp /68 | Sauassociation /68 | SauByteTrack /68 | SauF2 /68 |
| --- | --- | --- | --- | --- |
|2|47|36|29|29|
|5|47|36|22|22|
|10|47|36|13|13|

NgưỡngIoU≥0.5; ởIoU≥0.3, các cột lần lượt50/39/29/29(S2),50/39/24/24(S5),50/39/14/14(S10). Đây chỉ là số box đầu đã cung cấp mà pipeline tìm lại được, không phải recall đại diện/được kiểm chứng hoàn toàn. Unknown helmet label vẫn cung cấp box localization, không cung cấp positive/negative helmet ground truth.

43lượt xe được gánoccupied, IoU≥0.3: trước tracking31khớp; sau tracking19/14/8ởstride2/5/10; sauF2vẫn19/14/8. Không suyF2hoàn toàn an toàn: nhiều trường hợp chưa được annotate đầy đủ hoặc gánempty/unknown, bao gồm case cần rà t4.0s.

Dữ liệu hiện tại hỗ trợ ưu tiên ràassociation/ByteTrack trước khi thay model hoặc train. Head detector cũng chưa khớp mọi box; chưa phân biệt hết FNmodel với khác biệtbox/blur/nhãn từ tập này.

Đã lần ngược thêm các ví dụ cụ thể:

- **Frame10/20:** đầu có thật và head detector đã tìm ra, nhưng wrapper detector không trả motorcycle nên head không có xe để ghép. Chạy chẩn đoán YOLO trên đúng2frame giải mã từ video gốc, conf0.01 (không đổi default): frame20 có motorcycle score **0.0721**, union xe+person khớp box người dùng IoU0.833; score này bị loại ở conf0.2, và ngay cả khi qua đó cũng chưa đạt conf_moto0.3 của pipeline. Frame10 vẫn không có motorcycle ở đầu ra wrapper tại conf0.01. Chưa kết luận nguyên nhân bên trong raw logits/NMS của frame10.
- **Frame100:** hai unionbox khớp GT IoU0.885/0.892, score0.530/0.310, đầu ghép được vào cả hai. Tracker không xuất box nào: snapshot cótentativeID6, lostID1, chưa cóconfirmedID. Candidate0.310 cũng thấp hơn new_track_thresh0.45. Đây là gate xác nhận track, không phải classifier đã nhìn đầu rồi phân loại sai.
- Chỉ gọi lại YOLO cho2frame để lấy low-confidence candidates; không chạy lại DEIM/DINO/CLIP hay toàn video, không đổi thresholds/model hiện hành. `yolo_low_conf_diagnostic.json` lưu chi tiết. Không suy chế độNMS từ thuộc tính không được backend ghi nhận.

## Hỗ trợ sửa nhãn

Trang annotation đã bổ sung:

- **Đổi ID, giữ nguyên các khung**: đổi ID xe đang chọn mà không xóa box/đầu. Không cho hai xe cùng ID trong một frame. Đổi thành công sẽ bỏ dấureviewed, cần duyệt lại.
- **Tới frame**: nhập source index0/30/60/100…, giữ nguyên dữ liệu khi di chuyển, từ chối chỉ số không có trong bộmốc.
- Cảnh báo sốxe vẫnunknown và nhắc“đứngyên≠khôngngười”. Không tự chuyển các trường hợp thực sự mơ hồ thànhoccupied.

Chuẩn bị nháp `1102_labels_context_suggestions_DRAFT.json`:66lượt xe đã có đầu do người dùng vẽ nhưngoccupancyunknown được **gợi ý**occupied. Giữ nguyên mọibox/ID/head label/rider count; frame có thay đổi đều vềunreviewed. Không dùng nháp này tínhaccuracy. Có31frame giữreviewed trong nháp; file gốc vẫn49. Nháp **chưa sửa ID/missing boxes**. Có thể tải từreviewpage rồi mở qua“Mở file nhãn”; không tự thay dữ liệu trong trình duyệt.

## Kiểm chứng

Ba kiểm thử logicUI mới tái hiện chức năng còn thiếu, rồi16/16caseJSqua sau triển khai: đổiIDgiữhead/box, duplicate ID bị từ chối, jump không sửa nhãn, cùng13regression cũ. Fresh read-only review không cònImportant issue trong phạm vi thay đổi. Browser thật chưa được kiểm thử tự động; dùngharness biênDOM/canvas như đợt trước, không tuyên bố đã QA visual/device.

Toàn bộ90kiểm thử Python cũng qua lại trong49.072s; log`outputs/label_review_20261106/unit_tests_final.log`. Kiểm tra HTTP xác nhận trang mới và trangreview được phục vụ, hai file nhãn gốc vẫn nguyên byte. Không sửa model, backend hay cấu hình inference chính.

File `coverage_report.json` lưu hash nhãn, scope, các box khớp detector nhưng mất sau tracking. FilegốcSHA256`bef774d435603c34d294b7fdfa72033f3ae468fe0d98d261656cf3b5c530f0f1`được giữ để kiểm tra không ghi đè.


## Bản nháp bổ sung xe theo chấp thuận của người dùng

Người dùng cho phép bổ sung các xe có thể nhận ra từ ảnh. Đã rà 106 ảnh mốc, 107 gợi ý từ `stride10/video_stages.jsonl` và các crop ảnh gốc; không chạy lại model. Bản `1102_labels_assisted_DRAFT.json` dùng đúng snapshot nhãn nhận được (SHA256 `bef774d435603c34d294b7fdfa72033f3ae468fe0d98d261656cf3b5c530f0f1`).

- Thêm **112 lượt box xe tại 63 frame**, gồm 95 gợi ý được đối chiếu bằng ảnh và 17 box vẽ từ quan sát ảnh gốc; thêm 22 box đầu. Đây là lượt xuất hiện trên các frame, không phải số xe vật lý khác nhau. Tổng bản nháp: 368 lượt box xe.
- Loại/để lại 12 gợi ý: box trùng, box gộp nhiều xe, quá nhỏ/mờ, hoặc người ngồi xổm bị đề xuất nhầm thành xe. C2/frame700 được thay bằng box xe hẹp hơn. Không chép toàn bộ detector output làm nhãn.
- Giữ nguyên 256 đối tượng cũ về ID, hình học, head boxes, helmet labels và rider count. Chỉ gợi ý 66 trường hợp occupancy unknown→occupied từ các đầu người dùng đã gán, ghi riêng trong provenance như bản nháp trước.
- 79 frame có chỉnh sửa được giữ `reviewed=false`; 21 frame đã duyệt không bị sửa vẫn giữ trạng thái. Không tự duyệt cả video.
- Xe đỗ chưa rõ người vẫn unknown, không dùng đứng yên để suy empty. Đầu mờ giữ unknown. Một số box xe/người gần mép ảnh là đối tượng chỉ nhìn thấy một phần; phải xem lại khi duyệt.
- ID `A_F...` là mã lần xuất hiện để người dùng rà, không phải physical track ID. A_PURPLE/A_DARK/A_BROWN là liên kết gợi ý trong các đoạn nhìn thấy liên tiếp, chưa gộp vào các ID V cũ. Chưa thể đo ID switches/fragmentation từ bản này.

Bản nháp ở `path/to/1102_labels_assisted_DRAFT.json` và `outputs/label_review_20261106/1102_labels_assisted_DRAFT.json`. Trang **http://127.0.0.1:7862/review_additions.html** có 126 ảnh đối chiếu trước/sau, frame navigation và tải JSON. Mở file nhãn mới trong trang annotation để duyệt; không tự thay localStorage hiện tại. Hai file xuất JSON gốc không thay đổi. Các sửa mới trong trình duyệt sau snapshot 20:59 chưa được đọc; cần lưu riêng trước khi import.

`build_assisted_additions.py` ghi quyết định theo từng candidate và tọa độ thủ công; `additions_report.json` ghi số lượng, source/draft hashes, frame thay đổi. Kiểm tra preservation so sánh từng đối tượng cũ, không chỉ tổng số. `publish_assisted_review.py` dựng ảnh/trang đối chiếu và bản tải về. Các tiện ích này nằm dưới outputs, không thay pipeline/model.

**Giới hạn:** đây là nháp hỗ trợ gán nhãn, không phải ground truth được kiểm tra độc lập. Nguồn gợi ý lấy từ chính detector đang đánh giá có thể gây thiên lệch; vẫn có thể thiếu xe rất nhỏ, bị che hoặc trong nhóm chồng nhau. Frame960/970 bị che nhiều chưa được coi là negative. Chưa báo precision/recall mới hoặc dùng nháp để training. Người dùng cần rà nội dung còn thiếu và danh tính xe trước khi dùng làm benchmark cuối.
