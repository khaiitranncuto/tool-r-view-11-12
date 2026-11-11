# Kiểm tra tác vụ ec396b79 — 09/11/2026

Nguồn: `.jobs/ec396b79-4d11-4f95-a0a4-d6c3a2fbec43.json`, log đầy đủ
`logs/task_ec396b79-4d11-4f95-a0a4-d6c3a2fbec43.log`, code đóng gói hình,
code kiểm chứng và các khung hình gốc tại những mốc bị bác bỏ.
Không gọi Gemini/Groq mới và không sửa code trong lần chẩn đoán này.

## Vì sao lâu

- Tác vụ từ 14:21:59 đến 15:40:57, khoảng 79 phút, kết thúc thất bại vì read timeout phần 3.
- Chuẩn bị ban đầu khoảng 6 phút; cắt clip chi tiết khoảng 8 phút.
  API 1 báo quota ngày lúc 14:37:53; khi chuyển API, cùng 13 clip bị tạo lại
  từ 14:38:04 đến 14:46:31, mất thêm khoảng 8 phút.
- Xem 13 clip từ 14:46:31 đến 15:01:38, khoảng 15 phút.
- Phần 1: viết/ghép hình 15:01:51–15:07:27; kiểm chứng đến 15:13:03;
  sửa đến 15:15:59. Phần 2: viết/ghép hình 15:16:01–15:20:41;
  kiểm chứng đến 15:28:44; sửa đến 15:31:48.
- 98 yêu cầu có log USAGE thành công: 1 overview, 13 observation,
  23 viết lời, 32 ghép hình/sửa, 29 kiểm chứng. Đây không phải 98 retry.
- Kiểm chứng có 30 request bắt đầu, 29 thành công, 1 lỗi 503 quá tải.
  Mọi phản hồi thành công đều có verdict hợp lệ cho ID được yêu cầu.
  Sau lượt giới hạn, không còn ID thiếu phản hồi ở phần 1–2.
- Hai lần sửa sau kiểm chứng còn thất bại vì quota câu: yêu cầu 5 nhận 4,
  yêu cầu 4 nhận 3. Hệ thống phải giữ bản trước đó.
- Phần 3 đã viết xong nhóm cuối nhưng timeout khi gọi bước ghép hình.
  Lỗi cuối không phải quota ngày; quota ngày là sự cố API 1 ở đầu tác vụ.

## Kết quả kiểm chứng thực tế

89 đơn vị là **đoạn/tình huống**, không phải 89 câu nói. Mục tiêu vẫn là
281 câu nói. Phần 3 chưa được kiểm chứng xong nên không có kết luận cho toàn phim.

78 đoạn được Gemini chấp nhận; 11 đoạn bị bác bỏ:

| Đoạn | Lý do Gemini ghi | Nhận định từ đối chiếu |
| --- | --- | --- |
| S0006 | Gian không cười ở đúng hình khi Nobita nằm giường | Có dấu hiệu kiểm tra quá chặt quan hệ đồng thời: lời dùng “Trước sự chế giễu”, không khẳng định hai việc cùng một khung hình. Cần đánh giá thứ tự trong cả tình huống. |
| S0007 | Không có động tác Nobita quỳ xin giúp | Lỗi nội dung/hình có căn cứ; chính fact đầu vào đã khẳng định động tác không được chứng minh ở các shot gửi đi. Cảnh nguồn cho thấy Nobita và đồ vật, không đủ chứng minh quỳ xin. |
| S0012 | Không có bố mẹ Shizuka; không nhìn thấy lời “không phải chuột” | Chi tiết bố mẹ là lỗi có căn cứ và có sẵn trong fact sai. Phần lời “không phải chuột” có hỗ trợ bằng transcript, nhưng audit chỉ nhận observed_facts/visuals, không nhận toàn bộ transcript cấp tình huống. Không thể bác lời thoại chỉ vì không thấy hành động minh họa. |
| S0017 | Máy đã nằm đất; tên bột DONBURA/BONBURA khác nhau | Audit bác cả fact mà TTS không nói: TTS không nêu động tác đặt máy xuống hay tên bột. Đây không phải bằng chứng đủ để bác lời thực tế. |
| S0031 | Thiếu SHOT_000233 | Clip/manifest thiếu hình tham chiếu; chưa đủ để kết luận nội dung sai. |
| S0032 | Thiếu SHOT_000239, SHOT_000240 | Như trên. |
| S0035 | Thiếu SHOT_000244 | Như trên. |
| S0047 | Thiếu SHOT_000366 | Như trên. |
| S0082 | Thiếu SHOT_000676 | Như trên. |
| S0083 | Thiếu SHOT_000682 | Như trên. |
| S0085 | Thiếu SHOT_000691, SHOT_000693 | Như trên. |

**7/11 lý do bác bỏ là thiếu bằng chứng được đóng gói**, không phải đã xác nhận
7 đoạn kể sai. Hai đoạn có lỗi nội dung rõ, hai đoạn có lý do bác cần xem lại.
78 verdict đạt cũng không bảo đảm mọi câu hoàn toàn chính xác.

## Lỗi dữ liệu và luồng xử lý

1. `evidence_manifest` ưu tiên `locked_visual_shots`, là tập hình đã chọn.
   Nhưng audit nhận `observed_facts` đầy đủ, gồm cả facts có shot bị bỏ khỏi tập chọn.
   51/89 đoạn có ít nhất một ID hình trong facts không nằm trong clip audit.
   Việc chọn hình đại diện tự nó không sai; gửi bộ facts rộng rồi yêu cầu
   kiểm mọi fact như thể clip đầy đủ là hợp đồng không nhất quán.
2. Manifest đánh số lại `visual_id` theo vị trí sau khi lọc hình.
   Ví dụ V005 trở thành V003. 35/89 đoạn có mã V trong manifest khác mã V
   của hình đã khóa. ID SHOT vẫn còn, nên đây là nguy cơ gây nhầm chứ không
   chứng minh mọi verdict của 35 đoạn đều sai.
3. Có 27 lỗi `invalid_spoken_trigger`: 9 cụm không xuất hiện nguyên văn,
   2 cụm quá giới hạn 2–6 từ, 16 cụm có mặt đúng một lần nhưng bị quy tắc
   thứ tự/ranh giới chặn. Nhóm 16 không phải toàn bộ đã xác minh là lỗi trình tự;
   nó cần phân biệt điểm đầu đoạn, thứ tự lời và thứ tự hình.
4. Whisper có nhiễu: câu Nhật chung chung lặp trên vùng 30 giây;
   cache thô còn có cùng câu quảng bá lặp 42 lần. Một số dòng quảng bá đã
   được bước lọc loại, nhưng vài câu Nhật bất thường vẫn đi vào facts.
   Confidence mặc định 0.75 không đủ chứng minh transcript đó đúng.
5. Bản mô tả hình đầu vào vẫn có hallucination. Khóa ID/time không biến
   một mô tả sai thành bằng chứng đúng. Ví dụ “bố mẹ Shizuka” và “quỳ xin”.
6. Lỗi thiếu hình, lỗi trigger, sai tên và sai nội dung cùng đi vào vòng
   sửa lời; một sửa lỗi cấu trúc có thể kéo theo tạo lời mới và lệch số câu.

## Một phương án sửa thống nhất, không tăng retry

Giữ một hồ sơ tình huống có ID bất biến cho nhân vật, fact, shot, thời gian
gốc và lời thoại đã lọc. Dùng cùng hồ sơ ở mọi bước; không đánh lại mã V.

- Tái sử dụng clip đã tạo khi chuyển API. Cache clip theo file nguồn,
  vùng thời gian và cấu hình lấy hình; API key không quyết định danh tính clip.
- Trong **lượt viết đang có**, cung cấp đúng nhóm hình của tình huống cùng
  transcript đã lọc. Gemini viết dựa trên hình thực tế, trả câu và fact/shot
  chứng minh từng ý; chi tiết không được chứng minh phải bỏ hoặc viết lại
  ngay trong lượt này, trước khi khóa lời. Không thêm lượt kiểm chứng toàn bộ.
- Backend ghép từ các ID và vị trí từ trong lời đã chốt. Không yêu cầu Gemini
  chép lại lời rồi sinh cụm trigger 2–6 từ. Nếu thông tin là lời thoại, dùng
  cảnh nói/chứng kiến/phản ứng cùng tình huống và giữ loại bằng chứng là lời thoại.
- Gộp chức năng đối chiếu vào lượt viết hiện có, thay cho vòng ghép lại lời
  và kiểm video đại trà sau đó. Không chỉ tắt kiểm tra rồi tin facts cũ.
  Đếm đủ 281 câu và kiểm ID/phạm vi/thứ tự bằng code; lỗi metadata xử lý ở
  metadata, không gửi viết lại cả đoạn. Giữ số lượt retry hiện tại.

Phương án này dự kiến giảm số yêu cầu và thao tác đóng gói/upload lặp.
Chưa triển khai hoặc đo một lần chạy mới nên chưa cam kết số phút tiết kiệm.
