# Luồng recap có hình nguồn trước khi viết

1. Backend đánh số các vùng video nguồn dài tối đa 6 giây (`SHOT_000001`...).
   Đây là địa chỉ thời gian cố định, không phải kết quả nhận diện chuyển cảnh.
   Clip gửi Gemini được gắn trực tiếp các nhãn này lên hình.
2. Gemini xem các clip tối đa khoảng 8 phút, có phần chồng lấn, dàn ý toàn phim
   và trạng thái phần trước. Mỗi dữ kiện phải đi cùng ID nhìn thấy trên hình.
   Backend tra thời gian từ bảng ID; không dùng timestamp Gemini đoán.
3. Loại logo/credits, ID không tồn tại và ID ngoài clip. Gom các dữ kiện dùng chung
   vùng hình thành một đơn vị để tránh chia hình tùy ý hoặc đếm trùng sức chứa.
   Nếu thiếu, tối đa hai lượt xem bổ sung những vùng thiếu trước khi viết.
4. Lập mạch truyện, nhóm sự kiện và phân bổ đúng số ô câu mục tiêu. Backend chia
   dữ kiện đã quan sát vào các ô, giữ hành động và kết quả; không gọi AI để tạo
   thêm ý cho đủ quota. Một sự kiện có thể nhận nhiều câu nếu có dữ kiện/hình riêng.
5. Khóa fact ID và các vùng hình cho từng ô. Viết lời ba phần bằng text; mỗi ô
   trả một câu và liên kết các cụm lời với hình đã khóa. Kiểm tra lời có thêm
   chi tiết ngoài dữ kiện không, chỉ sửa những câu lỗi. Lưu kết quả kiểm tra để
   không kiểm tra lại câu không đổi.
6. Ghép ba phần, giữ ID và hình đã khóa; không tìm hình lại sau khi viết.
   Chỉ xuất khi đủ số câu, mọi câu có hình và qua kiểm tra dữ kiện.

Phiên bản bằng chứng 12 và workflow 7 không tái sử dụng kế hoạch/lời cũ thiếu
liên kết nguồn. Cần tạo lượt phân tích mới sau khi khởi động lại ứng dụng.

Bộ đọc quan sát nhận `{"events": [...]}`, danh sách sự kiện trực tiếp `[...]`
và một bộ kết quả bọc trong danh sách `[{"events": [...]}]`,
giữ đủ sự kiện trong danh sách. Phản hồi gốc và dàn ý được lưu trong
`.cache/grounded_observations`. Khi phân tích lại cùng file, mục tiêu, phiên bản
bằng chứng và phụ đề, tool dùng lại phản hồi hợp lệ trước khi gửi video.
File thay đổi kích thước/thời gian sửa, mục tiêu hoặc phụ đề thay đổi sẽ dùng
cache khác. Cache lỗi không thay thế việc kiểm tra SHOT ID và thời gian nguồn.
SHOT ID được chuẩn hóa phần số: `SHOT_0000162` và `SHOT_000162` cùng chỉ shot 162.
Mã sai chữ số, không tồn tại hoặc ngoài clip vẫn bị loại; không tìm mã gần giống.

Lỗi ID ở bước gom cụm được sửa từ danh mục sự kiện nguồn: bỏ ID không tồn tại,
loại quyền sở hữu trùng, bổ sung sự kiện bị sót và giữ thứ tự nguồn. Cụm hợp lệ
được giữ khi có thể; backend chia lại các cụm cần thiết rồi cân bằng đúng số câu.
Phản hồi thô và chẩn đoán được lưu trước khi kiểm tra. Nếu JSON gom cụm lỗi,
backend dựng nhóm từ sự kiện đã có. Nhãn truyện chưa xác định được sẽ để trống.

Nếu các vùng hình xen kẽ làm kế hoạch không thể ghép theo thứ tự, backend sắp
lại các dữ kiện vào đúng số ô câu và chọn một vùng hình đã quan sát cho mỗi ý.
Đây là hình minh họa từ bằng chứng, có thể chỉ thể hiện một phần của hành động
dài; không khẳng định hình minh họa thay thế toàn bộ diễn biến của hành động đó.
Không chọn hình ngoài danh mục để che lỗi ID hay tự tạo sự kiện mới.

Giới hạn: ID hợp lệ chứng minh vị trí nguồn, không tự chứng minh Gemini hiểu đúng
hành động. Việc nhận diện và kiểm tra ngữ nghĩa vẫn có thể sai. Nếu thực sự thiếu
bằng chứng sau ngân sách bổ sung, hệ thống dừng trước khi viết thay vì bịa câu.
Quota và timeout của dịch vụ vẫn có thể xảy ra; checkpoint giữ các bước đã xong.

Kiểm tra hồi quy: `python -m unittest discover -s tests -q`.
`tests/test_grounded_pipeline.py` kiểm tra đủ 156 câu trước khi viết, một sự kiện
nhiều câu, gom dữ kiện chung hình, ghép ba phần, nhiều hình/câu, mốc thời gian
nguồn và sửa lời không có bằng chứng. Các bài kiểm tra mô phỏng phản hồi Gemini,
không xác nhận chất lượng trên phim thật.
