# Tách ngữ cảnh và sự kiện dùng để viết

Hợp đồng evidence phiên bản 8 yêu cầu lượt xem chi tiết trả `granularity="atomic"` cho một hành động/thay đổi cụ thể, có thời gian nguồn. Một sự kiện có thể chứa nhiều ý/câu, không bị ép số giây. Tóm tắt một chặng truyện đi vào `story_context`. Mục không rõ được giữ ở `event_review_candidates` để kiểm tra, không được dùng để cấp câu hay timestamp dựng.

Backend giữ nguyên hồ sơ nguồn, gồm cả visual_sources, của mục tổng quan/chưa rõ; không xóa dữ liệu và không tự chia một vùng dài thành sự kiện giả. Các phụ thuộc tới mục này chuyển sang `context_dependencies` thay vì cạnh thứ tự giữa các sự kiện ngang hàng.

Với dữ liệu cũ thiếu nhãn, một mục dài, có nhiều fact và chứa nhiều sự kiện chi tiết hoặc không có visual_sources sẽ được đánh dấu cần kiểm tra, không tự khẳng định đó là summary. Thời lượng dài đơn lẻ không đủ để loại một sự kiện; hành động dài được xác nhận atomic vẫn giữ nguyên. Sức chứa câu có thể giảm sau khi tách; cần thu đủ sự kiện thật thay vì dùng tổng quan để bù.

Ở lượt triển khai, Gemini trả `event_plans`: một nhóm ID sự kiện với các `focuses` bên dưới. Backend ghép các nhóm theo thứ tự nguồn, giữ thứ tự ý trong cùng sự kiện. Một nhóm nhiều sự kiện liên tiếp có thể được nén thành một focus. ID không được xuất hiện ở nhiều nhóm. Bản nháp phẳng và patch cũ không được tự sắp lại nội dung; vẫn đi qua kiểm tra và cơ chế vá riêng.

Đây không phải bằng chứng rằng AI đã nhận diện đúng mọi cảnh hoặc đúng thứ tự ý trong một sự kiện. Vẫn kiểm tra ID, dữ kiện, ý lặp và liên kết hình. Cần phân tích mới để có đầy đủ dữ liệu hợp đồng mới; không tái sử dụng checkpoint evidence phiên bản 7 như dữ liệu đã chuẩn hóa.
