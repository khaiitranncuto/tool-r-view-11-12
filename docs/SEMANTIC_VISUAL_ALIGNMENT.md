# Ghép hình theo ý kể

Luồng áp dụng cho kịch bản tạo mới:

1. Giữ ba lượt phân tích chi tiết, sự kiện và ID nhân vật hiện có.
2. Mỗi vùng hành động đã xem được backend cấp ID `V001`, `V002`… trong phạm vi một câu.
3. Cùng lượt viết recap, Gemini trả `visual_links`: ID hình đầu câu và cụm từ bắt đầu mỗi ý đổi hình.
4. Backend kiểm tra ID, cụm từ duy nhất trong lời đọc, thứ tự nguồn và vùng thời gian. Model không tự tạo timestamp.
5. Liên kết thiếu/sai được đưa vào lượt sửa riêng đúng câu, tối đa ba lượt; câu đạt giữ nguyên.
6. BEAT được chèn trước cụm từ. TTS đọc nguyên câu, renderer dùng word timing để đặt điểm chuyển hình.
7. Trong mỗi vùng hành động, hình chạy theo thứ tự tự nhiên, giữ các chuyển cảnh nguồn; không lấy mẫu đều theo vị trí hoặc ép shot 2–3 giây.

Vùng hành động có thể dài đến 15 giây. Đây là giới hạn dữ liệu nguồn, không phải thời lượng bắt buộc của một shot. Một câu có thể có hơn ba vùng hình.

Checkpoint lời viết cũ không được tái sử dụng như hợp đồng liên kết mới. Checkpoint sự kiện và kế hoạch không bị đổi phiên bản bởi thay đổi này.

Giới hạn: kiểm tra ID/timestamp không chứng minh AI nhận diện đúng hành động. Chất lượng còn phụ thuộc bằng chứng hình đầu vào. Nếu thiếu word timing, renderer ghi cảnh báo và ước lượng từ cues/số từ. Nếu vùng hình ngắn hơn giọng, cơ chế giữ khung cuối hiện có vẫn có thể được dùng. Chưa có bước AI xem lại mọi frame video thành phẩm.
