# Lập kế hoạch recap ba giai đoạn

Ba giai đoạn xử lý bằng chứng chữ đã lấy từ video, không gửi lại phim. Tên hàm/checkpoint `two_stage` được giữ để tương thích dữ liệu, phiên bản workflow hiện tại là 2.

## 1. Xương sống toàn phim

Model đọc bản đồ sự kiện gọn: ID, thời gian, nhân vật, nguyên nhân, hành động, kết quả. Nó khóa sáu anchor, các bước ngoặt và mức quan trọng. Lượt này chưa chia cụm, chưa chia ba phần, chưa phân bổ câu. Checkpoint lưu riêng `backbone`. Dữ liệu shot và transcript đầy đủ không gửi trong yêu cầu này.

## 2. Chia cụm kể

Model nhận sự kiện và xương sống đã khóa, chia ba phần và nhóm các sự kiện liên tiếp. Backend giữ nguyên các anchor, bước ngoặt, nội dung xương sống và dùng lại mức quan trọng của lượt một. Số câu từng cụm chỉ là dự kiến, backend cân bằng để giữ tổng mục tiêu. Khung phân cụm có checkpoint riêng.

## 3. Triển khai từng cụm

Mỗi yêu cầu nhỏ nhận khung toàn phim, dữ kiện của cụm, context liên quan và hai ô cuối của cụm trước. Model trả ID sự kiện nguồn và ý riêng cho từng ô câu. Nó chưa viết TTS, không đổi khung và không tự tạo timestamp. Một sự kiện có thể có nhiều ô với ý khác nhau; nhiều sự kiện liền nhau có thể nằm trong một ô.

Backend kiểm tra số câu, ID, thứ tự, ý lặp nguyên văn và khả năng tạo các ô từ dữ kiện hiện có. Sau đó mới cấp ID câu và vùng hình, giữ nguyên ý do model phân bổ. Kiểm tra ID/ý lặp không bảo đảm phát hiện mọi trường hợp lặp nghĩa hoặc suy diễn.

Khi bản triển khai có ô đảo thứ tự/ID sai/ý trùng, backend lưu toàn bộ bản nháp trong `detail_drafts` và ghi `detail_diagnostics` với số ô, ID trước–sau, cặp ô xung đột và thứ tự ID mong đợi. Lượt sửa chỉ gửi các ô lỗi, ngữ cảnh liền kề chỉ đọc và các fact liên quan. Gemini trả patch theo `row_index`; backend chỉ thay đúng các ô được yêu cầu, không cho chèn/xóa/di chuyển ô hay sửa các ô còn lại. Sau patch phải kiểm tra lại toàn cụm, ý lặp, khả năng triển khai từ bằng chứng và tổng câu toàn phim trước khi ghép. Không tự sắp lại nội dung hoặc đổi ID để che lỗi. Ngắt kết nối trong lượt vá giữ bản nháp và các cụm đã hoàn tất cho lần tiếp tục cùng tác vụ.

## Checkpoint và retry

`two_stage_planning_checkpoint` lưu xương sống, khung và từng cụm hoàn tất cùng fingerprint của nguồn/cấu hình. Sửa dữ liệu sai tối đa ba lượt cho đúng bước hoặc cụm. Nhật ký lưu model và số thứ tự từng lần gọi. Chuỗi model: `3.8-flash → 3.7-flash → 3.6-flash → 3.5-flash → 3.5-flash-lite → 3.1-flash-lite`.

Trong workflow ba giai đoạn, bắt đầu bằng 3.8 và giữ model vừa trả kết quả thành công cho các yêu cầu sau, kể cả lượt sửa. Gặp 503 thì xuống một model; gặp lỗi mạng tạm thời/ngắt kết nối/timeout thì chờ 60 giây rồi retry đúng yêu cầu trên model hiện tại. Mỗi yêu cầu có tối đa 5 lần retry; quota ngày, lỗi xác thực và request sai không được retry theo cơ chế này. Lỗi xảy ra sau checkpoint nào chỉ cần tiếp tục từ checkpoint đó khi chạy lại cùng tác vụ.

Khi tác vụ được tiếp tục với cùng checkpoint, khung và các cụm đã xong được tái sử dụng. Bấm phân tích mới vẫn tạo tác vụ mới theo giao diện hiện có; tài liệu này không thay đổi cơ chế đó.

Số câu từng cụm là ngân sách dự kiến. Nếu khung đúng cấu trúc nhưng phân bổ vượt sức chứa hoặc sai tổng, backend tự cân bằng số nguyên trong sức chứa ước lượng, ưu tiên giữ phân bổ gần đề xuất và các cụm quan trọng. Chỉ thay `sentence_count`, giữ nguyên thứ tự, ID sự kiện, summary và anchor; không gọi Gemini chỉ để tính lại số câu. Ví dụ 26–90–40 với sức chứa 25–94–39 được cân bằng thành 25–92–39, vẫn đủ 156 câu.

Lượt triển khai có thể trả ít/nhiều hơn dự kiến nếu đủ ý mới và còn khả năng giữ tổng mục tiêu. Backend khóa số câu thực tế của cụm đã đạt rồi cân bằng lại chỉ các cụm chưa viết. Cụm cuối cần hoàn thành phần câu còn lại. Nếu sức chứa ước lượng toàn phim thiếu, dừng với số liệu và các fact cụ thể để kiểm tra lại cách tính, không tự giảm mục tiêu hay nhồi câu lặp. Checkpoint `spine_draft` từ phiên bản trước vẫn được cân bằng cục bộ khi nạp lại trong cùng tác vụ.

Ba giai đoạn không đồng nghĩa luôn chỉ có ba API request: giai đoạn ba gọi theo cụm để giảm độ nặng từng request. Tổng token có thể tăng do gửi lại phần khung chung. Không bảo đảm loại bỏ 503 từ dịch vụ.
