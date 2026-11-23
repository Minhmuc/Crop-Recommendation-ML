---
name: lanh-dao-sub-agent
description: Điều phối nhiều sub agent rà soát độc lập một sản phẩm (báo cáo, mã nguồn, tài liệu) rồi kiểm chứng chéo và tổng hợp. Dùng khi người dùng yêu cầu "rà soát bằng sub agent", "đóng vai lãnh đạo chỉ đạo agent", "đánh giá độc lập", "kiểm tra ảo giác", hoặc cần nhiều góc nhìn song song cho một sản phẩm lớn.
---

# Lãnh đạo sub agent rà soát

Vai trò của bạn là **trưởng nhóm**: chia việc, giao đề bài rõ ràng, giữ các agent độc lập, kiểm chứng kết quả của họ và chịu trách nhiệm về bản tổng hợp cuối. Bạn không chuyển tiếp nguyên văn kết quả agent cho người dùng.

## Nguyên tắc

1. **Độc lập:** không đưa kết luận, đánh giá trước đó hay "đáp án mong đợi" của bạn vào đề bài. Agent chỉ nhận tài liệu gốc, tiêu chí và câu hỏi.
2. **Có bằng chứng:** mọi phát hiện phải kèm vị trí (mục, số đoạn, trang, tên file/dòng) và trích dẫn ngắn hoặc giá trị đối chiếu. Phát hiện không có bằng chứng bị loại.
3. **Phân biệt đã kiểm chứng / suy luận:** agent ghi rõ mỗi phát hiện là `KIỂM CHỨNG` (đã đối chiếu với nguồn) hay `NHẬN ĐỊNH` (đánh giá chủ quan).
4. **Chỉ đọc:** agent không sửa file của người dùng, không chạy thứ có tác dụng phụ (huấn luyện mô hình, ghi đè output, cài thư viện) trừ khi người dùng cho phép.
5. **Lãnh đạo kiểm chứng lại:** trước khi báo cáo, tự kiểm tra các phát hiện nghiêm trọng và các phát hiện mâu thuẫn giữa agent. Ghi rõ phát hiện nào đã được bạn xác nhận, phát hiện nào bị bác bỏ và vì sao.
6. **Trung thực về giới hạn:** nói rõ phần nào chưa được rà (thiếu công cụ, ngoài phạm vi), không lấp chỗ trống bằng phỏng đoán.

## Quy trình

### Bước 1 – Chia góc rà soát
Chọn 3–5 góc không chồng lấn. Ví dụ cho báo cáo học thuật:

| Góc | Câu hỏi chính | Mô hình gợi ý |
|---|---|---|
| Đối chiếu tiêu chí chấm | Mỗi tiêu chí đạt bao nhiêu, mất điểm ở đâu? | Mô hình mạnh (cần phán đoán) |
| Kiểm chứng số liệu | Mọi con số có khớp dữ liệu/kết quả gốc không? | Mô hình trung bình, giỏi dùng công cụ |
| Văn phong và trình bày | Thuật ngữ, câu, quy cách hình–bảng–trích dẫn | Mô hình trung bình |
| Phản biện khoa học | Lỗ hổng phương pháp, lập luận yếu, câu hỏi vấn đáp | Mô hình mạnh (cần lập luận sâu) |

Chọn mô hình theo độ khó: phán đoán và lập luận → mô hình mạnh; dò soát có hệ thống, đối chiếu bằng công cụ → mô hình trung bình; tra cứu đơn giản → mô hình nhẹ.

### Bước 2 – Chuẩn bị đầu vào chung
- Trích văn bản sản phẩm sang định dạng dễ đọc (ví dụ docx → txt có đánh số đoạn và tên style).
- Chuẩn bị tiêu chí đánh giá ở dạng văn bản.
- Liệt kê đường dẫn tuyệt đối tới các nguồn gốc (dữ liệu, kết quả, bản PDF).
- Dùng cùng một bộ đầu vào cho mọi agent.

### Bước 3 – Viết đề bài cho từng agent
Mỗi đề bài tự đầy đủ (agent không thấy cuộc hội thoại), gồm:
1. Bối cảnh ngắn: sản phẩm là gì, dùng để làm gì.
2. Đường dẫn đầu vào và nguồn gốc được phép đọc.
3. Phạm vi góc rà soát và những gì **không** thuộc phạm vi.
4. Ràng buộc: chỉ đọc; lệnh/công cụ được phép.
5. Định dạng đầu ra bắt buộc: bảng phát hiện gồm mức độ (Nghiêm trọng / Nên sửa / Nhỏ), vị trí, bằng chứng, loại (KIỂM CHỨNG / NHẬN ĐỊNH), đề xuất sửa; kèm danh sách "đã kiểm tra và không có vấn đề" và "chưa kiểm tra được".
6. Giới hạn độ dài báo cáo trả về.

### Bước 4 – Chạy song song
Gửi tất cả agent trong cùng một lượt, chạy nền. Không đoán trước kết quả; đợi thông báo hoàn thành.

### Bước 5 – Kiểm chứng chéo
- Với mỗi phát hiện **Nghiêm trọng** hoặc có số liệu: tự đối chiếu với nguồn (đọc file, chạy script chỉ đọc).
- Gộp phát hiện trùng giữa các agent; ghi nhận khi nhiều agent độc lập cùng phát hiện (độ tin cậy cao hơn).
- Khi agent mâu thuẫn nhau: kiểm tra nguồn để phân xử.
- Đánh dấu kết quả: `Đã xác nhận`, `Bác bỏ (lý do)`, `Chưa xác minh`.

### Bước 6 – Tổng hợp cho người dùng
- Kết luận chung trước (đạt/chưa đạt, rủi ro lớn nhất).
- Bảng phát hiện đã xác nhận, xếp theo mức độ.
- Các phát hiện bị bác bỏ (ngắn gọn, để người dùng thấy quá trình kiểm chứng).
- Giới hạn của lần rà soát.
- Không tự sửa sản phẩm nếu người dùng chưa đồng ý.
