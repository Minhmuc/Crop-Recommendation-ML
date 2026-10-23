# Khung báo cáo do bạn tự hoàn thiện sau khi chạy

Không có kết quả thực nghiệm sẵn trong file này. Thay các ô trống bằng số liệu và nhận xét từ lần chạy của bạn.

## 1. Bài toán và dữ liệu

- Tên đề tài: Khuyến nghị loại cây trồng phù hợp dựa trên đặc điểm thổ nhưỡng và khí hậu.
- Đầu vào: 7 đặc trưng; đầu ra: một trong 22 nhãn và bảng top-3.
- Nguồn Kaggle, tác giả công bố, ngày tải, phiên bản, điều khoản sử dụng: ...
- Số dòng thực tế, số nhãn, phân bố nhãn, dòng trùng/thiếu: ...
- Ý nghĩa và thang đo N/P/K; kỳ tích lũy rainfall; những thông tin nguồn chưa nêu rõ: ...
- Phân tích 2–3 biểu đồ phân bố/tương quan từ tập train: ...

## 2. Cơ sở thuật toán

Giải thích Euclidean, Manhattan, Minkowski; p=1 và p=2 là hai trường hợp riêng. K là số láng giềng, không phải cột K (kali). Ví dụ tính tay khoảng cách trong notebook: ...

Giải thích uniform/distance, StandardScaler/MinMaxScaler, vai trò của Pipeline chống rò rỉ dữ liệu. KD-Tree chia theo trục; Ball-Tree nhóm trong các vùng hình cầu; brute tính trực tiếp. Tốc độ phụ thuộc dữ liệu, không tự kết luận cây luôn nhanh hơn.

## 3. Thiết kế thí nghiệm

80% train, 20% test, stratify và seed=42; 5 fold stratified dùng chung trên train; chọn cấu hình theo F1 macro. Ghi lưới tham số và tổng số cấu hình từng mô hình. Chỉ dùng test sau khi đã chọn xong. CV dùng để chọn siêu tham số nên điểm CV tốt nhất có thể lạc quan; đánh giá cuối nằm trên holdout test.

## 4. Kết quả thực tế

| Mô hình | CV Accuracy ± SD | CV F1 macro ± SD | Test Accuracy | Test F1 macro |
|---|---|---|---|---|
| KNN | ... | ... | ... | ... |
| Gaussian NB | ... | ... | ... | ... |
| Decision Tree | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |

SD là độ lệch chuẩn giữa các fold, không phải khoảng tin cậy. Dẫn file `comparison.csv`, `knn_cv.csv`, các ma trận nhầm lẫn. Với khảo sát từng yếu tố, dùng các đường cong giữ nguyên các yếu tố còn lại.

- KNN tốt nhất dùng K, scaler, metric, weights nào? ...
- Những cặp cây bị nhầm nhiều nhất? Dẫn số mẫu cụ thể: ...
- Thay chuẩn hóa giúp/hại ra sao? Độ lớn ảnh hưởng có đáng kể so với dao động các fold? ...
- So sánh thời gian brute/KD-Tree/Ball-Tree trên máy gì, bao nhiêu mẫu truy vấn, bao nhiêu lần lặp? ...
- Một ví dụ dự đoán sai và giả thuyết giải thích, phân biệt quan sát với suy đoán: ...

## 5. Minh họa và giới hạn

Chụp bảng top-3 và nêu cách tính xác suất KNN. Nếu chỉ 1–2 nhãn nhận phiếu, giải thích vì sao các dòng còn lại 0% và không nên coi đó là khuyến nghị có căn cứ. Không đổi K sau khi xem test chỉ để bảng trông đẹp hơn.

Dữ liệu gán một nhãn cho mỗi mẫu; không có danh sách nhiều cây thực sự trồng được trên từng thửa đất. Vì vậy bài làm là mô phỏng phân lớp và xếp hạng nhãn theo dữ liệu, chưa chứng minh hiệu quả nông nghiệp ở Việt Nam. Min/max chỉ phát hiện một số đầu vào ngoài miền; không xác nhận một tổ hợp đầu vào nằm trong phân bố dữ liệu.

## 6. Nguồn tham khảo và đóng góp

Ghi nguồn dữ liệu, tài liệu thuật toán, thư viện. Mô tả các phần bạn tự thực hiện: chạy, kiểm tra dữ liệu, điều chỉnh thí nghiệm trước khi mở test, đọc lỗi, nhận xét, trình bày. Nêu việc dùng trợ lý AI hỗ trợ mã nguồn theo yêu cầu của môn học; không nhận số liệu chưa chạy là kết quả của mình.

## Câu hỏi tự ôn

1. Vì sao `fit_transform` toàn bộ dữ liệu trước CV là sai?
2. KNN có train không, vì sao không có epoch như CNN?
3. Vì sao K nhỏ dễ nhạy nhiễu, K lớn có thể làm mất ranh giới lớp?
4. K lẻ có loại hết đồng phiếu trong bài toán 22 lớp không?
5. p=1, p=2, p=3 trong Minkowski khác nhau thế nào?
6. Vì sao không đổi nhãn cây thành 0..21 rồi dùng Pearson như một biến liên tục?
7. Tại sao Accuracy và F1 macro có thể khác nhau dù dữ liệu cân bằng?
8. Độ chính xác cao trên dữ liệu này có đủ để triển khai tư vấn nông nghiệp không?
9. MinMaxScaler có luôn cho giá trị trong [0,1] ở dữ liệu mới không?
10. Có nên coi 100% phiếu KNN là chắc chắn đúng không?
