# Meta-prompt: Hoàn thiện dự án Học máy nhóm 08 cho Bài thi kết thúc

> Ngày lập: 22/11/2026. Dùng làm bản yêu cầu chuẩn khi giao việc hoàn thiện cho trợ lý AI hoặc khi một thành viên tự thực hiện.

## Vai trò

Bạn là trợ giảng môn Học máy kiêm biên tập báo cáo ICTU. Nhiệm vụ là đưa dự án "Khuyến nghị loại cây trồng phù hợp dựa trên đặc điểm thổ nhưỡng và khí hậu" (nhóm 08) đến trạng thái nộp Bài thi kết thúc, bám bộ tiêu chí trong `tieuchicham.docx`.

## Bối cảnh đã xác minh

| Thành phần | Vị trí | Trạng thái |
|---|---|---|
| Notebook chính | `crop.ipynb` | 139 ô mã đã chạy, không có lỗi |
| Kết quả dùng cho báo cáo | `outputs/crop_a_02`, `outputs/crop_b_02` | Đủ CSV, PNG, `knn_crop.joblib` |
| Kết quả cũ, không dùng | `outputs/crop_a_01`, `outputs/crop_b_01` | Giữ nguyên, phải nêu trong báo cáo |
| Báo cáo chính | `nhom 08.docx` | Đã đủ nội dung KT2, thiếu KT3 và phần thi cuối |
| Mẫu trình bày | `nhom08/Template_DA_KLTN_ICTU`, `nhom08/Huong_dan_su_dung_mau_DA_KLTN_ICTU.pdf` | Bắt buộc |
| Tiêu chí chấm | `tieuchicham.docx` | KT1, KT2, KT3, thi cuối |

## Đầu ra cần tạo

1. **Báo cáo `nhom 08.docx` hoàn chỉnh**
   - Sao lưu trước khi sửa vào `backups/nhom08_truoc_hoan_thien_cuoi_01.docx`.
   - Thêm mục **3.4 Cải tiến và tối ưu mô hình KNN** gồm:
     - hạn chế của mô hình ban đầu, có bằng chứng (điểm train–kiểm định, learning curve, lỗi tập trung theo lớp);
     - bảng hạn chế → giải pháp, ít nhất 2 giải pháp có cơ sở lý thuyết;
     - ảnh hưởng của từng siêu tham số (k, khoảng cách, trọng số, chuẩn hóa) qua lưới 162 cấu hình;
     - chọn đặc trưng bằng mutual information (5/7) trong Pipeline;
     - bảng ablation 4 biến thể (`start_7`, `tuned_7`, `start_5`, `tuned_5`) và đóng góp của từng thay đổi;
     - learning curve và benchmark brute/KD-Tree/Ball-Tree;
     - lựa chọn mô hình cuối, so sánh kiểm định chéo với test và độ ổn định qua fold.
   - Thêm mục **3.5 Lưu mô hình và chương trình minh họa** mô tả gói `knn_crop.joblib`, kiểm tra đầu vào, top-3 và ứng dụng web.
   - Tách mục 3.4 cũ thành phần **KẾT LUẬN** riêng gồm kết quả đạt được, hạn chế và hướng phát triển.
   - Thêm **DANH MỤC TỪ VIẾT TẮT**.
   - Cập nhật Mở đầu, mục lục, danh mục hình và bảng. Giữ nguyên bìa và hai bảng phân công.
   - Ghi rõ lý do dùng lần chạy `_02` và việc tồn tại lần chạy `_01`.
   - Giải thích vì sao mô hình cuối không vượt baseline Gaussian NB (tiêu chí 4 của thi cuối).
2. **Ứng dụng demo Streamlit** `app.py`
   - Chỉ nạp `outputs/crop_{a,b}_02/knn_crop.joblib`, không fit lại mô hình.
   - Giao diện đẹp, có nút sinh **input ngẫu nhiên**: lấy một mẫu thật trong dữ liệu đã làm sạch, hoặc sinh giá trị trong khoảng train.
   - Hiển thị top-3 kèm điểm phiếu, các láng giềng gần nhất, cảnh báo khi giá trị nằm ngoài khoảng train, và bảng chỉ số của mô hình.
   - Dùng lại đúng quy tắc kiểm tra đầu vào như mục 16–17 của notebook.
3. **README.md** và **data/README.md** cập nhật đúng trạng thái: tên báo cáo, hai CSV A/B, cách chạy notebook và demo.
4. **requirements.txt** thêm `streamlit`. Người dùng tự cài.

## Ràng buộc bắt buộc

- Không chạy notebook, không fit mô hình, không benchmark, không cài thư viện. Chỉ đọc CSV, JSON và PNG đã lưu.
- Mọi số liệu trong báo cáo phải lấy từ outputs `_02`, ghi đúng chỉ số và đúng tập (kiểm định chéo hay test).
- Không bịa phân công, chữ ký, ngày tháng, nguồn tham khảo hay minh chứng đóng góp.
- Không ghi đè outputs. Không commit Git.
- Test A/B đã được xem ở các lần phát triển trước: không gọi là test độc lập mới.
- Văn phong khoa học, ngắn gọn. Không viết lời hướng dẫn của AI hay lời nhắc chấm điểm trong báo cáo.
- Hình do nhóm tạo không cần dòng "Nguồn". Mỗi hình và bảng phải có nhận xét.

## Kiểm tra trước khi bàn giao

- [ ] Số liệu mới khớp `knn_ablation.csv`, `knn_ablation_effects.csv`, `knn_learning_curve.csv`, `neighbors_benchmark.csv`, `mutual_information_train.csv`, `comparison.csv`.
- [ ] Đánh số hình, bảng liên tục và mục lục được cập nhật.
- [ ] Xuất PDF và xem bản render: số trang ở giữa chân trang, bìa không có số trang, phần đầu dùng số La Mã.
- [ ] `app.py` không có lỗi cú pháp và đọc đúng cấu trúc gói joblib.
- [ ] Báo lại những việc người dùng phải tự làm: cài streamlit, chạy demo, xác nhận phân công, bổ sung nhật ký đóng góp.
