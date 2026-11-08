# Khuyến nghị cây trồng bằng KNN

**Notebook chính của dự án là [crop.ipynb](crop.ipynb).** Dùng file này cho toàn bộ quá trình học, thực nghiệm và demo cuối môn. Notebook mẫu `Crop recommendation final.ipynb` dùng để tham khảo cách trình bày.

## Trạng thái

- `crop.ipynb` có 5 giai đoạn, 21 mục và 139 ô mã ngắn. Output đã được xóa để bạn tự chạy lại.
- Mặc định chọn dataset A với `RUN_VERSION = '02'`, lưu tại `outputs/crop_a_02`. Chọn B sẽ lưu tại `outputs/crop_b_02`.
- Notebook tự tạo thư mục kết quả khi bạn chạy. Phần tổng hợp đọc A và B có cùng `RUN_VERSION`.
- Báo cáo đang dùng là `dangbaocao.docx`. Cập nhật báo cáo từ output của đúng dataset và lần chạy sau khi đã kiểm tra.

## Dữ liệu và mô hình

| Dataset | File | Số dòng gốc | Số nhãn sau thống nhất chữ thường |
|---|---|---|---|
| A | `data/crop_data1.csv` | 2.200 | 22 |
| B | `data/crop_data2.csv` | 1.796 | 19 |

Đầu vào gồm `N, P, K, temperature, humidity, ph, rainfall`. Đầu ra là `label`. Cả hai CSV được tải từ [thư mục Datasets của kho KRUTHIKTR](https://github.com/KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning/tree/main/Datasets). Đã đối chiếu SHA-256 ngày 08/11/2026, cả hai khớp từng byte với bản trên GitHub. Kho công bố MIT, còn hồ sơ thu thập và gán nhãn riêng cho B cần làm rõ. Mô tả A được đối chiếu thêm với [Crop Recommendation trên Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

Hai bộ được thử nghiệm riêng. Có nhiều đầu vào chồng lặp và một số nhãn khác nhau giữa hai bộ, cần nêu hạn chế này khi nhận xét khả năng tổng quát hóa. Giữ nguyên các nhãn trong từng nguồn, kể cả `groundnut` và `groundnuts` khi chưa có căn cứ đồng nhất.

| Mô hình | Vai trò |
|---|---|
| KNN | Thuật toán chính, khảo sát K, khoảng cách, trọng số và chuẩn hóa |
| Gaussian Naive Bayes | Baseline đơn giản, thử ba giá trị var_smoothing trên CV |
| Decision Tree | Mô hình so sánh |
| Random Forest | Mô hình so sánh |
| KNN_final | Biến thể KNN được chọn sau thí nghiệm cấu hình và đặc trưng |

## Cách chạy

1. Mở `D:\Crop-Recommendation-ML` trong VS Code và mở `crop.ipynb`.
2. Chọn kernel `D:\Crop-Recommendation-ML\.venv\Scripts\python.exe`.
3. Chạy từng ô bằng Shift+Enter. Đọc mục 1 rồi chạy phần chuẩn bị ở mục 2–4 trước khi fit mô hình.
4. Hoàn thành lần chạy A, lưu notebook và kết quả. Restart Kernel, đổi cấu hình sang B rồi chạy lần lượt cho bộ B.

```python
DATA_MODE = 'nguon_a'  # Đổi thành 'nguon_b' cho lần chạy B.
RUN_VERSION = '02'
RUN_NAME = f'crop_{DATA_MODE[-1]}_{RUN_VERSION}'
```

Nếu kernel báo thiếu thư viện, tự cài vào đúng môi trường bằng terminal rồi Restart Kernel.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Nội dung từng phần

| Mục | Thực hiện | Kết quả cần đọc |
|---|---|---|
| 1–4 | Bài toán, đọc dữ liệu, thống nhất định dạng, loại trùng, chia tập | Số mẫu/nhãn, bảng trước–sau, train/test và năm fold |
| 5 | EDA trên train | Phân bố, tương quan, boxplot, ngoại lai và minh họa chuẩn hóa |
| 6 | Fit một KNN minh họa | Một mẫu validation, láng giềng, phiếu và xác suất |
| 7–8 | GridSearch KNN và khảo sát từng yếu tố | Tham số được chọn, CV ± SD, chênh lệch train–validation |
| 9 | Fit Gaussian NB, cây và rừng trên cùng fold | Kết quả ba mô hình đối chiếu |
| 10 | Ablation bốn biến thể, chọn đặc trưng bằng mutual information | Đóng góp của đổi cấu hình và chọn 5/7 đặc trưng |
| 11 | Learning curve và benchmark brute/KD-Tree/Ball-Tree | Độ ổn định, ảnh hưởng lượng dữ liệu, thời gian tìm láng giềng |
| 12 | Tổng hợp CV, chốt mô hình | selection.json và biểu đồ train–CV |
| 13–14 | Đánh giá test và phân tích lỗi | Accuracy/F1 macro, ma trận nhầm lẫn, chỉ số theo lớp, mẫu sai |
| 15–17 | Lưu/nạp Pipeline, kiểm tra và demo | knn_crop.joblib, kiểm tra đầu vào, top-3 nhãn |
| 18–20 | Tổng hợp theo dataset, bảng báo cáo và đóng gói | CSV, hình và ZIP của thư mục kết quả |
| 21 | Nguồn tham khảo | Dữ liệu, notebook mẫu và tài liệu thuật toán |

Các ô có `.fit` thực hiện huấn luyện. Hoàn tất lựa chọn ở mục 7–12 trước khi dùng test tại mục 13. KNN lưu mẫu khi fit, tìm láng giềng khi dự đoán, số lần fit CV phụ thuộc số cấu hình và số fold.

## Thiết kế thực nghiệm

- Chia khoảng 80/20 ở mức nhóm, phân tầng theo nhãn, seed 42. Những đầu vào giống nhau sau làm tròn sáu chữ số được giữ cùng phía, giá trị đưa vào mô hình được giữ nguyên.
- CV năm fold bằng StratifiedGroupKFold trên train. Các mô hình dùng cùng fold, scaler và bộ chọn đặc trưng fit riêng trong mỗi fold.
- KNN thử 162 cấu hình, gồm chín giá trị K, ba cách chuẩn hóa, Euclidean/Manhattan/Minkowski p=3 và hai cách bỏ phiếu.
- Gaussian NB thử ba cấu hình, Decision Tree sáu cấu hình, Random Forest bốn cấu hình với 200 cây.
- Ablation so sánh start_7, tuned_7, start_5, tuned_5 trên cùng fold. Chọn bằng CV F1 macro, khi đồng điểm lấy biến thể xuất hiện trước. KNN_final có thể giữ đủ bảy đặc trưng.
- Báo CV Accuracy/F1 macro trung bình ± độ lệch chuẩn, tách khỏi điểm hold-out. CV dùng để lựa chọn nên điểm tốt nhất có thể lạc quan.
- Mục 18 hiển thị dataset đã hoàn thành. Khi có một bộ, xuất `ket_qua_dataset_hien_co.csv`. Khi đủ A/B và các điều kiện khớp, xuất `doi_chieu_hai_dataset.csv`.

Test A và B đã từng được dùng trong những lần chạy trước. Khi chạy lại, cần mô tả đúng lịch sử đánh giá. Đổi RUN_NAME giúp lưu riêng kết quả, tính độc lập của test phụ thuộc thiết kế và cách sử dụng dữ liệu.

## Minh chứng Bài kiểm tra 2

| Tiêu chí | Vị trí |
|---|---|
| 1. Lựa chọn và lý giải thuật toán | Mục 1, 7, 9 |
| 2. Baseline | Gaussian NB ở mục 9, bảng mục 12–13 |
| 3. Quy trình thống nhất, tránh rò rỉ | Mục 4, Pipeline ở mục 7, 9–10 |
| 4. Cài đặt Python rõ ràng | Toàn notebook, output sau khi tự chạy |
| 5. CV trên train, test cuối, mean/SD | Mục 4, 7–13 |
| 6. Chỉ số, so sánh và ma trận | Mục 12–13, 18 |
| 7. Quá khớp và phân tích lỗi | Mục 8, 11–14 |
| 8. Giải thích nguyên lý | Mục 6–10, mỗi thành viên tự luyện |
| 9. Giải thích kết quả trên dữ liệu | EDA, ablation và bảng lỗi |
| 10. Thực hành cá nhân | Tự luyện thay tham số, chạy CV và đọc kết quả |

Mục 10–11 có thêm phần dùng cho kiểm tra 3 và thi cuối. Nhận xét, phân công thực tế và báo cáo ICTU được hoàn thiện từ kết quả đã xác nhận.

## Xem lại kết quả và xử lý lỗi

- Chỉ chạy lại ô đã lỗi và các ô phụ thuộc khi các biến trước đó còn trong kernel.
- Nếu kernel đã tắt, chạy lại các ô chuẩn bị. Đặt `RESTORE_SEARCHES=True` ở mục 7 để nạp kết quả của đúng lần chạy, bỏ qua ô fit tương ứng.
- Nếu đã có `comparison.csv` hoặc `test_predictions.csv`, dùng ô đọc kết quả đã lưu. Các ô fit/tuning và đánh giá test có kiểm tra để tránh ghi đè.
- Nếu nạp lại KNN_final, vẫn chạy ô khai báo lưới ở mục 10 và ô chọn `final_search` để khôi phục tên biến và danh sách biến thể.
- Lỗi learning curve do `pos_label=1` đã được sửa bằng scorer có `pos_label=None`, F1 macro và danh sách lớp cố định.
- Sửa `RUNS` ở mục 18 nếu đổi tên các thư mục lần chạy. Bộ chưa có đủ file sẽ được báo rõ và chờ hoàn thành.
- Mô hình demo ở `outputs/<RUN_NAME>/knn_crop.joblib`, gồm scaler, bộ chọn đặc trưng và KNN. Top-3 thể hiện điểm phiếu, khả năng canh tác thực tế cần được kiểm chứng thêm.

Có hỗ trợ soạn mã bằng AI. Người thực hiện tự chạy, kiểm tra và giải thích mã cùng kết quả. Các tài liệu tham khảo chi tiết được dẫn ở mục 21 của notebook.
