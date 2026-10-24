# Kiểm tra nguồn dữ liệu thứ hai và điều kiện gộp

Hiện báo cáo và notebook chính vẫn sử dụng một bộ Crop Recommendation. Chưa gộp dữ liệu mới, chưa thay số liệu trong báo cáo và chưa chạy thêm huấn luyện.

## 1. Yêu cầu của giảng viên

Theo yêu cầu bổ sung của giảng viên được nhóm xác nhận ngày 25/10/2026, dự án **bắt buộc sử dụng ít nhất hai bộ dữ liệu và gộp chúng**. Bộ tiêu chí bằng văn bản được thực hiện cùng yêu cầu bổ sung này. Dự án hiện có một nguồn, nên phần thu thập và tích hợp dữ liệu chưa hoàn thành.

Hai đường dẫn tải không nhất thiết là hai bộ dữ liệu độc lập. Việc nối hai bản sao chỉ tăng số dòng, không tăng thông tin. Hai nguồn khác đơn vị hoặc cách gán nhãn cũng không nên nối ngay.

## 2. Các nguồn đã tìm được

| Nguồn | Thông tin do bên công bố cung cấp | Cách xem xét |
|---|---|---|
| [Atharva Ingle — Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | Nguồn A hiện tại; CSV trong dự án có 2.200 mẫu, 7 đặc trưng và 22 nhãn. Metadata Kaggle ghi Apache 2.0. | Giữ làm bộ dữ liệu hiện tại. Nguồn chưa nêu rõ đơn vị N/P/K và kỳ tổng hợp rainfall. |
| [Crop recommendation — Sagana Thangatamilan](https://data.mendeley.com/datasets/vynxnppr7j/1) | Công bố 24/07/2025, CC BY 4.0. Mô tả có thông tin đất, khí hậu, dinh dưỡng và nói rõ tạo dữ liệu bằng CTGAN từ dữ liệu thu thập ở Tamil Nadu. | Ứng viên để tải và khảo sát thêm; cần công bố tính chất dữ liệu sinh. Chưa kiểm tra CSV, chưa xác nhận tương thích để gộp với nguồn A hoặc độc lập với A. Có thể nghiên cứu như một thí nghiệm riêng nếu phù hợp. |
| [Deep Learning Model Optimization for Crop Prediction and Recommendation — latha và Kumaresan](https://data.mendeley.com/datasets/2mht7hz98y/1) | Công bố 10/03/2025, CC BY 4.0; mô tả 2.200 mẫu, 7 đặc trưng, 22 cây. | Có cùng quy mô và cấu trúc với A. Phải kiểm tra trùng và nguồn gốc trước; chưa được coi là dữ liệu mới chỉ vì khác nơi đăng. |

Các thông tin trên được đối chiếu ngày 25/10/2026 từ trang công bố. Chưa có nguồn B nào được chốt là có thể gộp ngay. Giữ nguyên tên tác giả, DOI và giấy phép của từng nguồn khi sử dụng.

### Kết quả sàng lọc CSV ngày 25/10/2026

Việc kiểm tra dưới đây chỉ đọc dữ liệu ứng viên tải vào thư mục tạm, không chạy mô hình, không gộp hoặc thay dữ liệu của dự án:

| Ứng viên | Kết quả kiểm tra | Quyết định hiện tại |
|---|---|---|
| [raghavdharwal — field crop varieties](https://www.kaggle.com/datasets/raghavdharwal/crop-recommendation-dataset-field-crop-varieties) | CSV Crop_recommendation có 2.200 dòng; cả 2.200 dòng trùng nguồn A khi đọc vào pandas. | Không chọn làm nguồn quan sát bổ sung. |
| [irakozekelly — Crop Recommendation](https://www.kaggle.com/datasets/irakozekelly/crop-recommendation-dataset) | Có N/P/K, pH, rainfall, temperature và Crop, nhưng không có humidity. | Không gộp theo cấu trúc bảy đầu vào hiện tại. |
| [agriinnovate — Agricultural Crop Dataset](https://www.kaggle.com/datasets/agriinnovate/agricultural-crop-dataset) | Có 1.697 dòng và đủ bảy đầu vào. Sau làm tròn đặc trưng đến 4 chữ số thập phân, phép đối chiếu tìm được 1.603 cặp khớp đặc trưng với A, trong đó chỉ 1.056 cặp cùng nhãn. Các số là số cặp đối chiếu, không khẳng định số quan sát độc lập. | Cần làm rõ nguồn gốc và định nghĩa nhãn; không dùng các hàng gần trùng khác nhãn để gộp ngay. |
| [sanchitagholap — Western Maharashtra](https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra) | CSV có 4.513 dòng, gồm N/P/K, pH, rainfall, temperature, Crop và thông tin khác; không có humidity. | Chỉ xem xét nếu điều chỉnh bộ đặc trưng chung và xác minh thang đo, đơn vị; chưa chọn. |

Các nguồn thiếu humidity không thể bổ sung cột này bằng số ngẫu nhiên. Các nguồn trùng hoặc đổi nhãn cũng chưa đáp ứng mục tiêu có thêm dữ liệu phù hợp. Việc chọn nguồn B vẫn đang để mở.

## 3. Bạn tự kiểm tra trên Colab hoặc local

1. Tải CSV từ trang nguồn B, đọc phần mô tả, giấy phép và từ điển dữ liệu. Lưu file gốc riêng, ví dụ `data/nguon_b.csv`. Ghi ngày tải thực tế.
2. Mở `kiem_tra_hai_nguon_du_lieu.ipynb`. Trên Colab, upload notebook cùng hai CSV qua thanh **Files**. Trên local, mở notebook từ thư mục dự án.
3. Sửa `PATH_A`, `PATH_B`; Colab có thể dùng `/content/Crop_recommendation.csv` và `/content/nguon_b.csv`.
4. Nếu tên cột khác, điền `RENAME_B` sau khi xác minh ý nghĩa. Không coi soil moisture là relative humidity, không coi nhiệt độ đất là nhiệt độ không khí. Không đổi đơn vị N/P/K khi thiếu thông tin chuyển đổi.
5. Chạy **mục 1–3** trong notebook phụ. Đọc bảng phân bố, nhãn, số dòng trùng, số dòng mới và xung đột nhãn. Đây là thao tác kiểm tra dữ liệu, không huấn luyện.
6. Nếu thiếu một trong bảy đặc trưng, không tự tạo cột bằng số ngẫu nhiên hoặc điền 0 để gộp. Nếu có cây ngoài 22 nhãn, cần thống nhất lại phạm vi trước, không tự gom các loại đậu khác nhau thành một nhãn.

## 4. Khi nào có thể gộp?

Chỉ nối thêm dòng khi đã xác minh:

- Bảy đặc trưng có cùng ý nghĩa, thang đo, đơn vị và khoảng thời gian tổng hợp.
- Nhãn cùng một định nghĩa và cùng phạm vi đề tài.
- Nguồn B thực sự bổ sung quan sát, không chỉ đổi tên, sắp xếp lại hoặc làm tròn nguồn A.
- Không còn xung đột cùng đặc trưng nhưng khác nhãn chưa được giải quyết.
- Có hồ sơ nguồn, giấy phép và thông tin quan hệ giữa hai nguồn.

Với hai bảng có cùng cấu trúc, phép nối là `pd.concat([a, b])` theo chiều dọc. Không ghép ngang theo chỉ số dòng vì hai hàng cùng số thứ tự chưa chắc là cùng một mẫu đất.

Mục 4 của notebook chỉ xuất file khi bạn tự bật `EXPORT_MERGED` và điền căn cứ xác minh. Quy trình này chưa hỗ trợ trộn dữ liệu sinh: giữ các cờ mặc định nếu nguồn B là CTGAN hoặc chưa rõ nguồn gốc. Nếu mọi dòng B đã có trong A, kết luận kiểm tra là **không có mẫu phân biệt bổ sung**, không báo cáo gộp được 4.400 mẫu mới.

Nếu đủ điều kiện, đầu ra nằm riêng trong `data/merged_candidate/`:

- `Crop_recommendation_merged.csv`: bảy đầu vào và nhãn.
- `row_sources.csv`: dấu vết nguồn từng dòng; không đưa vào đặc trưng học.
- `sources.json`: hồ sơ và số lượng trước/sau.

## 5. Sau khi có dữ liệu mới

Chưa thay notebook chính ngay. Cần khảo sát lại dữ liệu và thiết kế chia tập phù hợp, xử lý trùng/gần trùng trước khi chia. Các quan sát cùng nhóm không được đi vào cả train và test.

Giữ `outputs/lan_01`. Khi quyết định chạy thí nghiệm mới, đổi `DATA_PATH` sang file mới và dùng `RUN_NAME` mới, ví dụ `lan_02_hai_nguon`. Các lần thử với dữ liệu khác nhau phải báo cáo riêng; không dùng điểm của tập test đã xem để lựa chọn cách gộp. Mọi thống kê ở Chương 2 cần cập nhật theo bộ dữ liệu thực tế được chốt.

Nếu hai nguồn không thể gộp, cần tìm nguồn khác phù hợp hơn; hai thí nghiệm riêng chưa đáp ứng yêu cầu gộp của giảng viên. Chỉ cập nhật báo cáo thành đã sử dụng hai nguồn sau khi hoàn thành việc kiểm tra và tích hợp thực tế.
