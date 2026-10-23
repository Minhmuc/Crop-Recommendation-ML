# Chuẩn bị dữ liệu

1. Mở [Crop Recommendation Dataset — Atharva Ingle, Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
2. Tải bằng nút Download (đăng nhập nếu Kaggle yêu cầu), giải nén trên máy.
3. Chép **Crop_recommendation.csv** vào chính thư mục này.
4. Đường dẫn cần có: `D:\Crop-Recommendation-ML\data\Crop_recommendation.csv`.

Notebook không tự tải dữ liệu. CSV cần có đúng 8 cột:

```text
N,P,K,temperature,humidity,ph,rainfall,label
```

Đề bài yêu cầu 22 nhãn cây trồng. Bản dữ liệu thường được dùng có 2.200 mẫu, 100 mẫu/nhãn; notebook sẽ đếm lại file bạn thực sự tải. Không tự tạo dữ liệu để thay file thiếu.

| Cột | Ý nghĩa và lưu ý |
|---|---|
| N, P, K | Giá trị các thành phần dinh dưỡng trong dữ liệu nguồn; giữ đúng thang đo của nguồn. Không tự gán đơn vị mg/kg hoặc kg/ha khi chưa xác nhận. |
| temperature | Nhiệt độ, độ C |
| humidity | Độ ẩm tương đối, %; không mặc định đây là độ ẩm đất |
| ph | pH đất, không có đơn vị |
| rainfall | Lượng mưa, mm; cần xác minh khoảng thời gian tích lũy trước khi nhập số liệu thực địa |
| label | Nhãn cây trồng, dùng nguyên tên tiếng Anh trong CSV |

Ghi lại ngày tải, phiên bản và điều khoản sử dụng hiển thị trên Kaggle vào báo cáo. Không suy luận rằng dữ liệu đại diện cho khí hậu/thổ nhưỡng Việt Nam. File gốc chưa có thông tin địa điểm, thời gian hoặc mã thửa ruộng thì không thể kiểm tra sự độc lập theo những nhóm đó.
