# Dữ liệu

Thư mục đã có sẵn hai tệp dùng trong dự án, không cần tải thêm:

| Tệp | Dataset | Số dòng | Số lớp | SHA-256 |
|---|---|---|---|---|
| `crop_data1.csv` | A | 2.200 | 22 | `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` |
| `crop_data2.csv` | B | 1.796 (1.693 sau loại 103 dòng trùng) | 19 | `20692cec87e697041ab37693a4216e1891a05f181eb10d7b34ccb10212a71b1a` |

Nguồn: [KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning, thư mục Datasets](https://github.com/KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning/tree/main/Datasets), giấy phép MIT của kho, tải ngày 27/10/2026, đối chiếu SHA-256 ngày 08/11/2026. Mô tả bộ A được đối chiếu thêm với [Crop Recommendation Dataset trên Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). Hồ sơ thu thập và gán nhãn riêng của bộ B chưa rõ.

Notebook kiểm tra SHA-256 khi tổng hợp kết quả; nếu tệp bị thay đổi, các ô đối chiếu sẽ báo lỗi. Không tự tạo hoặc sửa dữ liệu.

Cả hai tệp có đúng 8 cột:

```text
N,P,K,temperature,humidity,ph,rainfall,label
```

| Cột | Ý nghĩa và lưu ý |
|---|---|
| N, P, K | Tỷ lệ hàm lượng nitơ, phospho, kali; giữ đúng thang đo của nguồn, chưa rõ đơn vị |
| temperature | Nhiệt độ, độ C |
| humidity | Độ ẩm tương đối, %; không mặc định là độ ẩm đất |
| ph | pH đất, thang 0–14 |
| rainfall | Lượng mưa, mm; chưa rõ kỳ tích lũy |
| label | Nhãn cây trồng, giữ nguyên tên tiếng Anh trong từng tệp |

Hai bộ có nhiều đầu vào chồng lặp và một số nhãn khác nhau, nên được thực nghiệm riêng, không gộp. Dữ liệu không có địa điểm, thời gian hay mã thửa ruộng, không suy luận rằng nó đại diện cho điều kiện Việt Nam.
