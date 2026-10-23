# Khuyến nghị loại cây trồng phù hợp dựa trên đặc điểm thổ nhưỡng và khí hậu

Bài thực hành môn **Học máy**, dùng Python và scikit-learn. KNN là thuật toán chính; Gaussian Naive Bayes, Decision Tree và Random Forest là các mô hình so sánh. Làm bài chủ yếu trong một notebook, có demo top-3 ngay trong notebook hoặc bằng chương trình dòng lệnh.

**Trạng thái bàn giao:** đã chuẩn bị mã nguồn và hướng dẫn tiếng Việt. Chưa tải CSV, chưa cài thư viện, chưa thực thi ô notebook, huấn luyện, benchmark, kiểm thử hay demo. Tất cả output notebook đang trống. Các phiên bản thư viện được ghi để tái lập; chưa xác nhận bằng thực thi trên máy bạn.

## Chạy trên máy hay Google Colab?

Bạn có thể chọn một trong hai môi trường. Nếu dùng **Google Colab**, làm theo [HUONG_DAN_COLAB.md](HUONG_DAN_COLAB.md): tải notebook và các file phụ lên, tự chạy từng bước, tải mô hình/kết quả về máy rồi chạy demo. Các phần cài môi trường bên dưới dành cho **local Windows**.

**Định dạng hiện tại:** lưu cả scaler và KNN vào .joblib; dự án chưa có phần xuất ONNX. Dùng Colab không bắt buộc phải dùng ONNX.

## 1. Các file bạn cần mở

| File | Dùng để làm gì? |
|---|---|
| `crop-recommendation-knn.ipynb` | Lý thuyết, kiểm tra dữ liệu, EDA, thí nghiệm, đánh giá, lưu mô hình và demo |
| `demo.py` | Nhập 7 giá trị trong terminal, in top-3 từ mô hình đã lưu; không huấn luyện |
| `requirements.txt` | Thư viện cho môi trường Học máy riêng |
| `data/README.md` | Nguồn và cách tải dữ liệu, lưu ý về đơn vị |
| `BAO_CAO_GOI_Y.md` | Khung báo cáo và câu hỏi ôn tập; bạn tự điền kết quả thật |

Đường dẫn dự án: `D:\Crop-Recommendation-ML`. Mở **đúng thư mục này** bằng File → Open Folder trong VS Code. CPU là đủ cho quy mô bài; không cần TensorFlow, PyTorch hoặc GPU.

## 2. Bạn tự tạo môi trường

Mở PowerShell trong thư mục dự án. Dùng Python **3.12**. Chạy từng lệnh, chờ lệnh trước hoàn thành:

```powershell
cd D:\Crop-Recommendation-ML
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lệnh đầu xác định thư mục; lệnh thứ hai kiểm tra Python; lệnh thứ ba tạo môi trường riêng; hai lệnh cuối cài công cụ và thư viện. Nếu không có Python 3.12, cài Python 3.12 trước rồi thực hiện lại. Không chạy các lệnh này trong `.venv` của dự án Học sâu.

Dùng đường dẫn Python trực tiếp nên không cần `Activate.ps1` hoặc thay ExecutionPolicy.

Trong VS Code, cài extension Python và Jupyter nếu chưa có. Mở notebook → **Select Kernel → Python Environments → `.venv`**. Nếu chưa thấy, chọn interpreter `D:\Crop-Recommendation-ML\.venv\Scripts\python.exe` rồi mở lại danh sách kernel.

Nếu muốn dùng JupyterLab thay VS Code:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## 3. Bạn tự tải dữ liệu

Mở [Crop Recommendation Dataset — Atharva Ingle trên Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), Download và giải nén. Chép `Crop_recommendation.csv` vào:

```text
D:\Crop-Recommendation-ML\data\Crop_recommendation.csv
```

CSV cần có `N, P, K, temperature, humidity, ph, rainfall, label`, với 22 nhãn cây. Notebook kiểm tra số lượng thực tế, dữ liệu thiếu, miền giá trị và dòng trùng. Đọc thêm `data/README.md` để hiểu thang đo; không tự gán đơn vị cho N/P/K hoặc khoảng thời gian cho rainfall khi nguồn chưa rõ.

## 4. Lộ trình chạy notebook

**Lần đầu dùng Shift+Enter từng ô; chưa chọn Run All.** Đọc ô Markdown ngay phía trên mã để biết ô đó chuẩn bị, huấn luyện hay kiểm tra.

| Mục | Bạn thực hiện | Kết quả cần hiểu |
|---|---|---|
| 1 | Đọc lý thuyết | K, ba khoảng cách, trọng số, scaling, KD-Tree/Ball-Tree |
| 2 | Import, cấu hình, ví dụ tính khoảng cách | Đúng Python `.venv`; khoảng cách tính tay khớp |
| 3 | Đọc/kiểm tra CSV | Số dòng, 22 nhãn, dòng trùng/thiếu theo file thật |
| 4 | Tách 80% train, 20% test; tạo 5 fold train | Hai tập không chung dòng; phân tầng theo nhãn |
| 5 | Xem biểu đồ từ train | Phân bố, khác thang đo, tương quan, phân bố theo cây |
| 6 | **Chạy thử một KNN có fit** | Tự đếm 5 phiếu và đối chiếu xác suất; chưa khảo sát hàng loạt |
| 7 | **Chạy khảo sát KNN** | 162 cấu hình × 5 fold, chọn theo F1 macro |
| 8 | Đọc bảng và đường cong | Ảnh hưởng riêng K/scaler/metric/weights với các yếu tố khác cố định |
| 9 | **Chạy 3 mô hình so sánh**, từng ô | Cùng fold và tiêu chí chọn, lưới tham số được nêu rõ |
| 10 | **Chạy benchmark** | Trung vị thời gian brute/KD-Tree/Ball-Tree; đối chiếu kết quả |
| 11 | Chốt bằng CV, **đánh giá test cuối** | Accuracy, F1 macro, 4 ma trận nhầm lẫn, mẫu KNN sai |
| 12 | Lưu pipeline KNN | File chứa cả scaler và mô hình, chưa fit thêm |
| 13 | **Tự kiểm thử chương trình** | Xác suất trước/sau nạp khớp; đầu vào sai được chặn |
| 14 | Thay 7 giá trị để demo | Top-3 nhãn và xác suất; chạy lại ô này không train |
| 15 | Viết nhận xét và báo cáo | Kết luận từ kết quả đã chạy, ghi nguồn và giới hạn |

**Buổi đầu nên dừng sau mục 6** để hiểu dữ liệu và cách bỏ phiếu. Sau đó bạn mới chủ động chạy mục 7–10. Mình chưa có số Accuracy/F1 nào để bạn điền vào báo cáo.

KNN không có 30 epoch như CNN. `fit` lưu dữ liệu và có thể xây chỉ mục; phần tìm láng giềng diễn ra khi dự đoán. 810 lượt fit CV là 162 cấu hình được thử trên 5 fold, không phải 810 epoch.

## 5. Thiết kế thí nghiệm đã chuẩn bị

- K: `1, 3, 5, 7, 9, 11, 15, 21, 31`.
- Khoảng cách: Euclidean, Manhattan, Minkowski với `p=3`; p=1/p=2 là hai khoảng cách đã có nên không lặp cấu hình.
- Trọng số: `uniform`, `distance`.
- Scaling: không chuẩn hóa, StandardScaler, MinMaxScaler; đặt trong Pipeline để học thống kê **chỉ từ train của từng fold**.
- CV: StratifiedKFold, 5 fold, shuffle, seed=42; dùng cùng phân hoạch cho mọi mô hình.
- Gaussian NB: 3 giá trị `var_smoothing`; Decision Tree: 6 cấu hình; Random Forest: 4 cấu hình, mỗi rừng 200 cây.
- Chọn theo CV F1 macro; báo cả CV Accuracy và F1 macro, trung bình ± độ lệch chuẩn. Test 20% dùng sau khi chốt tham số.
- Benchmark: 3 cách tìm láng giềng, 5 lượt lặp, cùng dữ liệu/scaler/K/metric/weights, lấy trung vị.

Điểm CV tốt nhất được dùng chọn cấu hình nên có thể lạc quan; báo tách biệt với điểm holdout test. Đây không phải nested CV. Độ lệch chuẩn giữa fold không phải khoảng tin cậy. [Tài liệu về lựa chọn mô hình và CV](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Trọng tâm vẫn là KNN, vì vậy demo dùng KNN kể cả khi mô hình khác có điểm cao hơn. Báo cáo ghi trung thực kết quả so sánh. Không chọn lại mô hình hoặc sửa K sau khi nhìn test để làm đẹp kết quả.

## 6. Kết quả sẽ được tạo khi chính bạn chạy

Tất cả nằm trong `outputs/lan_01/`, hoặc tên `RUN_NAME` bạn đặt:

- `data_audit.json`, `config.json`, `split.json`: nguồn, kiểm tra dữ liệu, cấu hình, dòng train/test/fold.
- Các hình EDA, `knn_factors.png`: bằng chứng cho phần phân tích.
- `knn_cv.csv`, `knn_selected_folds.csv`, `*_cv_raw.csv`: điểm khảo sát.
- `neighbors_benchmark.csv` và bản raw: thời gian và đối chiếu cách tìm kiếm.
- `cv_summary.csv`, `selection.json`: lựa chọn trước khi xem test.
- `comparison.csv`, `*_classification_report.csv`, `*_confusion_matrix.png`: đánh giá.
- `test_predictions.csv`, `knn_errors.csv`: dự đoán từng dòng và mẫu sai.
- `knn_crop.joblib`, `model_metadata.json`: pipeline dùng cho demo và thông tin đi kèm.
- `*_search.joblib`: lưu kết quả GridSearch để có thể đọc lại.

Các file này **chưa tồn tại khi bàn giao**. Lưu notebook sau mỗi buổi để giữ output. Khi sửa lưới hoặc code thí nghiệm, tăng `experiment_revision` trong cấu hình và đổi `RUN_NAME` trước khi chạy lại. Xem test nhiều lần để thay thiết kế sẽ làm mất ý nghĩa của test độc lập.

## 7. Demo sau khi bạn đã chạy đến mục 13

Cách đơn giản nhất: dùng ngay **mục 14** trong notebook, sửa dict `new_soil`, chạy lại ô đó. Sau Restart Kernel vẫn dùng được riêng ô này nếu file mô hình đã tồn tại và thư mục làm việc đúng.

Hoặc chạy trong terminal tại gốc dự án:

```powershell
.\.venv\Scripts\python.exe demo.py --model outputs/lan_01/knn_crop.joblib
```

Chương trình lần lượt hỏi N, P, K, nhiệt độ, độ ẩm, pH, lượng mưa. Nhập số và nhấn Enter, dùng dấu chấm cho phần thập phân. Nó chỉ nạp mô hình để dự đoán. Chỉ nạp file joblib bạn tự tạo từ notebook.

Xác suất KNN là tỷ lệ phiếu (có hoặc không trọng số), **không phải xác suất canh tác thành công**. Nếu chỉ một lớp nhận phiếu, nó có thể hiện 100%, hai hàng còn lại 0%; các hàng 0% được đánh dấu thiếu căn cứ. Đồng điểm được ghi rõ, và không chuẩn hóa riêng top-3 về tổng 100%. [Tài liệu KNN](https://scikit-learn.org/1.7/modules/neighbors.html).

## 8. Cách tự kiểm tra và xử lý lỗi

- **Thiếu thư viện:** kiểm tra Python được in ở mục 2; thường do chọn nhầm kernel. Cài requirements vào đúng `.venv`.
- **Không tìm thấy CSV:** xem `DATA_PATH`; file phải được giải nén, không nằm thêm một tầng thư mục trong `data/`.
- **Không import được `demo`:** mở đúng thư mục dự án, kiểm tra `Path.cwd()`; `demo.py` phải ở cạnh notebook.
- **NameError:** đã bỏ qua ô định nghĩa hoặc restart kernel. Chạy lại phần chuẩn bị theo thứ tự, không tự chạy lại toàn bộ tìm kiếm chỉ để xem demo.
- **GridSearch mất thời gian:** xem dòng “Fitting 5 folds…” và số cấu hình. Chờ ô chạy xong; không coi nhiều lượt fit là nhiều epoch. `N_JOBS=1` để hạn chế chiếm CPU.
- **Đã có `comparison.csv`:** notebook chặn tuning tiếp trong cùng lần đã xem test. Đọc bảng CSV để xem lại kết quả; không xóa file chỉ để lựa chọn theo test.
- **Kiểm thử mục 6/13 lỗi:** đọc assertion/traceback, ghi lại ô lỗi và phiên bản mục 2. Không bỏ assert để tiếp tục.
- **Chưa có `knn_crop.joblib`:** chưa hoàn thành mục 12. Demo không tự train thay bạn.
- **Top-3 chứa 0%:** đây có thể là kết quả đúng của KNN, không tự thêm xác suất giả.

Nếu kernel bị tắt giữa chừng, các `*_search.joblib` có thể giúp bạn đọc lại tìm kiếm đã hoàn thành. Sau khi chạy lại phần chuẩn bị mục 2–4 và ô định nghĩa đầu mục 7 (không chạy ô tìm kiếm), có thể thêm ô sau để khôi phục những mô hình đã có:

```python
for name in ['KNN', 'GaussianNB', 'DecisionTree', 'RandomForest']:
    saved = OUT / f'{name}_search.joblib'
    if saved.is_file():
        searches[name] = joblib.load(saved)
if 'KNN' in searches:
    knn_search = searches['KNN']
```

Giữ nguyên file dữ liệu, cấu hình, phiên bản và mã nguồn khi khôi phục. Nếu đã có `comparison.csv`, chỉ đọc bảng hoặc dùng demo; không cần đánh giá lại test. Nếu chưa kịp lưu mô hình ở mục 12, chạy ô chốt CV đầu mục 11 để khôi phục `selection`, bỏ qua ô test, rồi chạy mục 12–13.

## 9. Hoàn thiện bài nộp

Đọc `BAO_CAO_GOI_Y.md`, điền số liệu thật và ít nhất một nhận xét cho mỗi nhóm thí nghiệm. Hiểu vì sao tách test, vì sao scaling nằm trong Pipeline, xác suất KNN được tính thế nào. Trích nguồn dữ liệu, tài liệu và ghi nhận hỗ trợ AI theo quy định của giảng viên.

Dữ liệu gán một nhãn cho mỗi mẫu; top-3 là xếp hạng nhãn dự đoán, chưa chứng minh ba cây đều phù hợp ngoài thực địa. Kết quả tốt trên bộ dữ liệu này chưa đủ để kết luận dùng được cho đất và khí hậu Việt Nam.

Nguồn chính: [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), [KNN](https://scikit-learn.org/1.7/modules/neighbors.html), [Pipeline và rò rỉ dữ liệu](https://scikit-learn.org/1.7/common_pitfalls.html), [GridSearchCV](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html), [StratifiedKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedKFold.html).
