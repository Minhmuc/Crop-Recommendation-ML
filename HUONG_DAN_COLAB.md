# Chạy bài Học máy trên Google Colab rồi tải mô hình về làm demo

Hướng dẫn này dành cho `crop-recommendation-knn.ipynb`. Bạn tự chạy từng ô; chưa có bước nào được thực hiện thay bạn. Mã hiện lưu **pipeline KNN + scaler thành `.joblib`**, chưa xuất ONNX. Colab là nơi thực hiện tính toán, không quyết định định dạng mô hình.

## 1. Chuẩn bị và mở notebook

Trên máy, chuẩn bị 4 file:

- `crop-recommendation-knn.ipynb`.
- `requirements.txt`.
- `demo.py` (notebook có import các hàm từ file này).
- `Crop_recommendation.csv`: tự tải, giải nén từ [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** và tải file `.ipynb`. Lưu một bản vào Google Drive để giữ notebook và output. Dùng runtime CPU; mã scikit-learn của bài không cần GPU. Chưa chọn Run all.

Colab dùng máy ảo riêng; đường dẫn `D:\...` trên máy bạn không tồn tại ở đó. Runtime có thể bị thu hồi; dữ liệu trong `/content` không được lưu chỉ vì notebook đã lưu lên Drive. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 2. Thêm các ô chuẩn bị trước mục 2 của notebook

Chèn ba ô Code dưới đây trước ô import của mục 2, chạy lần lượt. Các ô này chỉ tải file/cài môi trường/đặt đường dẫn, chưa fit mô hình.

**Ô A — đưa ba file phụ lên runtime:**

```python
%cd /content
from google.colab import files
uploaded = files.upload()
# Chọn requirements.txt, demo.py và Crop_recommendation.csv từ máy.
# Nếu mới chọn một file, chạy lại ô này để tải các file còn thiếu.
```

**Ô B — cài thư viện trước khi import scikit-learn/numpy:**

```python
import sys
print('Python của Colab:', sys.version)
%pip install -r /content/requirements.txt
```

Requirements được chuẩn bị cho Python 3.12 trên local. Phiên bản Python của Colab có thể thay đổi. Nếu pip báo phiên bản không tương thích, dừng và gửi lỗi cùng dòng Python ở trên, không bỏ các ràng buộc tùy tiện. Nếu pip yêu cầu **Restart session/runtime**, thực hiện restart trước khi chạy tiếp. Sau restart thông thường, kiểm tra lại các file; nếu Colab cấp máy ảo mới, bạn phải tải/cài lại.

**Ô C — đặt CSV vào đúng thư mục và kiểm tra file:**

```python
%cd /content
from pathlib import Path
import shutil

root = Path('/content')
for name in ['requirements.txt', 'demo.py', 'Crop_recommendation.csv']:
    if not (root / name).is_file():
        raise FileNotFoundError(f'Tải file {name} lên runtime trước.')
(root / 'data').mkdir(exist_ok=True)
shutil.copy2(root / 'Crop_recommendation.csv', root / 'data' / 'Crop_recommendation.csv')
print('Đã chuẩn bị:', root / 'data' / 'Crop_recommendation.csv')
```

Không cần tạo `.venv` trên Colab. Trong ô cấu hình hiện có, `ROOT = Path.cwd()` sẽ là `/content`, `DATA_PATH` sẽ trỏ tới `/content/data/Crop_recommendation.csv`. Giữ `RUN_NAME = 'lan_01'` cho lần đầu. Python in ở mục 2 là Python của Colab, không phải đường dẫn `.venv` Windows.

## 3. Bạn tự chạy thí nghiệm

Theo đúng thứ tự trong notebook:

| Mục | Việc làm |
|---|---|
| 2–5 | Đọc và kiểm tra dữ liệu, chia tập, xem biểu đồ; chưa huấn luyện |
| 6 | Tự chạy thử một KNN và đếm phiếu của 5 láng giềng |
| 7–8 | Chạy khảo sát KNN, đọc ảnh hưởng của tham số |
| 9 | Chạy lần lượt Gaussian NB, Decision Tree, Random Forest |
| 10 | Đo tốc độ brute/KD-Tree/Ball-Tree |
| 11 | Chốt cấu hình bằng CV rồi đánh giá test |
| 12 | Lưu pipeline vào `outputs/lan_01/knn_crop.joblib` |
| 13 | Tự kiểm thử lưu/nạp và đầu vào |
| 14 | Nhập mẫu mới và xem top-3 ngay trên Colab |

KNN không có epoch như CNN. Mục 7 thử nhiều cấu hình trên các fold; kết quả chỉ xuất hiện sau khi bạn thực hiện. Không sửa tham số dựa vào test để làm đẹp điểm số.

## 4. Tải mô hình và kết quả về máy

Sau khi mục 13 hoàn tất, thêm ô dưới cuối notebook và tự chạy. Ô này chỉ đóng gói/tải file, không huấn luyện:

```python
from pathlib import Path
from google.colab import files
import shutil
import sys
import json
from importlib.metadata import version

run_dir = Path('/content/outputs/lan_01')  # Sửa nếu bạn đổi RUN_NAME.
if not (run_dir / 'knn_crop.joblib').is_file():
    raise FileNotFoundError('Chưa có mô hình; hoàn thành mục 12–13 trước.')

# Lưu phiên bản thực tế để đối chiếu khi dùng trên máy cá nhân.
packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'threadpoolctl']
(run_dir / 'inference-requirements.txt').write_text(
    '\n'.join(f'{name}=={version(name)}' for name in packages) + '\n', encoding='utf-8'
)
(run_dir / 'python_runtime.json').write_text(
    json.dumps({'python': sys.version}, indent=2), encoding='utf-8'
)
archive = shutil.make_archive(
    '/content/crop_ml_results', 'zip',
    root_dir='/content', base_dir=f'outputs/{run_dir.name}',
)
files.download(archive)
```

Bạn nhận `crop_ml_results.zip`, bên trong có `outputs/lan_01/` gồm mô hình, metadata, hình, bảng điểm và kết quả tìm kiếm. **Tải thêm notebook đã chạy** bằng File → Download → Download .ipynb để giữ quá trình thực nghiệm. ZIP kết quả ở trên không chứa notebook.

Nếu dừng trước mục 12, có thể tải riêng những kết quả đã có bằng bảng Files ở bên trái Colab. Không coi việc lưu notebook là đã sao lưu file mô hình.

## 5. Dùng mô hình Colab cho demo trên máy

Giải nén ZIP vào `D:\Crop-Recommendation-ML`. Đường dẫn cuối phải là:

```text
D:\Crop-Recommendation-ML\outputs\lan_01\knn_crop.joblib
D:\Crop-Recommendation-ML\outputs\lan_01\model_metadata.json
```

Nếu local đã có `outputs/lan_01`, giải nén sang một thư mục riêng để giữ kết quả cũ; truyền đường dẫn mới qua `--model`. Không để thêm một tầng `crop_ml_results/` mà vẫn dùng đường dẫn cũ.

Tạo môi trường local theo README nếu chưa có. Sau đó trong PowerShell:

```powershell
cd D:\Crop-Recommendation-ML
.\.venv\Scripts\python.exe -m pip install -r outputs\lan_01\inference-requirements.txt
.\.venv\Scripts\python.exe demo.py --model outputs\lan_01\knn_crop.joblib
```

Dùng cùng các phiên bản thư viện đã huấn luyện để nạp `.joblib`; đối chiếu cả `python_runtime.json` nếu gặp lỗi tương thích. Không chép toàn bộ `pip freeze` của Colab vào Windows vì có thể chứa gói dành riêng cho môi trường Colab. [Tài liệu lưu/nạp scikit-learn](https://scikit-learn.org/1.7/model_persistence.html).

Chương trình hỏi 7 giá trị rồi in top-3. Để tự đối chiếu, nhập đúng mẫu bạn đã dùng ở mục 14 trên Colab và so nhãn/xác suất giữa hai nơi. Demo chỉ dự đoán từ mô hình đã lưu, không cần chạy lại GridSearch trên máy.

## 6. ONNX có cần không?

Hiện dự án dùng `.joblib`, chưa có mã chuyển ONNX. Đây là lựa chọn định dạng của dự án, không phải hạn chế của môn Học máy hay của Colab. Mục tiêu hiện tại là demo Python dùng lại pipeline scikit-learn.

Nếu bổ sung ONNX, cần chuyển cả scaler và KNN, xác minh khả năng chuyển cấu hình được chọn (khoảng cách/trọng số), kiểm tra thứ tự nhãn và sai lệch xác suất với scikit-learn, rồi sửa demo sang ONNX Runtime. Chỉ đổi đuôi `.joblib` thành `.onnx` không phải chuyển đổi mô hình. Chưa có kết quả xuất/kiểm tra ONNX cho dự án này.
