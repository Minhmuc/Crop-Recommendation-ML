# Khuyến nghị cây trồng bằng KNN – Nhóm 08, học phần Học máy (ICTU)

Đây là **tài liệu duy nhất cần đọc** để mọi thành viên hiểu dự án như nhau, tự chạy được notebook và demo, trình bày và bảo vệ được cả phần của mình lẫn phần của bạn cùng nhóm.

- **Thành viên:** Nguyễn Quang Minh, Nông Hữu Thịnh
- **Giáo viên hướng dẫn:** TS. Nguyễn Tuấn Anh
- **Đề tài:** Khuyến nghị loại cây trồng phù hợp dựa trên đặc điểm thổ nhưỡng và khí hậu
- **Sản phẩm nộp cho giảng viên:** báo cáo **`nhom08m.docx`** (60 trang, kèm `nhom08m.pdf`) – bản chốt ngày 23/11/2026. Notebook và demo được trình diễn trực tiếp ở buổi bảo vệ.

---

## Đọc file này thế nào?

| Bạn cần… | Đọc mục |
|---|---|
| Hiểu nhanh toàn bộ dự án trong 5 phút | 1, 8, 9 |
| Hiểu dữ liệu và cách làm | 2 – 7 |
| Cài đặt, chạy notebook, chạy demo | 11, 12 |
| Nắm lý thuyết để vấn đáp | 14 |
| Chuẩn bị buổi bảo vệ | 15, 16, 17, 20 |
| Tra cứu file kết quả | 10 |

## Mục lục

1. [Tóm tắt trong một phút](#1-tóm-tắt-trong-một-phút)
2. [Bài toán](#2-bài-toán)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Dữ liệu](#4-dữ-liệu)
5. [Quy trình tổng thể và notebook](#5-quy-trình-tổng-thể-và-notebook)
6. [Thiết kế thực nghiệm](#6-thiết-kế-thực-nghiệm)
7. [Các mô hình và lưới tham số](#7-các-mô-hình-và-lưới-tham-số)
8. [Kết quả](#8-kết-quả)
9. [Những con số dễ hiểu sai](#9-những-con-số-dễ-hiểu-sai)
10. [Thư mục outputs](#10-thư-mục-outputs)
11. [Cài đặt và chạy lại](#11-cài-đặt-và-chạy-lại)
12. [Ứng dụng demo](#12-ứng-dụng-demo)
13. [Báo cáo](#13-báo-cáo)
14. [Kiến thức nền cần nắm](#14-kiến-thức-nền-cần-nắm)
15. [Hướng dẫn bảo vệ](#15-hướng-dẫn-bảo-vệ)
16. [Câu hỏi vấn đáp thường gặp](#16-câu-hỏi-vấn-đáp-thường-gặp)
17. [Luyện thực hành tại chỗ](#17-luyện-thực-hành-tại-chỗ)
18. [Giới hạn và hướng phát triển](#18-giới-hạn-và-hướng-phát-triển)
19. [Phân công](#19-phân-công)
20. [Checklist trước buổi bảo vệ](#20-checklist-trước-buổi-bảo-vệ)
21. [Thuật ngữ](#21-thuật-ngữ)
22. [Sử dụng AI](#22-sử-dụng-ai)

---

## 1. Tóm tắt trong một phút

- **Đầu vào:** 7 chỉ số `N, P, K, temperature, humidity, ph, rainfall`. **Đầu ra:** tên cây trồng (`label`). Đây là bài toán **phân loại đa lớp có giám sát**.
- **Hai bộ dữ liệu, thực nghiệm riêng, không gộp:** A có 2.200 mẫu, 22 lớp. B có 1.796 dòng, còn 1.693 mẫu sau khi loại 103 dòng trùng, 19 lớp.
- **Vai trò các mô hình:** KNN là thuật toán chính. Gaussian Naive Bayes là **mô hình cơ sở (baseline)**. Decision Tree và Random Forest dùng để **so sánh**.
- **Cách đánh giá:** chia train/test 80/20 **theo nhóm đầu vào**, kiểm định chéo 5 fold `StratifiedGroupKFold`. Chọn cấu hình theo **F1 macro kiểm định chéo**; tập test chỉ dùng để đối chiếu sau cùng.
- **Cải tiến KNN:** dò 162 cấu hình, ablation 4 biến thể có chọn 5/7 đặc trưng bằng mutual information, learning curve, so sánh tốc độ brute/KD-Tree/Ball-Tree.
- **Kết quả:** KNN_final đạt F1 macro test **0,97714 (A)** và **0,99760 (B)**. **KNN không vượt Gaussian NB** (thấp hơn trên A, bằng trên test B); báo cáo giải thích nguyên nhân ở mục 3.4.6.
- **Tái lập:** notebook được chạy lại toàn bộ ngày 23/11/2026; mọi chỉ số CV và test trùng khớp lần trước.
- **Kết luận có kiểm định:** trên A, KNN_final thấp hơn Gaussian NB ở cả 5/5 fold và trên test (McNemar p ≈ 0,008); trên B hai mô hình không phân biệt được.
- **Sản phẩm:** notebook `crop.ipynb`, báo cáo `nhom08m.docx`, mô hình `knn_crop.joblib`, ứng dụng web Streamlit `app.py`.

**Câu trả lời một dòng khi bị hỏi "dự án làm gì?":**
> Dùng KNN gợi ý cây trồng từ 7 chỉ số đất và khí hậu. Nhóm đánh giá đúng phương pháp (chia theo nhóm, CV 5 fold, Pipeline) trên hai bộ dữ liệu, tối ưu KNN và chỉ ra vì sao KNN vẫn thua Gaussian NB.

---

## 2. Bài toán

| Mục | Nội dung |
|---|---|
| Loại bài toán | Phân loại đa lớp, học có giám sát |
| Đơn vị quan sát | Một bản ghi gồm 7 chỉ số đất/khí hậu và một nhãn cây trồng |
| Đầu vào | `N, P, K` (tỷ lệ dinh dưỡng đất, chưa rõ đơn vị), `temperature` (°C), `humidity` (%), `ph` (0–14), `rainfall` (mm) |
| Đầu ra | Một nhãn cây trồng, kèm top-3 nhãn có điểm phiếu cao nhất |
| Chỉ số chính | **F1 macro**: mỗi lớp có trọng số ngang nhau |
| Chỉ số phụ | Accuracy, precision/recall macro, ma trận nhầm lẫn, chỉ số theo lớp |
| Mô hình cơ sở | Gaussian Naive Bayes |

**Vì sao chọn F1 macro?** Bộ B có lớp rất ít mẫu (arecanut, groundnut, jackfruit, sugarcane chỉ có 5 mẫu test mỗi lớp). Accuracy vẫn có thể cao dù mô hình đoán sai gần hết một lớp nhỏ, còn F1 macro bị kéo xuống ngay.

**Phạm vi:** bài toán học máy trên dữ liệu công khai, chưa phải công cụ tư vấn canh tác thật. Dữ liệu không có địa điểm, thời gian hay đơn vị đo N, P, K.

---

## 3. Cấu trúc thư mục

### Sản phẩm chính

| File / thư mục | Nội dung |
|---|---|
| `nhom08m.docx`, `nhom08m.pdf` | **Báo cáo cuối – bản nộp** theo mẫu ICTU (60 trang) |
| `crop.ipynb` | **Notebook chính**: dữ liệu → khảo sát → huấn luyện → cải tiến → đánh giá → lưu mô hình → demo top-3 |
| `app.py` | Giao diện web Streamlit |
| `demo_core.py` | Phần xử lý của demo: nạp mô hình, kiểm tra đầu vào, top-3, láng giềng, sinh mẫu |
| `.streamlit/config.toml` | Màu và font của demo |
| `data/crop_data1.csv`, `data/crop_data2.csv` | Dữ liệu A và B ([mô tả chi tiết](data/README.md)) |
| `outputs/crop_a_02`, `outputs/crop_b_02` | **Kết quả chuẩn dùng trong báo cáo**: CSV, hình, mô hình |
| `requirements.txt` | Thư viện và phiên bản |
| `docs/meta-prompt/` | Bản yêu cầu cho giai đoạn hoàn thiện (`hoan_thien_du_an.md`) và giao diện demo (`demo_phong_cach_nong_trai.md`) |
| `docs/hinh_bao_cao/` | Ảnh chụp demo (Hình 3.14, 3.15) và Hình 3.10 vẽ lại từ `knn_cv.csv` (`knn_factors_A_v2.png`) |
| `.claude/skills/lanh-dao-sub-agent/` | Skill điều phối sub agent rà soát độc lập (đã dùng để rà soát báo cáo) |
| `tieuchicham.docx` | Tiêu chí chấm của học phần |
| `nhom08/Template_DA_KLTN_ICTU/` | Mẫu trình bày ICTU |

### Tài liệu cũ hoặc phụ (không dùng khi trình bày)

| File | Ghi chú |
|---|---|
| `Crop recommendation final.ipynb`, `crop recommendation model` | Notebook tiếng Anh và mô hình của giai đoạn đầu, đã được `crop.ipynb` thay thế |
| `nhom 08.docx`, `nhom 08.pdf` | Bản báo cáo trước lần rà soát bằng sub agent (58 trang), giữ để đối chiếu |
| `dangbaocao.docx`, `dangbaocao1.docx`, `mau.docx`, `mauBC.docx` | Bản nháp và mẫu báo cáo cũ |
| `HUONG_DAN_HAI_NGUON_DU_LIEU.md`, `pom.md`, `promptbangiao.md` | Ghi chép quá trình làm; một số đoạn đã cũ |
| `backups/` | Bản sao lưu báo cáo trước mỗi lần sửa lớn |
| `crop_ml_crop_a_02.zip`, `crop_ml_crop_b_02.zip` | Gói nén thư mục kết quả do notebook tạo ở mục 20 |

> `.gitignore` hiện **không** loại trừ `data/*.csv`, `outputs/` hay hai file `crop_ml_crop_*_02.zip` (chỉ loại `backups/`, `.venv/`, `data/*.zip` và vài file nháp). Sau khi commit, bản clone từ Git sẽ có sẵn dữ liệu và kết quả chuẩn. Nếu dùng bản sao không có `outputs/`, giải nén hai file `crop_ml_crop_*_02.zip` vào thư mục dự án (bên trong đã có đường dẫn `outputs/crop_*_02`).

---

## 4. Dữ liệu

### 4.1. Nguồn

Cả hai CSV được tải từ [kho KRUTHIKTR, thư mục Datasets](https://github.com/KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning/tree/main/Datasets) (giấy phép MIT của kho). SHA-256 được ghi trong `config.json` và notebook kiểm tra lại mỗi lần chạy. Bộ A trùng mô tả với [Crop Recommendation Dataset trên Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). Quy trình thu thập và gán nhãn của bộ B chưa rõ.

### 4.2. Kích thước và chất lượng

| Bộ | File | Dòng gốc | Ô thiếu | Dòng không hợp lệ | Trùng hoàn toàn | Sau loại trùng | Số lớp | Train / Test |
|---|---|---|---|---|---|---|---|---|
| A | `crop_data1.csv` | 2.200 | 0 | 0 | 0 | 2.200 | 22 | 1.760 / 440 |
| B | `crop_data2.csv` | 1.796 | 0 | 0 | 103 | 1.693 | 19 | 1.354 / 339 |

"Dòng không hợp lệ" gồm: giá trị không hữu hạn, N/P/K/rainfall âm, humidity ngoài [0, 100], ph ngoài [0, 14] (hàm `invalid_rows`, notebook mục 3).

- **A** cân bằng: 100 mẫu mỗi lớp (80 train, 20 test).
- **B** lệch: rice 139, soyabeans 130, maize 119, banana 106, mười một lớp 100, jackfruit/sugarcane/groundnut 25, arecanut 24.

### 4.3. Nhãn

- **A (22):** apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon.
- **B (19):** apple, arecanut, banana, beans, coffee, cotton, cowpeas, grapes, groundnut, groundnuts, jackfruit, maize, mango, orange, peas, rice, soyabeans, sugarcane, watermelon.

Nhãn được bỏ khoảng trắng và chuyển chữ thường (`Arecanut` → `arecanut`). **Không gộp `groundnut` với `groundnuts`**: groundnuts có cùng đầu vào với lớp mothbeans của A ở cả 100 nhóm, còn groundnut là 25 dòng chỉ có ở B.

### 4.4. Vì sao không gộp hai bộ?

Notebook mục 3 nối tạm A và B chỉ để kiểm tra:

- **1.494 nhóm đầu vào xuất hiện ở cả hai nguồn** (cùng 7 giá trị sau làm tròn 6 chữ số).
- **498 nhóm cùng đầu vào nhưng mang nhiều nhãn** (1.043 dòng).

- 1.494 nhóm chung tương ứng **88% số mẫu B** sau loại trùng; 498 nhóm khác nhãn đều thuộc **5 cặp nhãn đổi tên**: kidneybeans–beans, mungbean–cowpeas, mothbeans–groundnuts, chickpea–soyabeans, pigeonpeas–peas.
- B **không có** jute, lentil, blackgram – các lớp liên quan 8/10 lỗi của KNN trên test A.

Gộp lại thì cùng một đầu vào có hai đáp án, và cùng một bản ghi có thể nằm cả ở train lẫn test. Vì vậy hai bộ được thực nghiệm riêng. **Hệ quả:** B gần như một tập con của A với hệ nhãn khác; kết quả A và B không phải hai bằng chứng độc lập, và B "dễ" hơn một phần vì thiếu các cặp lớp hay nhầm (ví dụ một bản ghi maize nằm trong test của cả hai bộ và KNN đều đoán thành cotton).

### 4.5. Nhận xét khảo sát quan trọng

- **K** có một nhóm giá trị rất cao (khoảng 200) thuộc **apple và grapes**; IQR gắn cờ 160 giá trị K (9,09% train A, 11,82% train B) nhưng **được giữ** vì là đặc điểm thật của lớp.
- **P và K** tương quan Pearson 0,736 (A) và 0,803 (B); vẫn giữ cả hai để mô hình tự đánh giá.
- **rainfall** có đuôi lệch phải (trung bình > trung vị); tương quan thấp với các đầu vào khác nên mang thông tin riêng.
- **pH** tập trung 6–7, miền giá trị hẹp so với K và rainfall → ảnh hưởng lớn tới cách chuẩn hóa cho KNN.
- **maize và mungbean** chồng lấn trên K; phải kết hợp nhiều đặc trưng mới phân biệt được.

---

## 5. Quy trình tổng thể và notebook

```text
Đọc A, B ─► Chuẩn hóa cột/nhãn ─► Kiểm tra chất lượng & chồng lặp ─► Chọn 1 bộ (DATA_MODE), loại trùng
   ─► Chia train/test theo nhóm (80/20) ─► Tạo 5 fold trên train
   ─► Khảo sát dữ liệu (chỉ trên train)
   ─► KNN minh họa ─► GridSearch KNN (162 cấu hình) ─► GNB / DT / RF
   ─► Ablation 4 biến thể + mutual information ─► Learning curve ─► Benchmark tìm láng giềng
   ─► Chốt mô hình bằng CV (ghi selection.json) ─► MỞ TEST một lần ─► Phân tích lỗi
   ─► Lưu knn_crop.joblib ─► Nạp lại & kiểm tra ─► Demo top-3 ─► Đối chiếu A/B ─► Đóng gói
```

| Giai đoạn | Mục notebook | Nội dung |
|---|---|---|
| 1. Bài toán và dữ liệu | 1–3 | Phát biểu bài toán; cấu hình (`DATA_MODE`, `RUN_VERSION`, seed…); đọc A/B, chuẩn hóa, kiểm tra thiếu/miền giá trị/trùng/chồng lặp; chọn bộ và loại trùng |
| | 4 | Chia train/test theo nhóm, tạo 5 fold, lưu `split.json` |
| | 5 | Khảo sát **chỉ trên train**: thống kê, histogram, tương quan, boxplot, IQR, minh họa StandardScaler |
| 2. Xây dựng và so sánh | 6 | KNN minh họa: tự đếm phiếu 5 láng giềng và đối chiếu với `predict_proba` |
| | 7–8 | Pipeline + GridSearchCV 162 cấu hình KNN; đọc điểm CV, so sánh train và validation |
| | 9 | Gaussian NB, Decision Tree, Random Forest trên cùng các fold |
| 3. Cải tiến bằng CV | 10 | Ablation 4 biến thể và mutual information |
| | 11 | Learning curve; benchmark brute / KD-Tree / Ball-Tree |
| | 12 | Tổng hợp CV, chốt mô hình, ghi `selection.json` |
| 4. Đánh giá và sử dụng | 13–14 | Đánh giá test, ma trận nhầm lẫn, phân tích lỗi |
| | 15–17 | Lưu Pipeline, nạp lại, so khớp dự đoán, kiểm tra đầu vào sai, demo top-3 |
| 5. Tổng hợp | 18–21 | Đối chiếu hai bộ, bảng cho báo cáo, đóng gói zip, tài liệu tham khảo |

---

## 6. Thiết kế thực nghiệm

### 6.1. Tham số cố định

| Tham số | Giá trị |
|---|---|
| `SEED` | 42 |
| `TEST_SIZE` | 0,20 (tính theo nhóm) |
| `N_SPLITS` | 5 |
| `N_JOBS` | 1 (để kết quả lặp lại được) |
| `GROUP_DECIMALS` | 6 |
| `K_VALUES` | 1, 3, 5, 7, 9, 11, 15, 21, 31 |
| `SELECTED_FEATURE_COUNT` | 5 |
| `SCORING` | accuracy, f1_macro (refit theo f1_macro) |

### 6.2. Chia dữ liệu theo nhóm

1. Làm tròn 7 đặc trưng đến 6 chữ số; các dòng cùng giá trị thuộc **một nhóm** (`_group`).
2. `train_test_split` trên **danh sách nhóm**, phân tầng theo nhãn, 20% nhóm làm test.
3. Trên train, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` tạo 5 fold; không nhóm nào nằm cả ở phần học lẫn phần kiểm định.
4. Notebook dùng `assert` kiểm tra: train/test không chung nhóm, mọi fold đủ lớp, k lớn nhất (31) không vượt số mẫu học.

Kích thước fold: A có 1.408 học / 352 kiểm định; B có 1.083–1.084 học / 270–271 kiểm định.

**Vì sao chia theo nhóm?** Hai bản ghi giống hệt nhau rơi vào hai phía train/test thì KNN chỉ cần "nhớ" là đoán đúng → điểm test bị thổi phồng.

**Lưu ý trung thực:** sau khi loại trùng, mỗi nhóm chỉ còn 1 dòng (A 2.200/2.200, B 1.693/1.693), nên trên dữ liệu này chia theo nhóm cho kết quả tương đương chia phân tầng theo dòng. Bước này được giữ như **biện pháp phòng ngừa** (báo cáo mục 2.2.4).

### 6.3. Chống rò rỉ và chống "nhìn test"

- Scaler và bộ chọn đặc trưng nằm **trong `Pipeline`**, được fit lại trên phần học của từng fold.
- EDA chỉ làm trên train.
- `guard_training` **chặn huấn luyện lại** khi đã có `comparison.csv` hoặc `test_predictions.csv` (test đã mở).
- `selection.json` được ghi **trước** khi đánh giá test; ô đánh giá test kiểm tra lựa chọn không bị đổi.
- Mọi mô hình dùng **cùng 5 fold** (`cv_splits`) → so sánh công bằng.

### 6.4. Chỉ số

Kiểm định chéo báo cáo dạng **trung bình ± độ lệch chuẩn qua 5 fold** (ddof=0 theo GridSearchCV). Trên test: Accuracy, F1 macro, precision/recall macro, `classification_report` theo lớp và ma trận nhầm lẫn.

---

## 7. Các mô hình và lưới tham số

| Mô hình | Vai trò | Lưới tham số | Số cấu hình |
|---|---|---|---|
| KNN | Mô hình chính | `scale` ∈ {giữ nguyên, StandardScaler, MinMaxScaler} × `k` ∈ 9 giá trị × `weights` ∈ {uniform, distance} × `metric` ∈ {euclidean, manhattan, minkowski p=3} | 162 |
| Gaussian NB | Mô hình cơ sở | StandardScaler + `var_smoothing` ∈ {1e-11, 1e-9, 1e-7} | 3 |
| Decision Tree | So sánh | `max_depth` ∈ {None, 5, 10} × `min_samples_leaf` ∈ {1, 3} | 6 |
| Random Forest | So sánh | 200 cây; `max_depth` ∈ {None, 10} × `min_samples_leaf` ∈ {1, 2} | 4 |
| KNN_final | Biến thể KNN chọn bằng CV | 4 biến thể ablation (mục 8.3) | 4 |

Cấu hình KNN **ban đầu** (`start`, notebook mục 6): StandardScaler, k = 5, Euclidean, uniform, đủ 7 đặc trưng.

---

## 8. Kết quả

Toàn bộ số liệu lấy từ `outputs/crop_a_02` và `outputs/crop_b_02`.

### 8.1. Cấu hình KNN được chọn

| | A | B |
|---|---|---|
| k | 3 | 5 |
| Khoảng cách | Manhattan | Euclidean |
| Trọng số | distance | uniform |
| Thang đo | Giữ nguyên | MinMaxScaler |
| Số đặc trưng | 7 | 7 |

### 8.2. So sánh mô hình

**Kiểm định chéo (F1 macro, trung bình ± SD):**

| Mô hình | A | B |
|---|---|---|
| KNN | 0,98366 ± 0,00111 | 0,99566 ± 0,00462 |
| Gaussian NB | 0,99372 ± 0,00451 | **0,99940 ± 0,00121** |
| Decision Tree | 0,98748 ± 0,00686 | 0,98417 ± 0,01165 |
| Random Forest | **0,99428 ± 0,00508** | 0,99818 ± 0,00365 |

**Test:**

| Mô hình | A: F1 macro | A: đúng | B: F1 macro | B: đúng |
|---|---|---|---|---|
| KNN | 0,97714 | 430/440 | 0,99760 | 338/339 |
| Gaussian NB | **0,99544** | 438/440 | 0,99760 | 338/339 |
| Decision Tree | 0,97942 | 431/440 | 0,94745 | 334/339 |
| Random Forest | 0,99318 | 437/440 | **1,00000** | 339/339 |

Tham số tốt nhất: Gaussian NB `var_smoothing = 1e-11` (cả hai bộ); Decision Tree `max_depth = None, min_samples_leaf = 1` (cả hai bộ); Random Forest `max_depth = 10` (A) / `None` (B), `min_samples_leaf = 1`.

### 8.3. Ablation (đóng góp của từng thay đổi)

| Biến thể | Ý nghĩa | A: F1 CV | B: F1 CV |
|---|---|---|---|
| `start_7` | Cấu hình ban đầu, 7 đặc trưng | 0,97031 ± 0,01070 | 0,95742 ± 0,02636 |
| `tuned_7` | Cấu hình tối ưu, 7 đặc trưng | **0,98366 ± 0,00111** | **0,99566 ± 0,00462** |
| `start_5` | Ban đầu + chọn 5 đặc trưng bằng MI | 0,96509 ± 0,01100 | 0,99365 ± 0,00542 |
| `tuned_5` | Tối ưu + chọn 5 đặc trưng bằng MI | 0,97788 ± 0,00474 | 0,99346 ± 0,00580 |

| Thay đổi | A (điểm %) | B (điểm %) |
|---|---|---|
| Tối ưu cấu hình (`start_7 → tuned_7`) | +1,33 | +3,82 |
| Chọn đặc trưng, giữ cấu hình ban đầu (`start_7 → start_5`) | −0,52 | +3,62 |
| Chọn đặc trưng, giữ cấu hình tối ưu (`tuned_7 → tuned_5`) | −0,58 | −0,22 |

**Kết luận:** tối ưu cấu hình là cải tiến hiệu quả nhất; chọn đặc trưng chỉ có ích với cấu hình ban đầu trên B. **KNN_final = `tuned_7` ở cả hai bộ.** Chọn đặc trưng còn làm thời gian fit mỗi fold tăng từ khoảng 0,003 giây lên 0,07–0,11 giây.

**Cần nói rõ khi bảo vệ:** `start_7` là một trong 162 cấu hình của lưới và `tuned_7` là cấu hình tốt nhất trên **chính 5 fold này**, nên mức tăng +1,33 / +3,82 không thể âm và có phần lạc quan; đánh giá chặt hơn cần nested CV. `tuned_5` dùng lại siêu tham số của 7 đặc trưng, chưa tối ưu lại. Bề mặt điểm khá phẳng: trên B uniform và distance hòa tuyệt đối (0,99566), có 17 cấu hình khác cách cấu hình được chọn ≤ 0,002.

### 8.4. Mutual information (trên train)

| Đặc trưng | A (hạng) | B (hạng) |
|---|---|---|
| humidity | 1,732 (1) | 1,699 (2) |
| K | 1,652 (2) | 1,733 (1) |
| rainfall | 1,633 (3) | 1,429 (3) |
| P | 1,320 (4) | 1,386 (4) |
| N | 1,002 (5) | 1,091 (5) |
| temperature | 0,996 (6) | 1,000 (6) |
| ph | 0,709 (7) | 0,637 (7) |

### 8.5. Learning curve của KNN_final

| Bộ | Số mẫu học mỗi fold | F1 kiểm định |
|---|---|---|
| A | 352 → 704 → 1.056 → 1.408 | 0,95633 → 0,97284 → 0,98075 → 0,98366 |
| B | 270 → 541 → 812 → 1.083 | 0,85795 → 0,96204 → 0,99038 → 0,99566 |

Trên B điểm tăng mạnh khi thêm dữ liệu và hai đường train/kiểm định gần trùng ở cuối → hạn chế chính là **thiếu mẫu cho lớp nhỏ**, không phải quá khớp.

### 8.6. Tốc độ tìm láng giềng (lần chạy 23/11/2026)

KD-Tree truy vấn nhanh nhất ở cả hai bộ (A: 0,0045 ms/mẫu so với 0,0116 ms/mẫu của brute) nhưng fit chậm hơn; ba cách cho **cùng nhãn dự đoán (100%)**. Thời gian chỉ vài mili giây nên mô hình giữ `brute`.

### 8.7. Phân tích lỗi

- **A:** KNN sai 10/440. 4 mẫu **rice → jute**, 1 jute → rice; còn lại ở maize, pigeonpeas, mothbeans, lentil. Một mẫu rice bị **cả 4 mô hình** đoán thành jute → vùng chồng lấn thật giữa hai lớp.
- **B:** KNN sai 1 mẫu (maize → cotton). Decision Tree sai 4 jackfruit → arecanut, recall jackfruit chỉ 0,20 dù Accuracy vẫn 98,53%.

---

## 9. Những con số dễ hiểu sai

| Hiện tượng | Giải thích |
|---|---|
| **KNN A đạt F1 = 1,0 trên tập học** | `weights="distance"`: chấm lại một mẫu train thì chính nó là láng giềng ở khoảng cách 0, trọng số gần như vô hạn → luôn đúng. Điểm tập học **không mang thông tin**; đọc CV 0,98366 và test 0,97714. B dùng `uniform` nên tập học chỉ 0,99604. |
| **Decision Tree 1,0 trên tập học** | Cây không giới hạn độ sâu khớp hoàn toàn tập học → **quá khớp thật**: B chỉ còn CV 0,98417 ± 0,01165, test 0,94745. |
| **Random Forest 1,0 trên test B** | Test nhỏ, một số lớp chỉ 5 mẫu; trong CV có fold chỉ đạt 0,99089. Không có nghĩa là hoàn hảo. |
| **Demo hiện "Lúa 100%"** | Là **tỷ lệ phiếu**: cả 5 láng giềng đều là rice. Không phải xác suất trồng thành công. |
| **KNN thua Gaussian NB** | Trên A khác biệt **có ý nghĩa**: thấp hơn ở 5/5 fold, McNemar trên test p ≈ 0,008 (8 mẫu chỉ NB đúng, 0 mẫu chỉ KNN đúng); trên B không phân biệt được. Lý do: trong từng lớp các đặc trưng gần như **độc lập** (\|r\| trung bình 0,085 trên train A) – đúng giả định của Naive Bayes; NB học phương sai riêng cho từng đặc trưng trong từng lớp, KNN dùng một khoảng cách chung → khó tách rice–jute. **Lưu ý:** phân bố trong lớp *không* phải hình chuông (kurtosis ≈ −1,2, gần phân bố đều), nên đừng nói "vì dữ liệu Gaussian". KNN vẫn được giữ cho demo vì đề tài nghiên cứu KNN và KNN giải thích được qua láng giềng. |
| **Có rò rỉ dữ liệu không?** | Không: chia theo nhóm, B đã loại trùng, Pipeline fit theo fold. Nếu rò rỉ, CV và test cũng phải ≈ 1, trong khi KNN A chỉ 0,977–0,984. |
| **Chạy lại có ra số khác không?** | Không. Notebook được chạy lại toàn bộ ngày 23/11/2026 (lần chạy 02, revision 7): **mọi chỉ số CV và test trùng khớp** lần trước; chỉ thời gian đo (Bảng 3.15) thay đổi theo máy. Báo cáo ghi ở mục 3.1.3. |

---

## 10. Thư mục outputs

Mỗi thư mục `outputs/crop_<a|b>_02` có cùng bộ file:

| Nhóm | File | Nội dung |
|---|---|---|
| Hồ sơ chạy | `config.json` | Seed, tham số, phiên bản thư viện, SHA-256 dữ liệu, `revision` |
| | `python_runtime.json` | Phiên bản Python (3.12.14) |
| | `split.json` | `sample_id` của train/test, nhóm, các fold kiểm định, danh sách lớp |
| | `selection.json` | Mô hình tốt nhất theo CV, biến thể KNN cuối, tham số; ghi **trước khi mở test** |
| Dữ liệu | `dataset_used.csv` | Dữ liệu sau loại trùng; số thứ tự dòng chính là `sample_id` |
| | `row_sources.csv`, `truoc_sau_bo_trung.csv`, `chat_luong_hai_nguon.csv` | Nguồn từng dòng, số mẫu trước/sau loại trùng, chất lượng hai nguồn |
| | `chong_lap_hai_nguon.csv`, `xung_dot_nhan.csv` | Nhóm chung giữa A và B, nhóm mang nhiều nhãn |
| Khảo sát | `thong_ke_train.csv`, `tuong_quan_train.csv`, `ngoai_lai_iqr_train.csv`, `chuan_hoa_truoc_sau.csv`, các hình `class_distribution.png`, `feature_*.png`, `features_by_crop.png` | Khảo sát trên train |
| Huấn luyện | `*_search.joblib` | GridSearchCV đã fit, dùng khi `RESTORE_SEARCHES=True` |
| | `*_cv_raw.csv`, `knn_cv.csv`, `cv_summary.csv`, `kich_thuoc_fold.csv` | Điểm từng fold, bảng tổng hợp |
| | `knn_factors.png`, `knn_train_cv.png`, `models_train_cv.png` | Ảnh hưởng từng yếu tố KNN, so sánh train và CV |
| Cải tiến | `knn_ablation*.csv/png`, `mutual_information_train.*`, `knn_learning_curve.*`, `neighbors_benchmark*.csv`, `knn_selected_folds.csv` | Ablation, MI, learning curve, benchmark |
| Đánh giá | `comparison.csv` | **Bảng tổng hợp chính**: train, CV, test của 5 mô hình |
| | `test_predictions.csv` | Dự đoán của mọi mô hình trên từng mẫu test |
| | `*_classification_report.csv`, `*_confusion_matrix.png` | Chỉ số theo lớp, ma trận nhầm lẫn |
| | `knn_errors*.csv`, `knn_confused_pairs.csv` (và bản `knn_final_*`) | Mẫu sai, cặp nhãn hay nhầm |
| Mô hình | `knn_crop.joblib` | Gói mô hình cho demo |
| | `inference-requirements.txt` | Thư viện tối thiểu để nạp mô hình |
| Báo cáo | `bang_mo_hinh_bao_cao.csv`, `ket_qua_dataset_hien_co.*` (A), `doi_chieu_hai_dataset.*` (B) | Bảng, hình tổng hợp |

**Cấu trúc `knn_crop.joblib`** (một `dict`): `schema_version` (= 1), `pipeline` (KNN_final đã fit trên toàn bộ train), `features` (thứ tự 7 đặc trưng), `classes`, `training_bounds` (min/max từng đặc trưng trên train), `config`, `selection`, `versions`.

---

## 11. Cài đặt và chạy lại

### 11.1. Cài đặt (một lần)

Python 3.12. Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra trước khi chạy:
- `data/` có đủ hai CSV; thiếu thì notebook báo `FileNotFoundError` ngay ở mục 2.
- Có `outputs/crop_a_02` và `outputs/crop_b_02` nếu muốn xem kết quả hoặc chạy demo.

### 11.2. Chạy lại notebook

1. Mở `crop.ipynb`, chọn kernel `.venv\Scripts\python.exe`.
2. Sửa ô cấu hình ở mục 2:

   ```python
   DATA_MODE = 'nguon_a'   # 'nguon_a' cho bộ A, 'nguon_b' cho bộ B
   RUN_VERSION = '03'      # đặt số mới để không đụng kết quả chuẩn 02
   ```

3. Chạy lần lượt từ trên xuống; kết quả lưu ở `outputs/crop_<a|b>_<RUN_VERSION>`.
4. Đổi `DATA_MODE` và chạy lại cho bộ còn lại.
5. So `comparison.csv` của lần chạy mới với `outputs/crop_*_02/comparison.csv`: các chỉ số phải trùng khớp.

**Notebook cố ý báo lỗi** (cơ chế chống ghi đè) khi:
- đã có `<model>_search.joblib` → không fit lại mô hình đó;
- đã có `comparison.csv` hoặc `test_predictions.csv` → không huấn luyện/tuning thêm sau khi mở test;
- đã có `knn_crop.joblib` hoặc file zip → không ghi đè.

Gặp các lỗi này nghĩa là thư mục chạy đã có kết quả: đặt `RUN_VERSION` mới. **Không xóa hoặc sửa `outputs/crop_*_02`.** Khi kernel tắt giữa chừng, đặt `RESTORE_SEARCHES = True` ở mục 7 để nạp lại kết quả đã lưu.

**Tái lập:** seed 42 và `n_jobs=1` cho kết quả lặp lại trên cùng phiên bản thư viện. Tập test A/B đã được xem trong các lần phát triển trước, nên chạy lại với cùng cách chia **không tạo ra test độc lập mới**.

---

## 12. Ứng dụng demo

### 12.1. Chạy

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Trình duyệt mở `http://localhost:8501`. Ứng dụng **chỉ nạp mô hình đã lưu**, không huấn luyện lại. Tắt bằng `Ctrl + C` trong cửa sổ PowerShell.

### 12.2. Giao diện

Phong cách **sổ tay nông trại**: nền kem, xanh cỏ dịu, nâu gỗ, tiêu đề font Lora, chữ thân font Nunito.

| Thành phần | Chức năng |
|---|---|
| **Bộ dữ liệu** | Chuyển giữa bộ A và B, mỗi bộ có mô hình riêng |
| Tab **Gợi ý cây trồng** | Khung **Chỉ số đất và khí hậu**: nhập 7 chỉ số (dấu ? cho biết khoảng train), hoặc bấm: |
| | • **Mẫu thật từ tập test**: lấy ngẫu nhiên một mẫu test, kết quả hiện kèm nhãn thật và đúng/sai |
| | • **Ngẫu nhiên trong khoảng**: sinh từng giá trị ngẫu nhiên trong khoảng min–max của train |
| | • **Mẫu ví dụ**: N=85, P=40, K=40, 26 °C, 80%, pH 6,5, 200 mm (cùng mẫu với notebook mục 17) |
| | Khung **Cây trồng được gợi ý**: chỉ hiện cây có phiếu (tối đa 3); mỗi thẻ có biểu tượng, tên, hạng, thanh điểm phiếu; thẻ hạng 1 viền vàng. Cảnh báo khi giá trị ngoài khoảng train hoặc có nhãn đồng điểm |
| | Khung **Láng giềng gần nhất trong tập train**: các láng giềng KNN dùng để bỏ phiếu, khoảng cách và 7 chỉ số |
| Tab **Thông tin mô hình** | Cấu hình KNN_final; F1 macro CV/test, Accuracy, số mẫu test; bảng so sánh với Gaussian NB, Decision Tree, Random Forest; cách đọc kết quả |

**Kiểm tra đầu vào** (giống notebook mục 16): đủ 7 trường, số hữu hạn; N, P, K, rainfall không âm; humidity trong [0, 100], ph trong [0, 14]. Nhãn có điểm 0 không được hiện, nên số thẻ có thể ít hơn 3. Đổi bộ dữ liệu hoặc sửa tay một chỉ số thì thông báo nhãn thật của mẫu test cũ tự ẩn.

### 12.3. Kịch bản demo 2 phút (đã chạy thử)

| Bước | Thao tác | Kết quả mong đợi | Nói gì |
|---|---|---|---|
| 1 | Chọn **Bộ B**, bấm **Mẫu ví dụ** | Lúa, điểm phiếu 100% | "Cả 5 láng giềng gần nhất đều là rice, trùng notebook." |
| 2 | Cuộn xuống bảng láng giềng | 5 dòng, đều là Lúa | "KNN giải thích được dự đoán bằng chính các mẫu train này." |
| 3 | Chọn **Bộ A**, bấm **Mẫu thật từ tập test** | Thẻ gợi ý + thông báo xanh "dự đoán đúng" (đa số trường hợp) | "Mẫu này mô hình chưa từng học; KNN A đúng 430/440 mẫu test." |
| 4 | Sửa **Lượng mưa** thành 900 | Cảnh báo vàng: ngoài khoảng train [20,21; 298,56] | "Mô hình vẫn trả lời nhưng cảnh báo vì đang ngoại suy." |
| 5 | Mở tab **Thông tin mô hình** | Cấu hình k=3, Manhattan…; bảng so sánh | "Gaussian NB cao hơn; nhóm giải thích ở mục 3.4.6." |

Nếu bước 3 ra **sai** (hiếm, A sai 10/440): đó là cơ hội tốt để nói về phân tích lỗi, ví dụ cặp rice–jute.

---

## 13. Báo cáo

`nhom08m.docx` (60 trang, kèm `nhom08m.pdf`) theo mẫu ICTU:

| Phần | Nội dung chính |
|---|---|
| Đầu báo cáo | Bìa, mục lục, phân công nhiệm vụ, danh mục từ viết tắt, danh mục hình, danh mục bảng |
| Mở đầu | Lý do, mục tiêu, dữ liệu, bố cục |
| Chương 1. Cơ sở lý thuyết | Học máy, dữ liệu và EDA, tiền xử lý, quá trình học và tổng quát hóa, quy trình học máy |
| Chương 2. Phương pháp KNN | Phát biểu bài toán, dữ liệu, thuật toán KNN, các mô hình so sánh, quy trình huấn luyện |
| Chương 3. Thử nghiệm và đánh giá | 3.1 điều kiện thực nghiệm · 3.2 khảo sát dữ liệu · 3.3 kết quả và lỗi · 3.4 cải tiến KNN · 3.5 lưu mô hình và demo |
| Kết luận | Kết quả, hạn chế, hướng phát triển, việc sử dụng AI |
| Tài liệu tham khảo | 14 nguồn |

Bảng 3.1–3.17 và Hình 3.1–3.13 lấy từ `outputs/crop_*_02` (Hình 3.10 được vẽ lại từ `knn_cv.csv` để có nhãn trục). Hình 3.14–3.15 là ảnh chụp demo (`docs/hinh_bao_cao/`), Bảng 3.18 ghi kết quả chạy thử demo. Mọi bản trước khi sửa nằm trong `backups/`.

**Bản `nhom08m` khác bản `nhom 08` ở đâu** (sau rà soát độc lập bằng 4 sub agent và một agent kiểm chứng):
- Viết lại lý giải KNN thua NB (độc lập trong lớp thay vì "dữ liệu Gaussian"), thêm kiểm định McNemar và so sánh theo fold (mục 3.4.6).
- Nói rõ ablation lạc quan, bề mặt điểm phẳng, B ⊂ A 88% với 5 cặp nhãn đổi tên, chia theo nhóm chỉ là phòng ngừa, learning curve 25% lấy mẫu không phân tầng.
- Vẽ lại Hình 3.10, mọi bảng/hình được nhắc trong lời văn, tài liệu tham khảo đánh số theo thứ tự trích (không ghi ngày truy cập), bỏ MSE khỏi danh mục viết tắt, sửa các lỗi câu chữ.

**Bản đồ "tiêu chí chấm → nằm ở đâu trong báo cáo":**

| Tiêu chí | Mục báo cáo |
|---|---|
| Phát biểu bài toán, mục tiêu – chỉ số | 2.1, Bảng 2.1 |
| Nguồn, từ điển dữ liệu | 2.2.1–2.2.2, Bảng 2.2–2.3 |
| EDA, quan hệ đặc trưng – nhãn | 3.2.1–3.2.3, Hình 3.1–3.7 |
| Chất lượng dữ liệu, trước – sau | 3.2.4, Bảng 3.3–3.4 |
| Chia dữ liệu, chống rò rỉ | 2.2.4, 3.1.1, Bảng 3.1 |
| Thuật toán, baseline | 2.3–2.4, Bảng 2.4–2.5 |
| CV, test, so sánh mô hình | 3.3.1–3.3.2, Bảng 3.5–3.7 |
| Phân tích lỗi, quá khớp | 3.3.3, Hình 3.8–3.9, Bảng 3.8 |
| Hạn chế – giải pháp | 3.4.1, Bảng 3.9 |
| Tối ưu siêu tham số | 3.4.2, Bảng 3.10, Hình 3.10 |
| Cải tiến đặc trưng, ablation | 3.4.3–3.4.4, Bảng 3.11–3.13, Hình 3.11–3.12 |
| Learning curve, tổng quát hóa | 3.4.5–3.4.6, Bảng 3.14–3.16, Hình 3.13 |
| Không vượt baseline – giải thích | 3.4.6 |
| Demo | 3.5, Hình 3.14–3.15, Bảng 3.18 |
| Tái lập, trung thực | 3.1.3 |
| Sử dụng AI | Kết luận (đoạn cuối) |

---

## 14. Kiến thức nền cần nắm

Mỗi mục là "đủ để trả lời vấn đáp"; chi tiết hơn ở Chương 1–2 của báo cáo.

**KNN (K-Nearest Neighbors).** Không học tham số; lúc `fit` chỉ lưu dữ liệu train. Khi dự đoán: tính khoảng cách từ mẫu mới tới mọi mẫu train → lấy k mẫu gần nhất → bỏ phiếu. Điểm của lớp c = tổng trọng số láng giềng thuộc c / tổng trọng số; `predict_proba` trả về các điểm này.
- **k nhỏ** nhạy với nhiễu; **k lớn** làm mờ ranh giới lớp, lớp nhỏ bị lấn át (B với k = 31 chỉ đạt 0,95629).
- **uniform**: mỗi láng giềng 1 phiếu. **distance**: trọng số 1/khoảng cách, láng giềng gần nặng hơn.
- **Khoảng cách Minkowski** bậc p: d(x, z) = (Σ|xⱼ − zⱼ|ᵖ)^(1/p). p = 1 là **Manhattan**, p = 2 là **Euclidean**.
- **Độ phức tạp:** fit gần như không tốn; dự đoán brute là O(n·d) cho mỗi mẫu. KD-Tree/Ball-Tree tăng tốc truy vấn khi số chiều thấp.

**Chuẩn hóa thang đo.** KNN tính khoảng cách trực tiếp nên đặc trưng có số lớn (rainfall, K) lấn át đặc trưng số nhỏ (ph).
- **z-score (StandardScaler):** (x − μ)/σ → mọi cột trung bình 0, SD 1 → mọi đặc trưng ngang nhau.
- **min–max (MinMaxScaler):** (x − min)/(max − min) → về [0, 1].
- Ở dự án này StandardScaler **kém nhất** vì làm ph và temperature (ít thông tin nhất theo MI) nặng ngang K và rainfall.

**Gaussian Naive Bayes.** Dùng định lý Bayes, giả định các đặc trưng độc lập có điều kiện theo lớp; mỗi đặc trưng trong mỗi lớp là một phân bố chuẩn (học trung bình, phương sai). `var_smoothing` cộng thêm một lượng nhỏ vào phương sai để ổn định.

**Decision Tree / Random Forest.** Cây chia dữ liệu theo ngưỡng để giảm độ hỗn tạp Gini = 1 − Σp_c². Cây sâu dễ quá khớp. Random Forest = nhiều cây (200) học trên mẫu bootstrap và tập con đặc trưng ngẫu nhiên, bỏ phiếu → ổn định hơn một cây.

**Kiểm định chéo (CV) 5 fold.** Chia train thành 5 phần; lần lượt học trên 4, kiểm định trên 1; báo cáo trung bình ± SD. Dùng để **chọn cấu hình**; test chỉ dùng một lần ở cuối. `StratifiedGroupKFold` = giữ tỷ lệ lớp + không tách một nhóm ra hai phía.

**Rò rỉ dữ liệu.** Thông tin của dữ liệu kiểm định/test lọt vào bước học (ví dụ fit scaler trên toàn bộ dữ liệu trước khi chia, chọn đặc trưng bằng nhãn của toàn bộ dữ liệu, cùng bản ghi ở cả train và test) → điểm bị thổi phồng. Phòng tránh: Pipeline, chia theo nhóm, khảo sát trên train.

**Chỉ số.** Accuracy = số đúng / tổng. Với lớp c: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). **F1 macro** = trung bình F1 các lớp (mỗi lớp ngang nhau).

**Mutual information (MI).** Đo mức phụ thuộc giữa một đặc trưng và nhãn (kể cả phi tuyến); = 0 khi độc lập. `SelectKBest(mutual_info_classif, k=5)` giữ 5 đặc trưng MI cao nhất, đặt trong Pipeline nên fit lại theo fold.

**Quá khớp / dưới khớp, learning curve.** Quá khớp: tốt trên tập học, kém trên kiểm định. Dưới khớp: kém cả hai. Learning curve vẽ điểm theo số mẫu học: hai đường cùng tăng và gần nhau như B → thiếu dữ liệu, không quá khớp.

**Ablation.** Bật/tắt từng thay đổi trên cùng fold, cùng chỉ số để đo đóng góp riêng của từng thay đổi.

---

## 15. Hướng dẫn bảo vệ

### 15.1. Cách giảng viên chấm (tóm tắt `tieuchicham.docx`, Bài thi kết thúc)

- **Tiêu chí nhóm [N] 1–5:** hoàn thiện, quy trình đúng, chạy lại được, kết quả (vượt baseline hoặc giải thích), báo cáo. Giảng viên hỏi nhanh từng người; **điểm cá nhân = MIN(điểm nhóm, mức hiểu của chính bạn)** → sản phẩm tốt nhưng không giải thích được thì vẫn mất điểm.
- **Tiêu chí cá nhân [CN] 6–10:** trình bày + demo, giải thích phương pháp **kể cả phần của thành viên kia**, phân tích kết quả và hạn chế, trả lời phản biện, thực hành tại chỗ trên mã của nhóm; đóng góp thể hiện qua bảng phân công và lịch sử commit/nhật ký.
- **Liêm chính:** báo cáo đã ghi phạm vi dùng AI; mỗi người phải giải thích được mọi đoạn mã và nội dung đã nộp.

### 15.2. Kịch bản trình bày 10 phút

Phân vai theo bảng phân công; nhóm có thể đổi người nhưng **mỗi người phải trả lời được toàn bộ**.

| Thời gian | Người | Nội dung | Điểm nhấn / số liệu nên nói |
|---|---|---|---|
| 0:00–1:30 | Minh | Bài toán, mục tiêu, chỉ số | Phân loại 7 đầu vào → nhãn; KNN chính, Gaussian NB cơ sở; F1 macro vì B lệch lớp |
| 1:30–4:00 | Thịnh | Dữ liệu và khảo sát | A 2.200/22 lớp, B 1.796 → 1.693 (103 trùng); 1.494 nhóm chung, 498 nhóm khác nhãn → không gộp; K cao của apple/grapes được giữ dù IQR gắn cờ |
| 4:00–6:30 | Minh | Quy trình và kết quả | Chia theo nhóm, 5 fold, Pipeline; 162 cấu hình; KNN CV 0,98366 / 0,99566, test 0,97714 / 0,99760; so với NB |
| 6:30–7:30 | Thịnh | Phân tích lỗi | rice → jute 4 mẫu (một mẫu cả 4 mô hình sai); DT B jackfruit recall 0,20 dù Accuracy 98,53% |
| 7:30–8:30 | Minh | Cải tiến và chọn mô hình | Ablation: tối ưu cấu hình +1,33 / +3,82; MI chỉ có ích với cấu hình ban đầu B; KNN_final = tuned_7; vì sao thua NB |
| 8:30–10:00 | Thịnh | Demo + giới hạn | Kịch bản 12.3 (bước 1, 3, 4); giới hạn: B chưa rõ nguồn gốc nhãn, A/B chồng lặp, test đã xem, thiếu địa điểm/đơn vị |

**Bản rút gọn 5 phút:** 0:00–1:00 bài toán (Minh) → 1:00–2:00 dữ liệu và lý do không gộp (Thịnh) → 2:00–3:30 quy trình + bảng kết quả + so baseline (Minh) → 3:30–5:00 demo bước 1 và 3 (Thịnh).

### 15.3. Mẹo khi trả lời

- Trả lời theo khung: **kết luận → bằng chứng (số liệu, mục báo cáo) → giới hạn**. Ví dụ: "KNN thua NB (kết luận); A thấp hơn 1,01 điểm CV, 1,83 điểm test (bằng chứng, mục 3.4.6); đây là giải thích dựa trên cơ chế, chưa có thí nghiệm riêng (giới hạn)."
- Không biết thì nói rõ phần đã kiểm chứng và phần chưa; **không bịa số liệu**.
- Luôn phân biệt **CV** (chọn mô hình) và **test** (đối chiếu cuối); **điểm phiếu** khác **xác suất**.
- Khi giảng viên chỉ vào một hình: nói trục là gì, điểm nào cao nhất/thấp nhất, kết luận gì.

---

## 16. Câu hỏi vấn đáp thường gặp

<details>
<summary><b>1. Vì sao chọn KNN?</b></summary>

Đề tài nghiên cứu KNN. Dữ liệu nhỏ (≈ 2.000 mẫu), 7 đặc trưng số, nên KNN chạy nhanh và giải thích được dự đoán qua các mẫu train cụ thể. KNN không giả định dạng phân bố. Đây là lựa chọn để nghiên cứu, không khẳng định KNN tốt nhất (mục 2.3.1).
</details>

<details>
<summary><b>2. KNN thua baseline, dự án có thất bại không?</b></summary>

Không. Tiêu chí chấm chấp nhận khi giải thích được lý do. Báo cáo thừa nhận khác biệt trên A có ý nghĩa (5/5 fold, McNemar p ≈ 0,008) và nêu ba nguyên nhân (mục 3.4.6): trong từng lớp các đặc trưng gần như độc lập (đúng giả định Naive Bayes); KNN dùng một khoảng cách chung cho mọi lớp; có mẫu khó cả 4 mô hình cùng sai. Nếu chỉ cần độ chính xác thì nên dùng Gaussian NB, báo cáo nói rõ.
</details>

<details>
<summary><b>2b. Dữ liệu có phải phân bố Gaussian không?</b></summary>

Không. Trong từng lớp, độ nhọn dư (kurtosis) có trung vị khoảng −1,2 ở cả 7 đặc trưng, gần phân bố đều. Gaussian NB vẫn tốt vì giả định *độc lập có điều kiện* gần đúng và việc học trung bình, phương sai riêng từng lớp đủ để tách lớp.
</details>

<details>
<summary><b>3. Vì sao không chọn mô hình theo điểm test?</b></summary>

Chọn theo test thì test không còn là dữ liệu "chưa thấy" → điểm lạc quan. Vì vậy chọn theo CV, ghi `selection.json` trước khi mở test, và notebook chặn huấn luyện lại sau khi mở test.
</details>

<details>
<summary><b>4. Vì sao chia theo nhóm, không chia ngẫu nhiên từng dòng?</b></summary>

Có các đầu vào giống nhau (đặc biệt ở B). Chia từng dòng thì bản sao có thể nằm ở cả train và test, KNN chỉ cần nhớ → rò rỉ. Nhóm = các dòng cùng 7 giá trị sau làm tròn 6 chữ số (mục 2.2.4). Trả lời trung thực nếu bị hỏi thêm: sau khi loại trùng mỗi nhóm chỉ còn 1 dòng, nên trên dữ liệu này nó tương đương chia phân tầng thường; nhóm giữ như biện pháp phòng ngừa.
</details>

<details>
<summary><b>5. Vì sao không gộp A và B cho nhiều dữ liệu hơn?</b></summary>

1.494 nhóm đầu vào xuất hiện ở cả hai, 498 nhóm mang nhãn khác nhau; hệ nhãn cũng khác (groundnut/groundnuts). Gộp tạo mâu thuẫn nhãn và rò rỉ train–test (mục 3.2.4).
</details>

<details>
<summary><b>6. Vì sao giữ các giá trị K bị IQR gắn cờ là ngoại lai?</b></summary>

Nhóm K ≈ 200 thuộc apple và grapes – là đặc điểm thật giúp phân biệt lớp, không phải lỗi đo. IQR chỉ đánh dấu "hiếm" về thống kê, không chứng minh sai (mục 3.2.3–3.2.4).
</details>

<details>
<summary><b>7. Vì sao đặt scaler trong Pipeline?</b></summary>

Fit scaler trên toàn bộ train trước khi chia fold thì min/max hoặc trung bình của phần kiểm định lọt vào bước học. Pipeline fit lại scaler trên phần học của từng fold.
</details>

<details>
<summary><b>8. Vì sao StandardScaler lại cho điểm thấp nhất?</b></summary>

Sau z-score, ph và temperature (MI thấp nhất) đóng góp vào khoảng cách ngang với K và rainfall. Giữ thang gốc hoặc MinMaxScaler thì ph (miền hẹp) đóng góp ít hơn (mục 3.4.2).
</details>

<details>
<summary><b>9. Vì sao A dùng Manhattan, B dùng Euclidean?</b></summary>

GridSearch chọn theo F1 macro CV trên từng bộ. Trên A, Manhattan + thang gốc đạt 0,98366; trên B, Euclidean + MinMax đạt 0,99566. Khác biệt giữa các khoảng cách nhỏ (B: 0,99566 so với 0,99507) nên không khái quát "khoảng cách nào tốt hơn".
</details>

<details>
<summary><b>10. k được chọn thế nào? k lớn thì sao?</b></summary>

Thử 9 giá trị k cùng thang đo, khoảng cách, trọng số (162 tổ hợp), chọn F1 macro CV cao nhất. k = 31 trên B chỉ còn 0,95629 vì lớp nhỏ chỉ ~19–20 mẫu train, vùng 31 láng giềng chứa nhiều lớp khác.
</details>

<details>
<summary><b>11. Mutual information là gì, sao không dùng tương quan Pearson?</b></summary>

MI đo mức phụ thuộc đặc trưng – nhãn kể cả phi tuyến. Nhãn là biến danh nghĩa nên không tính Pearson trực tiếp với nhãn được. Bộ chọn MI đặt trong Pipeline.
</details>

<details>
<summary><b>12. Chọn đặc trưng có giúp không?</b></summary>

Chỉ giúp cấu hình ban đầu trên B (+3,62 điểm). Với cấu hình đã tối ưu, bỏ 2 đặc trưng làm giảm nhẹ ở cả hai bộ (−0,58 / −0,22) và tốn thời gian hơn → mô hình cuối giữ đủ 7 đặc trưng.
</details>

<details>
<summary><b>13. Learning curve cho biết gì?</b></summary>

B: F1 kiểm định tăng từ 0,858 (270 mẫu) lên 0,996 (1.083 mẫu), hai đường gần trùng ở cuối → không quá khớp, hạn chế là thiếu dữ liệu lớp nhỏ; thêm dữ liệu có thể còn cải thiện. Lưu ý: ở mức 25%, tập con lấy ngẫu nhiên không phân tầng nên một fold thiếu hẳn một lớp và vài lớp ≤ 3 mẫu, nên 0,858 phản ánh cả ảnh hưởng của cách lấy mẫu.
</details>

<details>
<summary><b>14. Vì sao test A thấp hơn CV?</b></summary>

Test A 0,97714 thấp hơn CV 0,98366 khoảng 0,0065. Với 440 mẫu test, sai số chuẩn của Accuracy khoảng 0,007, nên chênh lệch này cùng cỡ với dao động do lấy mẫu (không so với SD qua fold). CV có thể hơi lạc quan vì cấu hình là cấu hình cao nhất trong 162 cấu hình trên cùng các fold (mục 3.4.6).
</details>

<details>
<summary><b>15. F1 macro khác Accuracy thế nào?</b></summary>

Accuracy tính trên toàn bộ mẫu; F1 macro tính F1 từng lớp rồi lấy trung bình không trọng số. Decision Tree B: Accuracy 98,53% nhưng F1 macro chỉ 0,94745 vì jackfruit đúng 1/5.
</details>

<details>
<summary><b>16. Điểm 1,0 trên tập học của KNN có phải quá khớp?</b></summary>

Không hẳn. Với `weights="distance"`, mẫu train tự là láng giềng ở khoảng cách 0 nên luôn đúng; điểm này không mang thông tin. Đọc CV và test.
</details>

<details>
<summary><b>17. Brute, KD-Tree, Ball-Tree khác nhau ra sao?</b></summary>

Brute tính khoảng cách tới mọi mẫu; KD-Tree/Ball-Tree dựng cây để bỏ qua vùng xa. Cùng kết quả (khớp 100%), chỉ khác tốc độ; dữ liệu nhỏ nên chênh vài mili giây.
</details>

<details>
<summary><b>18. Kết quả có tái lập được không?</b></summary>

Có. Seed 42, `n_jobs=1`, phiên bản thư viện ghi trong `config.json`. Chạy lại toàn bộ ngày 23/11/2026 cho chỉ số trùng khớp; chỉ thời gian đo thay đổi (mục 3.1.3).
</details>

<details>
<summary><b>19. Ý nghĩa thực tế của kết quả?</b></summary>

Trong phạm vi dữ liệu đã có, 7 chỉ số phân biệt rất tốt các loại cây (F1 > 0,97). Nhưng dữ liệu không có địa điểm, mùa vụ, đơn vị N/P/K, nguồn B chưa rõ → chưa dùng để tư vấn canh tác thật; cần dữ liệu thực địa Việt Nam và chuyên gia kiểm chứng.
</details>

<details>
<summary><b>20. Nếu dữ liệu có giá trị thiếu thì làm gì?</b></summary>

Dữ liệu hiện không thiếu (0 ô). Nếu có: đặt `SimpleImputer` (trung vị cho biến số có giá trị cực đoan) **trong Pipeline** trước scaler để fit theo fold, tránh rò rỉ. Xem đoạn mã S7 ở mục 17.
</details>

<details>
<summary><b>21. Nhóm dùng AI thế nào?</b></summary>

AI hỗ trợ soạn và rà soát mã notebook, demo, biên tập báo cáo. Sinh viên tự chạy notebook, lưu output, đối chiếu số liệu với CSV và chịu trách nhiệm giải thích. Báo cáo ghi ở đoạn cuối Kết luận.
</details>

---

## 17. Luyện thực hành tại chỗ

Giảng viên có thể yêu cầu trong 10–15 phút: đổi một siêu tham số, thêm một chỉ số, thay thuật toán, đổi khoảng tìm kiếm, vẽ đường cong kiểm định, thống kê/vẽ lại một đặc trưng, đổi cách điền khuyết.

**Chuẩn bị:** tạo notebook mới `thuc_hanh.ipynb` **trong thư mục dự án**, chọn kernel `.venv`. Các đoạn mã dưới đây **không cần chạy lại `crop.ipynb`** và **không ghi gì vào `outputs/`**: chúng dựng lại đúng tập train/test và 5 fold đã dùng từ file đã lưu.

> Trạng thái kiểm chứng: đoạn **S0, S8, S9** đã chạy thử trên cả hai bộ (S0 dựng đúng kích thước 5 fold như `kich_thuoc_fold.csv`; S8–S9 cho đúng 430/440 và 338/339). Các đoạn có huấn luyện (S1–S7) **chưa được chạy thử** – hãy tự chạy một lần trước buổi bảo vệ.

### S0. Khởi tạo (chạy đầu tiên)

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()          # notebook phải mở trong thư mục dự án
BO = "b"                   # "a" hoặc "b"
RUN = ROOT / "outputs" / f"crop_{BO}_02"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
SEED = 42
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

data = pd.read_csv(RUN / "dataset_used.csv")              # chỉ số dòng = sample_id
split = json.loads((RUN / "split.json").read_text(encoding="utf-8"))
train = data.loc[split["train_sample_ids"]]
test = data.loc[split["test_sample_ids"]]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# Dựng lại đúng 5 fold của notebook (theo vị trí trong X_train)
pos = {sid: i for i, sid in enumerate(X_train.index)}
cv_splits = []
for val_ids in split["cv_validation_sample_ids"]:
    val = np.array([pos[s] for s in val_ids])
    fit = np.setdiff1d(np.arange(len(X_train)), val)
    cv_splits.append((fit, val))

print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Fold (học, kiểm định):", [(len(f), len(v)) for f, v in cv_splits])
```

Kết quả đúng: A → `(1760, 7)`, `(440, 7)`, mỗi fold `(1408, 352)`; B → `(1354, 7)`, `(339, 7)`, fold `(1083–1084, 270–271)`.

**Tự kiểm tra:** chạy S1 với đúng cấu hình KNN_final (B: MinMaxScaler, k = 5, euclidean, uniform; A: bỏ scaler bằng `("scale", "passthrough")`, k = 3, manhattan, distance) phải ra F1 CV **0,99566 ± 0,00462** (B) / **0,98366 ± 0,00111** (A).

### S1. Đổi siêu tham số KNN và chạy kiểm định chéo

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

model = Pipeline([
    ("scale", MinMaxScaler()),            # thử StandardScaler() hoặc "passthrough"
    ("knn", KNeighborsClassifier(n_neighbors=7, metric="euclidean", weights="uniform")),
])
r = cross_validate(model, X_train, y_train, cv=cv_splits, scoring=SCORING, return_train_score=True)
print(f"F1 macro CV : {r['test_f1_macro'].mean():.5f} ± {r['test_f1_macro'].std():.5f}")
print(f"Accuracy CV : {r['test_accuracy'].mean():.5f} ± {r['test_accuracy'].std():.5f}")
print(f"F1 tập học  : {r['train_f1_macro'].mean():.5f}")
```

Đọc kết quả: so với KNN_final (mục 8.1); chênh lệch nhỏ hơn SD thì chưa đủ kết luận tốt hơn.

### S2. Thêm chỉ số đánh giá

```python
scoring2 = {
    "accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro", "recall_macro": "recall_macro",
}
r = cross_validate(model, X_train, y_train, cv=cv_splits, scoring=scoring2)
for name in scoring2:
    s = r[f"test_{name}"]
    print(f"{name:18s} {s.mean():.5f} ± {s.std():.5f}")
```

`balanced_accuracy` = trung bình recall các lớp (hữu ích khi lệch lớp); `f1_weighted` lấy trọng số theo số mẫu lớp.

### S3. Thay thuật toán

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

models = {
    "LogisticRegression": Pipeline([("scale", StandardScaler()),
                                    ("model", LogisticRegression(max_iter=2000))]),
    "SVC (RBF)": Pipeline([("scale", StandardScaler()), ("model", SVC())]),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
}
for name, m in models.items():
    r = cross_validate(m, X_train, y_train, cv=cv_splits, scoring=SCORING)
    print(f"{name:20s} F1 CV {r['test_f1_macro'].mean():.5f} ± {r['test_f1_macro'].std():.5f}")
```

Lưu ý: mô hình dựa trên khoảng cách/hệ số (LogReg, SVC) cần scaler trong Pipeline; mô hình cây thì không.

### S4. Đổi khoảng tìm kiếm siêu tham số (GridSearchCV)

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("scale", MinMaxScaler()), ("knn", KNeighborsClassifier(algorithm="brute"))])
grid = {
    "knn__n_neighbors": [2, 4, 6, 8, 10],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}
gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="f1_macro", cv=cv_splits, n_jobs=1)
gs.fit(X_train, y_train)
print("Tốt nhất:", gs.best_params_, f"F1 CV = {gs.best_score_:.5f}")
res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_f1_macro", "std_test_f1_macro", "rank_test_f1_macro"]]
res.sort_values("rank_test_f1_macro").head(10)
```

Nguyên tắc cần nói: chỉ dùng CV trên train để chọn, **không** dùng test.

### S5. Vẽ đường cong kiểm định theo k

```python
from sklearn.model_selection import validation_curve

ks = [1, 3, 5, 7, 9, 11, 15, 21, 31]
pipe = Pipeline([("scale", MinMaxScaler()), ("knn", KNeighborsClassifier())])
tr, va = validation_curve(pipe, X_train, y_train, param_name="knn__n_neighbors",
                          param_range=ks, cv=cv_splits, scoring="f1_macro")
plt.errorbar(ks, va.mean(axis=1), yerr=va.std(axis=1), marker="o", capsize=3, label="Kiểm định (± SD)")
plt.plot(ks, tr.mean(axis=1), marker="s", label="Tập học")
plt.xlabel("Số láng giềng k"); plt.ylabel("F1 macro")
plt.title(f"Đường cong kiểm định theo k – bộ {BO.upper()}"); plt.legend(); plt.grid(alpha=.3); plt.show()
```

Đọc: k nào cao nhất; khoảng cách giữa hai đường (lớn → quá khớp); k tăng thì điểm giảm do lớp nhỏ bị lấn át.

### S6. Thống kê và vẽ lại một đặc trưng

```python
f = "rainfall"                      # đổi tên đặc trưng theo yêu cầu
print(X_train[f].describe().round(3))
print("Trung vị:", round(X_train[f].median(), 3))

X_train[f].plot.hist(bins=30, edgecolor="white")
plt.xlabel(f); plt.ylabel("Số mẫu"); plt.title(f"Phân bố {f} trên tập train {BO.upper()}"); plt.show()

train.boxplot(column=f, by=TARGET, rot=90, figsize=(12, 4))
plt.suptitle(""); plt.title(f"{f} theo nhãn"); plt.ylabel(f); plt.show()
```

### S7. Đổi phương pháp điền khuyết (giả lập dữ liệu thiếu)

```python
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(SEED)
X_miss = X_train.mask(rng.random(X_train.shape) < 0.05)     # xóa ngẫu nhiên 5% ô
print("Số ô thiếu giả lập:", int(X_miss.isna().sum().sum()))

for strategy in ["mean", "median", "most_frequent"]:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy=strategy)),           # fit theo từng fold → không rò rỉ
        ("scale", MinMaxScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    r = cross_validate(pipe, X_miss, y_train, cv=cv_splits, scoring="f1_macro")
    print(f"{strategy:14s} F1 CV {r['test_score'].mean():.5f} ± {r['test_score'].std():.5f}")
```

Dữ liệu thật không thiếu ô nào; đoạn này chỉ để minh họa cách đặt bộ điền khuyết đúng chỗ.

### S8. Xem láng giềng của một mẫu test (dùng mô hình đã lưu)

```python
import joblib

pipe = joblib.load(RUN / "knn_crop.joblib")["pipeline"]
i = 0                                   # đổi chỉ số mẫu test
x = X_test.iloc[[i]]
print("Nhãn thật:", y_test.iloc[i], "| Dự đoán:", pipe.predict(x)[0])

z = x
for name in ("scale", "select"):
    step = pipe.named_steps.get(name)
    if step is not None and not isinstance(step, str):
        z = step.transform(z)
knn = pipe.named_steps["knn"]
z = np.asarray(z, dtype=float)
if hasattr(knn, "feature_names_in_"):
    z = pd.DataFrame(z, columns=knn.feature_names_in_)
dist, idx = knn.kneighbors(z)
X_train.iloc[idx[0]].assign(label=y_train.iloc[idx[0]].values, khoang_cach=dist[0])
```

### S9. Báo cáo theo lớp và ma trận nhầm lẫn của mô hình đã lưu

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

pred = pipe.predict(X_test)
print(classification_report(y_test, pred, digits=4, zero_division=0))
print("Số mẫu đúng:", int((pred == y_test.values).sum()), "/", len(y_test))
ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=90, colorbar=False)
plt.gcf().set_size_inches(9, 9); plt.show()
```

Kết quả đúng: A 430/440, B 338/339.

**Lỗi hay gặp:** `FileNotFoundError` → notebook không mở trong thư mục dự án (sửa `ROOT`); `NameError` → chưa chạy S0 hoặc chưa chạy đoạn định nghĩa `model`/`pipe`; tên tham số Pipeline phải có tiền tố bước (`knn__n_neighbors`, không phải `n_neighbors`).

---

## 18. Giới hạn và hướng phát triển

**Giới hạn:**
- Chưa rõ quy trình thu thập và gán nhãn của bộ B.
- B có 88% mẫu trùng đầu vào với A, khác A chủ yếu ở 5 cặp nhãn đổi tên → không phải hai bằng chứng độc lập.
- Không có địa điểm, thời điểm, đơn vị đo N/P/K; không suy ra điều kiện Việt Nam.
- Tập test đã được xem trong các lần phát triển trước; một số lớp B chỉ có 5 mẫu test.
- Kiểm định thống kê mới làm cho một số cặp mô hình (McNemar trên test, so sánh theo fold), chưa lặp với nhiều seed.
- Mức tăng của bước tối ưu cấu hình được đo trên chính các fold dùng để chọn (chưa có nested CV).
- Điểm top-3 là tỷ lệ phiếu, chưa hiệu chỉnh thành xác suất.

**Hướng phát triển:** thu thập dữ liệu thực địa có địa điểm, mùa vụ, đơn vị đo và đánh giá trên tập test mới; lặp CV với nhiều seed và kiểm định thống kê; học trọng số đặc trưng cho khoảng cách KNN; hiệu chỉnh điểm thành xác suất; đối chiếu với chuyên gia nông nghiệp.

---

## 19. Phân công

| Thành viên | Nhiệm vụ chính |
|---|---|
| Nguyễn Quang Minh | Phát biểu bài toán và chỉ số đánh giá; xây dựng Pipeline KNN và kiểm định chéo; cài đặt Gaussian NB, Decision Tree, Random Forest; tổng hợp kết quả A/B |
| Nông Hữu Thịnh | Thu thập dữ liệu, kiểm tra nguồn và bản ghi trùng; khảo sát phân bố và quan hệ đặc trưng; phân tích ma trận nhầm lẫn và lỗi theo lớp; biên tập cơ sở lý thuyết và mô tả dữ liệu |
| Cả hai | Tổng hợp kết quả A/B và rà soát báo cáo |

Bảng chi tiết ở phần "Phân công nhiệm vụ" của báo cáo.

---

## 20. Checklist trước buổi bảo vệ

**Báo cáo**
- [ ] In `nhom08m.pdf` (60 trang); kiểm tra trang trống bất thường trước khi in quyển.
- [ ] Hai thành viên ký cột "Chữ ký" ở bảng phân công.

**Máy trình diễn**
- [ ] `.venv` đã cài đủ thư viện (`pip install -r requirements.txt`).
- [ ] Có `data/` và `outputs/crop_a_02`, `outputs/crop_b_02`.
- [ ] Chạy thử `streamlit run app.py`, đi hết kịch bản 12.3.
- [ ] Mở sẵn `crop.ipynb` (đã có output) và `thuc_hanh.ipynb` đã chạy S0.
- [ ] Đã tự chạy S1–S7 ít nhất một lần, ghi lại thời gian chạy.

**Kiến thức (mỗi người)**
- [ ] Đọc mục 1, 8, 9, 14, 16.
- [ ] Nói được 6 con số chính: CV và test KNN của A, B; NB tương ứng; số mẫu A/B.
- [ ] Giải thích được phần của thành viên còn lại.
- [ ] Tập trình bày theo kịch bản 15.2, bấm giờ.

---

## 21. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Baseline (mô hình cơ sở) | Mô hình đơn giản làm mốc so sánh – ở đây là Gaussian NB |
| CV (kiểm định chéo) | Chia train thành 5 phần; lần lượt học trên 4, kiểm định trên 1 |
| Fold | Một lượt chia trong kiểm định chéo |
| Pipeline | Chuỗi bước biến đổi + mô hình, fit cùng nhau trong từng fold |
| GridSearchCV | Thử mọi tổ hợp tham số, chọn tổ hợp có điểm CV cao nhất |
| Ablation | Bật/tắt từng thay đổi để đo đóng góp riêng |
| Mutual information (MI) | Mức thông tin một đặc trưng cung cấp về nhãn |
| Learning curve | Điểm mô hình theo số mẫu học |
| Validation curve | Điểm mô hình theo giá trị một siêu tham số |
| Rò rỉ dữ liệu (leakage) | Thông tin kiểm định/test lọt vào bước học, làm điểm bị thổi phồng |
| Quá khớp (overfitting) | Học thuộc tập học, kém trên dữ liệu mới |
| `weights="distance"` | Láng giềng càng gần, phiếu càng nặng |
| Điểm phiếu top-3 | Tỷ lệ (có trọng số) láng giềng thuộc mỗi nhãn |
| SD (ddof) | Độ lệch chuẩn; ddof=0 chia cho n (CV), ddof=1 chia cho n−1 (Bảng 3.2) |

Danh mục từ viết tắt đầy đủ ở đầu báo cáo.

---

## 22. Sử dụng AI

Công cụ AI hỗ trợ soạn và rà soát mã notebook, ứng dụng demo, tài liệu và biên tập báo cáo. Sinh viên tự chạy notebook, lưu output và chịu trách nhiệm giải thích mã cùng kết quả. Mọi số liệu trong báo cáo và tài liệu này được đối chiếu với các file CSV trong `outputs/crop_*_02`.
