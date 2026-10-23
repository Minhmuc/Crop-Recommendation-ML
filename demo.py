"""Demo top-3 dùng KNN đã lưu từ notebook. Không chứa lệnh huấn luyện.

Chạy sau mục 12–13 của notebook:
    python demo.py --model outputs/lan_01/knn_crop.joblib
"""

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def validate_sample(sample):
    """Kiểm tra 7 số đầu vào, không tự đổi đơn vị hoặc cắt giá trị."""
    if set(sample) != set(FEATURES):
        raise ValueError(f"Cần đúng 7 trường: {', '.join(FEATURES)}")
    if any(isinstance(sample[name], (bool, np.bool_)) for name in FEATURES):
        raise ValueError("Hãy nhập số đo, không nhập True/False.")
    try:
        row = pd.DataFrame([{name: float(sample[name]) for name in FEATURES}])
    except (TypeError, ValueError) as exc:
        raise ValueError("Tất cả đầu vào phải là số; dùng dấu chấm thập phân.") from exc
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("Đầu vào không được chứa NaN hoặc vô cực.")
    for name in ["N", "P", "K", "rainfall"]:
        if row.at[0, name] < 0:
            raise ValueError(f"{name} không được âm.")
    if not 0 <= row.at[0, "humidity"] <= 100:
        raise ValueError("humidity phải nằm trong [0, 100] %.")
    if not 0 <= row.at[0, "ph"] <= 14:
        raise ValueError("ph phải nằm trong [0, 14] cho minh họa này.")
    return row


def load_bundle(path):
    """Chỉ mở file joblib do bạn tự tạo từ notebook của dự án."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có mô hình: {path}. Hãy tự chạy notebook đến mục 12 trước."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("File mô hình không đúng định dạng của dự án.")
    if bundle.get("features") != FEATURES:
        raise ValueError("Danh sách hoặc thứ tự đặc trưng không đúng.")
    if not {"pipeline", "training_bounds", "classes"}.issubset(bundle):
        raise ValueError("File mô hình thiếu thông tin dự đoán.")
    if list(bundle["pipeline"].classes_) != bundle["classes"]:
        raise ValueError("Thứ tự nhãn trong metadata không khớp mô hình.")
    return bundle


def recommend_top3(bundle, sample):
    """Trả về 3 hàng và các chú thích; xác suất giữ trên toàn bộ 22 lớp.

    Không chuẩn hóa riêng 3 kết quả thành tổng 100%. Các cây không nhận
    phiếu vẫn có thể xuất hiện với xác suất 0 và được đánh dấu rõ.
    """
    row = validate_sample(sample)
    notes = []
    for name in FEATURES:
        lower, upper = bundle["training_bounds"][name]
        value = float(row.at[0, name])
        if value < lower or value > upper:
            notes.append(
                f"{name}={value:g} ngoài khoảng train [{lower:g}, {upper:g}]; "
                "kết quả cho mẫu này cần được xem xét thận trọng."
            )

    model = bundle["pipeline"]
    probabilities = np.asarray(model.predict_proba(row)[0], dtype=float)
    classes = np.asarray(model.classes_)
    if probabilities.shape != (len(classes),) or len(classes) < 3:
        raise ValueError("Mô hình phải trả về xác suất cho ít nhất 3 lớp.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Xác suất mô hình không hợp lệ.")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Tổng xác suất của các lớp khác 1.")

    # Sắp xếp ổn định: nếu bằng nhau, giữ thứ tự classes_ của mô hình.
    order = np.argsort(-probabilities, kind="stable")[:3]
    result = pd.DataFrame({
        "hang": [1, 2, 3],
        "cay_trong": classes[order],
        "xac_suat": probabilities[order],
        "xac_suat_%": probabilities[order] * 100,
        "dien_giai": [
            "Có phiếu từ láng giềng" if value > 0 else "Không có phiếu; không đủ căn cứ đề xuất"
            for value in probabilities[order]
        ],
    })
    if np.count_nonzero(probabilities > 0) < 3:
        notes.append("Có ít hơn 3 lớp nhận phiếu; các hàng 0% không phải đề xuất có căn cứ.")
    if any(np.count_nonzero(probabilities == value) > 1 for value in probabilities[order]):
        notes.append("Có đồng hạng; thứ tự giữa các lớp cùng xác suất không mang ý nghĩa ưu tiên.")
    notes.append(
        "Đây là xác suất phân lớp từ phiếu KNN, chưa hiệu chỉnh; "
        "không phải xác suất canh tác thành công."
    )
    return result, notes


def main():
    parser = argparse.ArgumentParser(description="Nhập 7 đặc trưng để xem top-3 nhãn cây trồng.")
    parser.add_argument("--model", type=Path, default=Path("outputs/lan_01/knn_crop.joblib"))
    args = parser.parse_args()
    try:
        bundle = load_bundle(args.model)
        print("Nhập số theo thang đo dữ liệu Kaggle, dùng dấu chấm thập phân.")
        print("temperature: độ C; humidity: %; ph: pH; rainfall: mm theo kỳ của dữ liệu nguồn.")
        sample = {}
        for name in FEATURES:
            while True:
                try:
                    value = float(input(f"{name}: ").strip())
                    if not np.isfinite(value):
                        raise ValueError
                    sample[name] = value
                    break
                except ValueError:
                    print("Hãy nhập một số hữu hạn, ví dụ 25.5.")
        result, notes = recommend_top3(bundle, sample)
        print(result.drop(columns="xac_suat").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        for note in notes:
            print(f"- {note}")
    except (FileNotFoundError, ValueError, EOFError, KeyboardInterrupt) as exc:
        parser.exit(1, f"\nChưa thể dự đoán: {exc}\n")


if __name__ == "__main__":
    main()
