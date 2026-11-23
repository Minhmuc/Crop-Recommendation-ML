"""Phần xử lý cho ứng dụng minh họa (không phụ thuộc Streamlit).

Chỉ nạp mô hình và kết quả đã lưu trong outputs/crop_{a,b}_02.
Không huấn luyện lại mô hình.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN_VERSION = "02"
RUNS = {
    "A": ROOT / "outputs" / f"crop_a_{RUN_VERSION}",
    "B": ROOT / "outputs" / f"crop_b_{RUN_VERSION}",
}
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_INFO = {
    "N": ("Nitơ (N)", "Tỷ lệ hàm lượng nitơ, đơn vị theo nguồn dữ liệu"),
    "P": ("Phospho (P)", "Tỷ lệ hàm lượng phospho, đơn vị theo nguồn dữ liệu"),
    "K": ("Kali (K)", "Tỷ lệ hàm lượng kali, đơn vị theo nguồn dữ liệu"),
    "temperature": ("Nhiệt độ (°C)", "Nhiệt độ trung bình"),
    "humidity": ("Độ ẩm (%)", "Độ ẩm tương đối, 0–100"),
    "ph": ("pH đất", "Thang 0–14"),
    "rainfall": ("Lượng mưa (mm)", "Kỳ tích lũy chưa rõ trong nguồn"),
}

CROP_VI = {
    "apple": ("Táo", "🍎"), "arecanut": ("Cau", "🌴"), "banana": ("Chuối", "🍌"),
    "beans": ("Đậu", "🫘"), "blackgram": ("Đậu đen", "🫘"), "chickpea": ("Đậu gà", "🫘"),
    "coconut": ("Dừa", "🥥"), "coffee": ("Cà phê", "☕"), "cotton": ("Bông", "☁️"),
    "cowpeas": ("Đậu đũa", "🫛"), "grapes": ("Nho", "🍇"), "groundnut": ("Lạc (groundnut)", "🥜"),
    "groundnuts": ("Lạc (groundnuts)", "🥜"), "jackfruit": ("Mít", "🍈"), "jute": ("Đay", "🌿"),
    "kidneybeans": ("Đậu tây", "🫘"), "lentil": ("Đậu lăng", "🫘"), "maize": ("Ngô", "🌽"),
    "mango": ("Xoài", "🥭"), "mothbeans": ("Đậu bướm", "🫘"), "mungbean": ("Đậu xanh", "🫛"),
    "muskmelon": ("Dưa lưới", "🍈"), "orange": ("Cam", "🍊"), "papaya": ("Đu đủ", "🥭"),
    "peas": ("Đậu Hà Lan", "🫛"), "pigeonpeas": ("Đậu triều", "🫘"), "pomegranate": ("Lựu", "🍎"),
    "rice": ("Lúa", "🌾"), "soyabeans": ("Đậu tương", "🫘"), "sugarcane": ("Mía", "🎋"),
    "watermelon": ("Dưa hấu", "🍉"),
}


def crop_label(name: str) -> str:
    vi, icon = CROP_VI.get(name, (name, "🌱"))
    return f"{icon} {vi} · {name}"


def crop_name(name: str) -> str:
    """Tên ngắn để hiển thị: biểu tượng + tên tiếng Việt."""
    vi, icon = CROP_VI.get(name, (name, "🌱"))
    return f"{icon} {vi}"


@dataclass
class DemoRun:
    name: str
    bundle: dict
    data: pd.DataFrame          # dữ liệu sau làm sạch (dataset_used.csv)
    train_ids: list[int]
    test: pd.DataFrame          # test_predictions.csv
    comparison: pd.DataFrame    # comparison.csv

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    @property
    def bounds(self) -> dict:
        return self.bundle["training_bounds"]

    @property
    def x_train(self) -> pd.DataFrame:
        return self.data.loc[self.train_ids, FEATURES]


def load_run(name: str) -> DemoRun:
    run_dir = RUNS[name]
    bundle = joblib.load(run_dir / "knn_crop.joblib")
    if bundle.get("schema_version") != 1 or bundle["features"] != FEATURES:
        raise ValueError(f"Gói mô hình {run_dir} không đúng cấu trúc.")
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    data = pd.read_csv(run_dir / "dataset_used.csv")
    test = pd.read_csv(run_dir / "test_predictions.csv")
    comparison = pd.read_csv(run_dir / "comparison.csv")
    return DemoRun(name, bundle, data, list(split["train_sample_ids"]), test, comparison)


def validate_sample(sample: dict, features: list[str] = FEATURES) -> pd.DataFrame:
    """Cùng quy tắc với mục 16 của notebook crop.ipynb."""
    if set(sample) != set(features):
        raise ValueError(f"Cần đúng các trường: {features}")
    if any(isinstance(sample[f], (bool, np.bool_)) for f in features):
        raise ValueError("Nhập số đo, không nhập True/False.")
    row = pd.DataFrame([{f: float(sample[f]) for f in features}])
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("Không nhận NaN hoặc vô cực.")
    if row[["N", "P", "K", "rainfall"]].lt(0).any().any():
        raise ValueError("N/P/K và rainfall không âm.")
    if not 0 <= row.at[0, "humidity"] <= 100 or not 0 <= row.at[0, "ph"] <= 14:
        raise ValueError("humidity thuộc [0,100], ph thuộc [0,14].")
    return row


def recommend(run: DemoRun, sample: dict, top: int = 3):
    """Trả về (bảng top-k, danh sách ghi chú). Điểm là tỷ lệ phiếu KNN."""
    row = validate_sample(sample)
    model = run.pipeline
    score = np.asarray(model.predict_proba(row)[0])
    order = np.argsort(-score, kind="stable")[:top]
    result = pd.DataFrame({"cay_trong": model.classes_[order], "diem": score[order]})
    notes = []
    for f, (low, high) in run.bounds.items():
        if not low <= row.at[0, f] <= high:
            notes.append(f"{FEATURE_INFO[f][0]} = {row.at[0, f]:g} nằm ngoài khoảng train [{low:g}, {high:g}].")
    if any(score[i] > 0 and (score == score[i]).sum() > 1 for i in order):
        notes.append("Có nhãn đồng điểm; thứ tự giữa các nhãn cùng điểm không thể hiện ưu tiên.")
    return result, notes


def neighbors(run: DemoRun, sample: dict) -> pd.DataFrame:
    """Các láng giềng gần nhất trong train mà KNN dùng để bỏ phiếu."""
    row = validate_sample(sample)
    pipe = run.pipeline
    transformed = row
    for step in ("scale", "select"):
        est = pipe.named_steps.get(step)
        if est is not None and not isinstance(est, str):
            transformed = est.transform(transformed)
    knn = pipe.named_steps["knn"]
    names = getattr(knn, "feature_names_in_", None)
    transformed = np.asarray(transformed, dtype=float)
    if names is not None:
        transformed = pd.DataFrame(transformed, columns=list(names))
    dist, idx = knn.kneighbors(transformed)
    ids = [run.train_ids[i] for i in idx[0]]
    table = run.data.loc[ids, FEATURES + ["label"]].copy()
    table.insert(0, "khoang_cach", dist[0])
    table.index.name = "sample_id"
    return table


def random_test_sample(run: DemoRun, rng: np.random.Generator) -> tuple[dict, str, int]:
    """Lấy ngẫu nhiên một bản ghi trong tập test (mô hình chưa học bản ghi này)."""
    rec = run.test.iloc[int(rng.integers(len(run.test)))]
    return {f: float(rec[f]) for f in FEATURES}, str(rec["true_label"]), int(rec["sample_id"])


def random_uniform_sample(run: DemoRun, rng: np.random.Generator) -> dict:
    """Sinh từng giá trị ngẫu nhiên đều trong khoảng train (có thể không giống cây nào)."""
    out = {}
    for f, (low, high) in run.bounds.items():
        value = rng.uniform(low, high)
        out[f] = float(round(value)) if f in ("N", "P", "K") else round(float(value), 2)
    return out


def metrics_table(run: DemoRun) -> pd.DataFrame:
    names = {"KNN_final": "KNN (mô hình demo)", "GaussianNB": "Gaussian NB (cơ sở)",
             "DecisionTree": "Decision Tree", "RandomForest": "Random Forest"}
    t = run.comparison.set_index("model").loc[list(names)]
    return pd.DataFrame({
        "Mô hình": [names[m] for m in t.index],
        "F1 macro CV": [f"{a:.5f} ± {b:.5f}" for a, b in zip(t["cv_f1_macro"], t["cv_f1_sd"])],
        "F1 macro test": t["test_f1_macro"].round(5).to_numpy(),
        "Accuracy test": (100 * t["test_accuracy"]).round(2).to_numpy(),
    })
