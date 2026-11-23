"""Ứng dụng minh họa khuyến nghị cây trồng bằng KNN (nhóm 08).

Chạy:  streamlit run app.py
Ứng dụng chỉ nạp mô hình đã lưu trong outputs/crop_{a,b}_02/knn_crop.joblib, không huấn luyện lại.
Màu và font ở .streamlit/config.toml (phong cách sổ tay nông trại); CSS dưới đây chỉ thêm nền thẻ.
"""
from html import escape

import numpy as np
import streamlit as st

import demo_core as dc

st.set_page_config(page_title="Khuyến nghị cây trồng · Nhóm 08", page_icon="🌾", layout="wide")

# Nền sáng và bóng nhẹ cho các khung (container có key panel_* / card_*), thẻ gợi ý chính viền vàng.
st.html(
    """
<style>
[class*="st-key-panel_"], [class*="st-key-card_"] {
  background: #fffdf7;
  box-shadow: 0 1px 2px rgba(61,50,38,.05), 0 8px 24px rgba(61,50,38,.06);}
.st-key-card_0 {background: #fffaea; border-color: #d9a521 !important;}
.crop-icon {font-size: 52px; line-height: 1.1; margin: 0;}
.page-head {display: flex; align-items: center; gap: 14px; padding: 18px 22px; margin-bottom: 6px;
  border-radius: 14px; background: linear-gradient(100deg, #eef3e2 0%, #f7efd9 100%);
  border: 1px solid #ddd0b4;}
.page-head .leaf {font-size: 40px;}
.page-head h1 {font-family: 'Lora', serif; font-size: 1.9rem; margin: 0; color: #3d3226;}
.page-head p {margin: 2px 0 0; color: #6b5a45;}
</style>
"""
)


@st.cache_resource(show_spinner="Đang nạp mô hình đã lưu…")
def get_run(name: str) -> dc.DemoRun:
    return dc.load_run(name)


EXAMPLE = {"N": 85.0, "P": 40.0, "K": 40.0, "temperature": 26.0,
           "humidity": 80.0, "ph": 6.5, "rainfall": 200.0}
LIMITS = {  # (min, max, step, format)
    "N": (0.0, 300.0, 1.0, "%.0f"), "P": (0.0, 300.0, 1.0, "%.0f"), "K": (0.0, 300.0, 1.0, "%.0f"),
    "temperature": (-10.0, 60.0, 0.1, "%.2f"), "humidity": (0.0, 100.0, 0.5, "%.2f"),
    "ph": (0.0, 14.0, 0.05, "%.2f"), "rainfall": (0.0, 1000.0, 1.0, "%.2f"),
}
ICONS = {"N": ":material/science:", "P": ":material/science:", "K": ":material/science:",
         "temperature": ":material/thermostat:", "humidity": ":material/humidity_percentage:",
         "ph": ":material/water_ph:", "rainfall": ":material/rainy:"}


def set_inputs(values: dict, truth: str | None = None) -> None:
    for f in dc.FEATURES:
        low, high, _, _ = LIMITS[f]
        st.session_state[f"in_{f}"] = float(min(max(values[f], low), high))
    st.session_state["truth"] = truth


def clear_truth() -> None:
    st.session_state["truth"] = None


def fill_test(run: dc.DemoRun) -> None:
    sample, label, _ = dc.random_test_sample(run, np.random.default_rng())
    set_inputs(sample, truth=label)


def fill_uniform(run: dc.DemoRun) -> None:
    set_inputs(dc.random_uniform_sample(run, np.random.default_rng()))


if "in_N" not in st.session_state:
    set_inputs(EXAMPLE)

# ---------------------------------------------------------------- Đầu trang
st.html(
    """
<div class="page-head">
  <span class="leaf">🌾</span>
  <div><h1>Khuyến nghị cây trồng</h1>
  <p>Nhập chỉ số đất và khí hậu; mô hình K-Nearest Neighbors tìm các mẫu gần nhất và gợi ý cây trồng phù hợp.</p></div>
</div>
"""
)

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    dataset = st.segmented_control(
        "Bộ dữ liệu", ["A", "B"], default="A", required=True, key="dataset",
        format_func=lambda x: f"Bộ {x}", on_change=clear_truth,
    )
    run = get_run(dataset)
    n_rows = f"{len(run.data):,}".replace(",", ".")
    st.caption(f"{n_rows} mẫu · {len(run.bundle['classes'])} loại cây · mỗi bộ có mô hình KNN riêng")

tab_rec, tab_model = st.tabs([":material/eco: Gợi ý cây trồng", ":material/insights: Thông tin mô hình"])

# ---------------------------------------------------------------- Tab 1: gợi ý
with tab_rec:
    with st.container(border=True, key="panel_inputs"):
        st.subheader("Chỉ số đất và khí hậu", anchor=False)
        with st.container(horizontal=True, gap="small"):
            st.button("Mẫu thật từ tập test", key="btn_test", icon=":material/casino:",
                      on_click=fill_test, args=(run,),
                      help="Lấy ngẫu nhiên một bản ghi test mà mô hình chưa học; nhãn thật được hiện để đối chiếu.")
            st.button("Ngẫu nhiên trong khoảng", key="btn_rand", icon=":material/shuffle:",
                      on_click=fill_uniform, args=(run,),
                      help="Sinh từng chỉ số ngẫu nhiên trong khoảng min–max của train. Tổ hợp có thể không giống cây nào.")
            st.button("Mẫu ví dụ", key="btn_ex", icon=":material/restart_alt:",
                      on_click=set_inputs, args=(EXAMPLE,),
                      help="N=85, P=40, K=40, 26 °C, 80%, pH 6,5, 200 mm – cùng mẫu với notebook.")
        cols = st.columns(4)
        for i, f in enumerate(dc.FEATURES):
            low, high, step, fmt = LIMITS[f]
            name, hint = dc.FEATURE_INFO[f]
            lo_t, hi_t = run.bounds[f]
            cols[i % 4].number_input(
                name, min_value=low, max_value=high, step=step, format=fmt, icon=ICONS[f],
                key=f"in_{f}", on_change=clear_truth,
                help=f"{hint}. Khoảng trong train: {lo_t:g} – {hi_t:g}.",
            )

    sample = {f: float(st.session_state[f"in_{f}"]) for f in dc.FEATURES}
    try:
        top3, notes = dc.recommend(run, sample)
        nb = dc.neighbors(run, sample)
    except ValueError as err:
        st.error(f"Chỉ số chưa hợp lệ: {err}", icon=":material/error:")
        st.stop()

    voted = top3[top3["diem"] > 0]
    with st.container(border=True, key="panel_result"):
        st.subheader("Cây trồng được gợi ý", anchor=False)
        st.caption(f"Tối đa 3 cây có điểm phiếu cao nhất từ {len(nb)} láng giềng gần nhất. "
                   "Chỉ hiện cây có phiếu.")
        cards = st.columns(3)
        for rank, (col, row) in enumerate(zip(cards, voted.itertuples())):
            vi, icon = dc.CROP_VI.get(row.cay_trong, (row.cay_trong, "🌱"))
            with col.container(border=True, key=f"card_{rank}", height="stretch"):
                st.html(f'<p class="crop-icon">{icon}</p>')
                st.markdown(f"#### {escape(vi)}")
                if rank == 0:
                    st.badge("Phù hợp nhất", icon=":material/star:", color="orange")
                else:
                    st.badge(f"Hạng {rank + 1}", color="gray")
                st.progress(float(row.diem), text=f"Điểm phiếu {100 * row.diem:.0f}%")

        truth = st.session_state.get("truth")
        if truth:
            ok = truth == top3.iloc[0]["cay_trong"]
            msg = f"Nhãn thật của bản ghi test: **{dc.crop_name(truth)}** — mô hình dự đoán {'đúng' if ok else 'sai'}."
            if ok:
                st.success(msg, icon=":material/check_circle:")
            else:
                st.error(msg, icon=":material/cancel:")
        for note in notes:
            st.warning(note, icon=":material/warning:")

    with st.container(border=True, key="panel_neighbors"):
        st.subheader("Láng giềng gần nhất trong tập train", anchor=False)
        show = nb.reset_index(drop=True)
        show["label"] = show["label"].map(dc.crop_name)
        show = show[["label", "khoang_cach", *dc.FEATURES]].rename(
            columns={"label": "Cây trồng", "khoang_cach": "Khoảng cách",
                     **{f: dc.FEATURE_INFO[f][0] for f in dc.FEATURES}})
        st.dataframe(
            show, hide_index=True, alt="Các láng giềng gần nhất trong tập train và khoảng cách",
            column_config={"Khoảng cách": st.column_config.NumberColumn(format="%.4f")},
        )

# ---------------------------------------------------------------- Tab 2: thông tin mô hình
with tab_model:
    knn = run.pipeline.named_steps["knn"]
    scale = run.pipeline.named_steps["scale"]
    scale_name = "Giữ thang gốc" if isinstance(scale, str) else type(scale).__name__

    with st.container(border=True, key="panel_config"):
        st.subheader(f"Cấu hình KNN_final · bộ {dataset}", anchor=False)
        with st.container(horizontal=True):
            st.metric("Số láng giềng k", knn.n_neighbors, border=True)
            st.metric("Khoảng cách", knn.metric.capitalize(), border=True)
            st.metric("Trọng số phiếu", knn.weights, border=True)
            st.metric("Thang đo", scale_name, border=True)

    m = dc.metrics_table(run)
    knn_row = m.iloc[0]
    cv = run.comparison.set_index("model").loc["KNN_final"]
    with st.container(border=True, key="panel_metrics"):
        st.subheader("Kết quả đã lưu (lần chạy 02)", anchor=False)
        with st.container(horizontal=True):
            st.metric("F1 macro kiểm định chéo", f"{cv['cv_f1_macro']:.5f}", border=True,
                      help=f"Trung bình 5 fold trên train, SD = {cv['cv_f1_sd']:.5f}")
            st.metric("F1 macro test", f"{knn_row['F1 macro test']:.5f}", border=True)
            st.metric("Accuracy test", f"{knn_row['Accuracy test']:.2f}%", border=True)
            st.metric("Số mẫu test", f"{len(run.test)}", border=True)
        st.dataframe(
            m, hide_index=True, alt=f"So sánh KNN với các mô hình khác trên bộ {dataset}",
            column_config={"F1 macro test": st.column_config.NumberColumn(format="%.5f"),
                           "Accuracy test": st.column_config.NumberColumn(format="%.2f%%")},
        )
        st.caption("Gaussian NB (mô hình cơ sở) đạt điểm cao hơn hoặc bằng KNN. KNN được dùng cho demo "
                   "vì có thể giải thích dự đoán qua các láng giềng cụ thể.")

    with st.container(border=True, key="panel_guide"):
        st.subheader("Cách đọc kết quả", anchor=False)
        st.markdown(
            """
- **Điểm phiếu** là tỷ lệ phiếu (có trọng số) của các láng giềng gần nhất cho mỗi cây,
  *không phải* xác suất canh tác thành công. Chỉ cây có phiếu mới được hiện.
- **Mẫu thật từ tập test** là bản ghi mô hình chưa học, nên có thể so dự đoán với nhãn thật.
- **Ngẫu nhiên trong khoảng** sinh từng chỉ số độc lập; tổ hợp có thể không giống cây nào,
  khi đó gợi ý chỉ phản ánh mẫu train gần nhất.
- **Giới hạn:** dữ liệu không có địa điểm, mùa vụ và đơn vị đo N, P, K; hồ sơ gán nhãn của bộ B
  chưa rõ. Kết quả chỉ minh họa phương pháp, cần chuyên gia nông nghiệp kiểm chứng trước khi áp dụng.
"""
        )
