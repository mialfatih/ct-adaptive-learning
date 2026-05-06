import pandas as pd
import streamlit as st

from utils import get_supabase

st.set_page_config(page_title="Monitoring CT Adaptive", layout="wide")

st.title("Dashboard Monitoring Siswa")
st.caption("Monitoring progres pretest, treatment, posttest, dan export data uji instrumen")


# =========================
# FETCH ALL DATA SUPABASE
# =========================
def fetch_all_rows(table_name: str, page_size: int = 1000):
    supabase = get_supabase()
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1

        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(start, end)
            .execute()
        )

        rows = response.data or []
        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        start += page_size

    return all_rows


# =========================
# DATA LOAD
# =========================
@st.cache_data(ttl=15)
def load_data():
    siswa_rows = fetch_all_rows("siswa")
    session_rows = fetch_all_rows("session_pembelajaran")
    jawaban_rows = fetch_all_rows("jawaban_siswa")

    siswa_df = pd.DataFrame(siswa_rows)
    session_df = pd.DataFrame(session_rows)
    jawaban_df = pd.DataFrame(jawaban_rows)

    if siswa_df.empty:
        siswa_df = pd.DataFrame(columns=["id", "nis", "nama", "kelas"])

    if session_df.empty:
        session_df = pd.DataFrame(columns=[
            "id", "siswa_id", "status_session", "treatment_status",
            "d_score", "p_score", "a_score", "alg_score", "total_score",
            "weakest_indicator", "prediksi_ml", "answered_count",
            "current_ct", "current_level", "posttest_score", "gain_score",
            "updated_at", "finished_at"
        ])

    if jawaban_df.empty:
        jawaban_df = pd.DataFrame(columns=[
            "id", "session_id", "siswa_id", "nis", "nama", "kelas",
            "phase", "question_id", "ct", "level", "materi",
            "selected_answer", "correct_answer", "is_correct",
            "score_binary", "score_weighted", "attempt_order", "created_at"
        ])

    if not session_df.empty:
        data = session_df.merge(
            siswa_df,
            left_on="siswa_id",
            right_on="id",
            how="left",
            suffixes=("_session", "_siswa")
        )
    else:
        data = session_df.copy()

    return data, jawaban_df


# =========================
# HELPERS
# =========================
def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def make_status(row):
    status = str(row.get("status_session", "")).strip()
    treatment = str(row.get("treatment_status", "")).strip()

    if status == "selesai":
        return "Selesai"
    if status == "posttest":
        return "Posttest"
    if status == "treatment" and treatment == "skip":
        return "Skip Treatment"
    if status == "treatment":
        return "Treatment"
    if status == "pretest":
        return "Pretest"
    return "Belum Mulai"


def build_item_matrix(jawaban_df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    Export format uji instrumen:
    NIS | nama | kelas | Q1 | Q2 | ... | total_score_binary
    Isi Q = 1 benar, 0 salah.
    """
    if jawaban_df.empty:
        return pd.DataFrame()

    if "phase" not in jawaban_df.columns:
        return pd.DataFrame()

    df = jawaban_df[jawaban_df["phase"].astype(str).str.lower() == phase].copy()

    if df.empty:
        return pd.DataFrame()

    df["question_id"] = df["question_id"].astype(str)
    df["score_binary"] = (
        pd.to_numeric(df["score_binary"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    matrix = df.pivot_table(
        index=["nis", "nama", "kelas"],
        columns="question_id",
        values="score_binary",
        aggfunc="first"
    ).reset_index()

    question_cols = [c for c in matrix.columns if c not in ["nis", "nama", "kelas"]]
    question_cols = sorted(question_cols)

    matrix = matrix[["nis", "nama", "kelas"] + question_cols]
    matrix["total_score_binary"] = matrix[question_cols].sum(axis=1)

    return matrix


def download_button_csv(label, df, file_name):
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv"
    )


# =========================
# LOAD DATA
# =========================
try:
    data, jawaban_df = load_data()
except Exception as e:
    st.error(f"Gagal membaca data Supabase: {e}")
    st.stop()

if data.empty:
    st.warning("Belum ada data siswa.")
    st.stop()


# =========================
# NORMALISASI DATA SESSION
# =========================
text_cols = [
    "nis", "nama", "kelas", "status_session", "treatment_status",
    "prediksi_ml", "weakest_indicator", "current_ct", "current_level"
]

for col in text_cols:
    if col in data.columns:
        data[col] = clean_text(data[col])

num_cols = [
    "d_score", "p_score", "a_score", "alg_score",
    "total_score", "answered_count", "posttest_score", "gain_score"
]

for col in num_cols:
    if col in data.columns:
        data[col] = (
            pd.to_numeric(data[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

data["status_label"] = data.apply(make_status, axis=1)

if "updated_at" in data.columns:
    data["updated_at_dt"] = pd.to_datetime(data["updated_at"], errors="coerce")
else:
    data["updated_at_dt"] = pd.NaT


# =========================
# NORMALISASI JAWABAN
# =========================
if not jawaban_df.empty:
    for col in ["nis", "nama", "kelas", "phase", "question_id", "ct", "level"]:
        if col in jawaban_df.columns:
            jawaban_df[col] = clean_text(jawaban_df[col])

    if "score_binary" in jawaban_df.columns:
        jawaban_df["score_binary"] = (
            pd.to_numeric(jawaban_df["score_binary"], errors="coerce")
            .fillna(0)
            .astype(int)
        )


# =========================
# FILTER
# =========================
with st.container(border=True):
    st.subheader("Filter")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kelas_list = ["Semua"] + sorted([
            x for x in data["kelas"].dropna().unique().tolist()
            if str(x).strip() != ""
        ])
        kelas_filter = st.selectbox("Kelas", kelas_list)

    with c2:
        status_list = ["Semua"] + sorted([
            x for x in data["status_label"].dropna().unique().tolist()
            if str(x).strip() != ""
        ])
        status_filter = st.selectbox("Status", status_list)

    with c3:
        pred_list = ["Semua"] + sorted([
            x for x in data["prediksi_ml"].dropna().unique().tolist()
            if str(x).strip() != ""
        ])
        pred_filter = st.selectbox("Prediksi ML", pred_list)

    with c4:
        keyword = st.text_input("Cari NIS / Nama", "")


filtered = data.copy()

if kelas_filter != "Semua":
    filtered = filtered[filtered["kelas"] == kelas_filter]

if status_filter != "Semua":
    filtered = filtered[filtered["status_label"] == status_filter]

if pred_filter != "Semua":
    filtered = filtered[filtered["prediksi_ml"] == pred_filter]

if keyword.strip():
    key = keyword.strip().lower()
    filtered = filtered[
        filtered["nis"].str.lower().str.contains(key, na=False) |
        filtered["nama"].str.lower().str.contains(key, na=False)
    ]


# =========================
# RINGKASAN
# =========================
total_siswa = filtered["siswa_id"].nunique() if "siswa_id" in filtered.columns else len(filtered)
pretest_count = len(filtered[filtered["status_label"] == "Pretest"])
treatment_count = len(filtered[filtered["status_label"] == "Treatment"])
posttest_count = len(filtered[filtered["status_label"] == "Posttest"])
selesai_count = len(filtered[filtered["status_label"] == "Selesai"])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Siswa", total_siswa)
m2.metric("Pretest", pretest_count)
m3.metric("Treatment", treatment_count)
m4.metric("Posttest", posttest_count)
m5.metric("Selesai", selesai_count)

with st.caption("Data jawaban tersimpan"):
    pass

j1, j2, j3 = st.columns(3)
with j1:
    st.metric("Baris Pretest", len(jawaban_df[jawaban_df["phase"] == "pretest"]) if "phase" in jawaban_df.columns else 0)
with j2:
    st.metric("Baris Treatment", len(jawaban_df[jawaban_df["phase"] == "treatment"]) if "phase" in jawaban_df.columns else 0)
with j3:
    st.metric("Baris Posttest", len(jawaban_df[jawaban_df["phase"] == "posttest"]) if "phase" in jawaban_df.columns else 0)


# =========================
# MONITORING UTAMA
# =========================
st.subheader("Monitoring Siswa")

monitor_cols = [
    "nis", "nama", "kelas", "status_label",
    "prediksi_ml", "weakest_indicator",
    "total_score", "answered_count",
    "current_ct", "current_level",
    "posttest_score", "gain_score", "updated_at"
]

existing_cols = [c for c in monitor_cols if c in filtered.columns]
monitor_df = filtered[existing_cols].copy()

rename_map = {
    "nis": "NIS",
    "nama": "Nama",
    "kelas": "Kelas",
    "status_label": "Status",
    "prediksi_ml": "ML",
    "weakest_indicator": "Weakest",
    "total_score": "Pretest",
    "answered_count": "Treatment",
    "current_ct": "CT",
    "current_level": "Level",
    "posttest_score": "Posttest",
    "gain_score": "Gain",
    "updated_at": "Update"
}

monitor_df = monitor_df.rename(columns=rename_map)

if "Update" in monitor_df.columns:
    monitor_df = monitor_df.sort_values("Update", ascending=False, na_position="last")

st.dataframe(
    monitor_df,
    use_container_width=True,
    hide_index=True,
    height=520
)


# =========================
# RINGKASAN DISTRIBUSI
# =========================
with st.expander("Lihat Ringkasan Distribusi"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Status**")
        status_df = (
            filtered["status_label"]
            .value_counts()
            .reset_index()
        )
        status_df.columns = ["Status", "Jumlah"]
        st.dataframe(status_df, hide_index=True, use_container_width=True)

    with c2:
        st.write("**Prediksi ML**")
        pred_df = (
            filtered["prediksi_ml"]
            .replace("", "-")
            .value_counts()
            .reset_index()
        )
        pred_df.columns = ["Prediksi", "Jumlah"]
        st.dataframe(pred_df, hide_index=True, use_container_width=True)

    with c3:
        st.write("**Weakest Indicator**")
        weak_df = (
            filtered["weakest_indicator"]
            .replace("", "-")
            .value_counts()
            .reset_index()
        )
        weak_df.columns = ["Weakest", "Jumlah"]
        st.dataframe(weak_df, hide_index=True, use_container_width=True)


# =========================
# EXPORT
# =========================
st.subheader("Export Data")

tab1, tab2, tab3, tab4 = st.tabs([
    "Monitoring",
    "Pretest 0/1",
    "Posttest 0/1",
    "Jawaban Detail"
])

with tab1:
    st.write("Export data monitoring session siswa.")
    download_button_csv(
        "Download CSV Monitoring",
        filtered,
        "monitoring_siswa.csv"
    )

with tab2:
    pretest_matrix = build_item_matrix(jawaban_df, "pretest")

    if pretest_matrix.empty:
        st.info("Belum ada data jawaban pretest.")
    else:
        st.write(f"Jumlah siswa pada matrix pretest: **{len(pretest_matrix)}**")
        st.dataframe(pretest_matrix, use_container_width=True, hide_index=True, height=520)

        download_button_csv(
            "Download CSV Pretest 0/1",
            pretest_matrix,
            "data_uji_instrumen_pretest.csv"
        )

with tab3:
    posttest_matrix = build_item_matrix(jawaban_df, "posttest")

    if posttest_matrix.empty:
        st.info("Belum ada data jawaban posttest.")
    else:
        st.write(f"Jumlah siswa pada matrix posttest: **{len(posttest_matrix)}**")
        st.dataframe(posttest_matrix, use_container_width=True, hide_index=True, height=520)

        download_button_csv(
            "Download CSV Posttest 0/1",
            posttest_matrix,
            "data_uji_instrumen_posttest.csv"
        )

with tab4:
    if jawaban_df.empty:
        st.info("Belum ada data jawaban detail.")
    else:
        show_cols = [
            "nis", "nama", "kelas", "phase", "question_id",
            "ct", "level", "selected_answer", "correct_answer",
            "is_correct", "score_binary", "score_weighted",
            "attempt_order", "created_at"
        ]

        existing_show_cols = [c for c in show_cols if c in jawaban_df.columns]
        detail = jawaban_df[existing_show_cols].copy()

        st.write(f"Jumlah baris jawaban detail: **{len(detail)}**")
        st.dataframe(detail, use_container_width=True, hide_index=True, height=520)

        download_button_csv(
            "Download CSV Jawaban Detail",
            detail,
            "jawaban_siswa_detail.csv"
        )
