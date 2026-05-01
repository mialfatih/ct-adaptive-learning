import pandas as pd
import streamlit as st

from utils import get_supabase

st.set_page_config(page_title="Monitoring CT Adaptive", layout="wide")

st.title("Dashboard Monitoring Siswa")
st.caption("Monitoring progres pretest, treatment, posttest, dan export data uji instrumen")

# =========================
# DATA LOAD
# =========================
@st.cache_data(ttl=15)
def load_data():
    supabase = get_supabase()

    siswa_res = supabase.table("siswa").select("*").execute()
    session_res = supabase.table("session_pembelajaran").select("*").execute()
    jawaban_res = supabase.table("jawaban_siswa").select("*").execute()

    siswa_df = pd.DataFrame(siswa_res.data or [])
    session_df = pd.DataFrame(session_res.data or [])
    jawaban_df = pd.DataFrame(jawaban_res.data or [])

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
    NIS | nama | kelas | Q1 | Q2 | ... | total_score
    Isi Q = 1 benar, 0 salah.
    """
    if jawaban_df.empty:
        return pd.DataFrame()

    df = jawaban_df[jawaban_df["phase"] == phase].copy()

    if df.empty:
        return pd.DataFrame()

    df["question_id"] = df["question_id"].astype(str)
    df["score_binary"] = pd.to_numeric(df["score_binary"], errors="coerce").fillna(0).astype(int)

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


try:
    data, jawaban_df = load_data()
except Exception as e:
    st.error(f"Gagal membaca data Supabase: {e}")
    st.stop()

if data.empty:
    st.warning("Belum ada data siswa.")
    st.stop()

# =========================
# NORMALISASI
# =========================
for col in ["nis", "nama", "kelas", "status_session", "treatment_status", "prediksi_ml", "weakest_indicator", "current_ct", "current_level"]:
    if col in data.columns:
        data[col] = clean_text(data[col])

for col in ["d_score", "p_score", "a_score", "alg_score", "total_score", "answered_count", "posttest_score", "gain_score"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

data["status_label"] = data.apply(make_status, axis=1)

if "updated_at" in data.columns:
    data["updated_at_dt"] = pd.to_datetime(data["updated_at"], errors="coerce")
else:
    data["updated_at_dt"] = pd.NaT

# =========================
# FILTER
# =========================
with st.container(border=True):
    st.subheader("Filter")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kelas_list = ["Semua"] + sorted([x for x in data["kelas"].dropna().unique().tolist() if x])
        kelas_filter = st.selectbox("Kelas", kelas_list)

    with c2:
        status_list = ["Semua"] + sorted([x for x in data["status_label"].dropna().unique().tolist() if x])
        status_filter = st.selectbox("Status", status_list)

    with c3:
        pred_list = ["Semua"] + sorted([x for x in data["prediksi_ml"].dropna().unique().tolist() if x])
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
# RINGKASAN DISTRIBUSI RINGKAS
# =========================
with st.expander("Lihat Ringkasan Distribusi"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Status**")
        st.dataframe(
            filtered["status_label"].value_counts().reset_index().rename(
                columns={"status_label": "Jumlah", "index": "Status"}
            ),
            hide_index=True,
            use_container_width=True
        )

    with c2:
        st.write("**Prediksi ML**")
        st.dataframe(
            filtered["prediksi_ml"].replace("", "-").value_counts().reset_index().rename(
                columns={"prediksi_ml": "Jumlah", "index": "Prediksi"}
            ),
            hide_index=True,
            use_container_width=True
        )

    with c3:
        st.write("**Weakest Indicator**")
        st.dataframe(
            filtered["weakest_indicator"].replace("", "-").value_counts().reset_index().rename(
                columns={"weakest_indicator": "Jumlah", "index": "Weakest"}
            ),
            hide_index=True,
            use_container_width=True
        )

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
    csv_monitor = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV Monitoring",
        data=csv_monitor,
        file_name="monitoring_siswa.csv",
        mime="text/csv"
    )

with tab2:
    pretest_matrix = build_item_matrix(jawaban_df, "pretest")
    if pretest_matrix.empty:
        st.info("Belum ada data jawaban pretest.")
    else:
        st.dataframe(pretest_matrix, use_container_width=True, hide_index=True)
        csv_pretest = pretest_matrix.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Pretest 0/1",
            data=csv_pretest,
            file_name="data_uji_instrumen_pretest.csv",
            mime="text/csv"
        )

with tab3:
    posttest_matrix = build_item_matrix(jawaban_df, "posttest")
    if posttest_matrix.empty:
        st.info("Belum ada data jawaban posttest.")
    else:
        st.dataframe(posttest_matrix, use_container_width=True, hide_index=True)
        csv_posttest = posttest_matrix.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Posttest 0/1",
            data=csv_posttest,
            file_name="data_uji_instrumen_posttest.csv",
            mime="text/csv"
        )

with tab4:
    if jawaban_df.empty:
        st.info("Belum ada data jawaban detail.")
    else:
        show_cols = [
            "nis", "nama", "kelas", "phase", "question_id",
            "ct", "level", "selected_answer", "correct_answer",
            "is_correct", "score_binary", "score_weighted", "attempt_order", "created_at"
        ]
        existing_show_cols = [c for c in show_cols if c in jawaban_df.columns]
        detail = jawaban_df[existing_show_cols].copy()

        st.dataframe(detail, use_container_width=True, hide_index=True, height=500)

        csv_detail = detail.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Jawaban Detail",
            data=csv_detail,
            file_name="jawaban_siswa_detail.csv",
            mime="text/csv"
        )
