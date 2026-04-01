import pandas as pd
import streamlit as st

from utils import get_supabase

st.set_page_config(page_title="Dashboard Monitoring Siswa", layout="wide")

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .main-card {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(200,200,200,0.18);
        background: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    .small-muted {
        font-size: 0.85rem;
        opacity: 0.8;
    }
    .student-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(200,200,200,0.15);
        background: rgba(255,255,255,0.03);
        margin-bottom: 0.8rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dashboard Monitoring Siswa")
st.caption("Monitoring progres pretest, treatment, dan posttest berbasis Supabase")

# =========================
# DATA LOAD
# =========================
@st.cache_data(ttl=20)
def load_monitoring_data():
    supabase = get_supabase()

    siswa_res = supabase.table("siswa").select("*").execute()
    session_res = supabase.table("session_pembelajaran").select("*").execute()

    siswa_df = pd.DataFrame(siswa_res.data or [])
    session_df = pd.DataFrame(session_res.data or [])

    if siswa_df.empty:
        siswa_df = pd.DataFrame(columns=["id", "nis", "nama", "kelas", "created_at", "updated_at"])

    if session_df.empty:
        session_df = pd.DataFrame(columns=[
            "id", "siswa_id", "status_session", "pretest_selesai", "posttest_selesai",
            "treatment_status", "d_score", "p_score", "a_score", "alg_score", "total_score",
            "weakest_indicator", "prediksi_ml", "answered_count", "current_ct", "current_level",
            "points", "current_ct_idx", "priority_order_json", "start_level_map_json",
            "mastered_ct_json", "history_ids_json", "served_items_json", "project_ready",
            "posttest_score", "gain_score", "created_at", "updated_at", "finished_at"
        ])

    if not siswa_df.empty and not session_df.empty:
        df = session_df.merge(
            siswa_df,
            left_on="siswa_id",
            right_on="id",
            how="left",
            suffixes=("_session", "_siswa")
        )
    else:
        df = session_df.copy()

    return df


def clean_text_series(series):
    return series.fillna("").astype(str).str.strip()


def to_datetime_safe(series):
    return pd.to_datetime(series, errors="coerce")


def make_status_label(row):
    status_session = str(row.get("status_session", "")).strip()
    treatment_status = str(row.get("treatment_status", "")).strip()

    if status_session == "selesai":
        return "Selesai"
    if status_session == "posttest":
        return "Posttest"
    if status_session == "treatment":
        if treatment_status == "skip":
            return "Skip Treatment"
        return "Treatment"
    if status_session == "pretest":
        return "Pretest"
    return "Belum Mulai"


def get_status_html(label: str):
    label_lower = label.lower()

    if "selesai" in label_lower:
        bg = "rgba(34,197,94,0.18)"
        fg = "#22c55e"
    elif "posttest" in label_lower:
        bg = "rgba(59,130,246,0.18)"
        fg = "#60a5fa"
    elif "skip" in label_lower:
        bg = "rgba(245,158,11,0.18)"
        fg = "#f59e0b"
    elif "treatment" in label_lower:
        bg = "rgba(168,85,247,0.18)"
        fg = "#c084fc"
    else:
        bg = "rgba(156,163,175,0.18)"
        fg = "#d1d5db"

    return f'<span class="status-pill" style="background:{bg}; color:{fg};">{label}</span>'


def get_clean_value_counts(series):
    s = clean_text_series(series)
    s = s[s != ""]
    return s.value_counts()


# =========================
# LOAD
# =========================
try:
    data = load_monitoring_data()
except Exception as e:
    st.error(f"Gagal membaca data Supabase: {e}")
    st.stop()

if data.empty:
    st.warning("Belum ada data siswa / sesi pembelajaran.")
    st.stop()

# =========================
# NORMALISASI
# =========================
text_cols = [
    "nis", "nama", "kelas", "status_session", "treatment_status",
    "prediksi_ml", "weakest_indicator", "current_ct", "current_level"
]
for col in text_cols:
    if col in data.columns:
        data[col] = clean_text_series(data[col])

num_cols = [
    "d_score", "p_score", "a_score", "alg_score",
    "total_score", "answered_count", "points",
    "posttest_score", "gain_score"
]
for col in num_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

for col in ["created_at_session", "updated_at_session", "finished_at"]:
    if col in data.columns:
        data[f"{col}_dt"] = to_datetime_safe(data[col])

if "updated_at" in data.columns and "updated_at_dt" not in data.columns:
    data["updated_at_dt"] = to_datetime_safe(data["updated_at"])

data["status_label"] = data.apply(make_status_label, axis=1)

# =========================
# HEADER SUMMARY
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Ringkasan Monitoring")

total_siswa = data["siswa_id"].nunique() if "siswa_id" in data.columns else 0
total_session = len(data)
sedang_aktif = len(data[data["status_session"] != "selesai"]) if "status_session" in data.columns else 0
sudah_selesai = len(data[data["status_session"] == "selesai"]) if "status_session" in data.columns else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Siswa", total_siswa)
with m2:
    st.metric("Total Session", total_session)
with m3:
    st.metric("Sedang Aktif", sedang_aktif)
with m4:
    st.metric("Selesai", sudah_selesai)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FILTER
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Filter dan Pencarian")

c1, c2, c3, c4 = st.columns(4)

with c1:
    kelas_options = ["Semua"]
    if "kelas" in data.columns:
        kelas_values = clean_text_series(data["kelas"])
        kelas_options += sorted([x for x in kelas_values.unique().tolist() if x != ""])
    kelas_filter = st.selectbox("Filter Kelas", kelas_options)

with c2:
    status_options = ["Semua"]
    if "status_label" in data.columns:
        status_values = clean_text_series(data["status_label"])
        status_options += sorted([x for x in status_values.unique().tolist() if x != ""])
    status_filter = st.selectbox("Filter Status", status_options)

with c3:
    prediksi_options = ["Semua"]
    if "prediksi_ml" in data.columns:
        pred_values = clean_text_series(data["prediksi_ml"])
        prediksi_options += sorted([x for x in pred_values.unique().tolist() if x != ""])
    prediksi_filter = st.selectbox("Filter Prediksi ML", prediksi_options)

with c4:
    treatment_options = ["Semua"]
    if "treatment_status" in data.columns:
        treatment_values = clean_text_series(data["treatment_status"])
        treatment_options += sorted([x for x in treatment_values.unique().tolist() if x != ""])
    treatment_filter = st.selectbox("Filter Treatment", treatment_options)

keyword = st.text_input("Cari berdasarkan NIS atau Nama", value="").strip().lower()
st.markdown("</div>", unsafe_allow_html=True)

filtered = data.copy()

if kelas_filter != "Semua":
    filtered = filtered[filtered["kelas"] == kelas_filter]

if status_filter != "Semua":
    filtered = filtered[filtered["status_label"] == status_filter]

if prediksi_filter != "Semua":
    filtered = filtered[filtered["prediksi_ml"] == prediksi_filter]

if treatment_filter != "Semua":
    filtered = filtered[filtered["treatment_status"] == treatment_filter]

if keyword:
    filtered = filtered[
        filtered["nis"].str.lower().str.contains(keyword, na=False) |
        filtered["nama"].str.lower().str.contains(keyword, na=False)
    ]

# =========================
# LIVE OVERVIEW
# =========================
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Status Siswa Saat Ini")

    status_counts = get_clean_value_counts(filtered["status_label"])
    if len(status_counts) > 0:
        st.bar_chart(status_counts)
    else:
        st.info("Belum ada data status.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Distribusi Prediksi CT")

    pred_counts = get_clean_value_counts(filtered["prediksi_ml"])
    if len(pred_counts) > 0:
        st.bar_chart(pred_counts)
    else:
        st.info("Belum ada data prediksi.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LEADERBOARD
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Leaderboard Hasil Akhir")

leaderboard = filtered.copy()

if "posttest_score" in leaderboard.columns:
    leaderboard = leaderboard.sort_values(
        by=["posttest_score", "gain_score"],
        ascending=[False, False],
        na_position="last"
    )

leaderboard_display = leaderboard[[
    "nis",
    "nama",
    "kelas",
    "prediksi_ml",
    "total_score",
    "posttest_score",
    "gain_score",
    "status_label"
]].copy()

leaderboard_display.columns = [
    "NIS",
    "Nama",
    "Kelas",
    "Prediksi ML",
    "Pretest",
    "Posttest",
    "Gain",
    "Status"
]

st.dataframe(leaderboard_display.head(10), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# KARTU MONITORING SISWA
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Monitoring Per Siswa")

if "updated_at_dt" in filtered.columns:
    filtered_cards = filtered.sort_values("updated_at_dt", ascending=False, na_position="last")
else:
    filtered_cards = filtered.copy()

if filtered_cards.empty:
    st.info("Tidak ada data sesuai filter.")
else:
    max_cards = min(len(filtered_cards), 30)

    for _, row in filtered_cards.head(max_cards).iterrows():
        nama = row.get("nama", "-")
        nis = row.get("nis", "-")
        kelas = row.get("kelas", "-")
        status_label = row.get("status_label", "-")
        prediksi = row.get("prediksi_ml", "-")
        weakest = row.get("weakest_indicator", "-")
        current_ct = row.get("current_ct", "-")
        current_level = row.get("current_level", "-")
        answered_count = row.get("answered_count", 0)
        total_score = row.get("total_score", 0)
        posttest_score = row.get("posttest_score", 0)
        gain_score = row.get("gain_score", 0)

        with st.container():
            st.markdown('<div class="student-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 1.2, 1.2])

            with c1:
                st.markdown(f"**{nama}**")
                st.markdown(f'<div class="small-muted">NIS: {nis} | Kelas: {kelas}</div>', unsafe_allow_html=True)
                st.markdown(get_status_html(status_label), unsafe_allow_html=True)

            with c2:
                st.write(f"**Prediksi:** {prediksi if prediksi else '-'}")
                st.write(f"**Weakest:** {weakest if weakest else '-'}")
                st.write(f"**Soal Treatment:** {int(answered_count) if pd.notna(answered_count) else 0}")

            with c3:
                st.write(f"**CT Saat Ini:** {current_ct if current_ct else '-'}")
                st.write(f"**Level Saat Ini:** {str(current_level).capitalize() if current_level else '-'}")
                st.write(f"**Gain:** {int(gain_score) if pd.notna(gain_score) else 0}")

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Pretest", int(total_score) if pd.notna(total_score) else 0)
            with mc2:
                st.metric("Posttest", int(posttest_score) if pd.notna(posttest_score) else 0)
            with mc3:
                st.metric("Progress Treatment", int(answered_count) if pd.notna(answered_count) else 0)

            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TABEL DETAIL
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Tabel Detail Session")

detail_columns = [
    "nis",
    "nama",
    "kelas",
    "status_label",
    "prediksi_ml",
    "weakest_indicator",
    "d_score",
    "p_score",
    "a_score",
    "alg_score",
    "total_score",
    "current_ct",
    "current_level",
    "answered_count",
    "posttest_score",
    "gain_score",
    "updated_at"
]

existing_detail_columns = [col for col in detail_columns if col in filtered.columns]
detail_df = filtered[existing_detail_columns].copy()

rename_map = {
    "nis": "NIS",
    "nama": "Nama",
    "kelas": "Kelas",
    "status_label": "Status",
    "prediksi_ml": "Prediksi ML",
    "weakest_indicator": "Weakest",
    "d_score": "D",
    "p_score": "P",
    "a_score": "A",
    "alg_score": "Alg",
    "total_score": "Pretest",
    "current_ct": "CT Saat Ini",
    "current_level": "Level Saat Ini",
    "answered_count": "Jumlah Treatment",
    "posttest_score": "Posttest",
    "gain_score": "Gain",
    "updated_at": "Last Update"
}
detail_df = detail_df.rename(columns=rename_map)

st.dataframe(detail_df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# EXPORT
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Export Data")

export_df = filtered.copy()
csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Monitoring",
    data=csv_data,
    file_name="monitoring_session_pembelajaran.csv",
    mime="text/csv"
)
st.markdown("</div>", unsafe_allow_html=True)
