import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Dashboard Data Siswa", layout="wide")

st.title("Dashboard Rekap Data Siswa")
st.caption("Monitoring sinkronisasi worksheet Data_Siswa")

conn = st.connection("gsheets", type=GSheetsConnection)

EXPECTED_COLUMNS = [
    "record_id",
    "timestamp",
    "NIS",
    "nama",
    "kelas",
    "D_score",
    "P_score",
    "A_score",
    "Alg_score",
    "total_score",
    "weakest_indicator",
    "prediksi_ml",
    "treatment_materi",
    "treatment_level",
    "treatment_jumlah_soal",
    "treatment_status",
    "posttest_score",
    "gain_score",
    "status_selesai"
]


@st.cache_data(ttl=30)
def load_data(_conn):
    df = _conn.read(worksheet="Data_Siswa")

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df.columns = [str(c).strip() for c in df.columns]

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[EXPECTED_COLUMNS].copy()


try:
    data = load_data(conn)
except Exception as e:
    st.error(f"Gagal membaca Data_Siswa: {e}")
    st.stop()

if data.empty:
    st.warning("Data_Siswa masih kosong.")
    st.stop()

st.success(f"Data berhasil dimuat ✅ ({len(data)} baris)")

# =========================
# NORMALISASI ANGKA
# =========================
for col in [
    "D_score", "P_score", "A_score", "Alg_score",
    "total_score", "treatment_jumlah_soal", "posttest_score", "gain_score"
]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# =========================
# FILTER
# =========================
st.subheader("Filter Data")

c1, c2, c3, c4 = st.columns(4)

with c1:
    kelas_options = ["Semua"]
    if "kelas" in data.columns:
        kelas_values = data["kelas"].dropna().astype(str)
        kelas_options += sorted([x for x in kelas_values.unique().tolist() if x.strip() != ""])
    kelas_filter = st.selectbox("Filter Kelas", kelas_options)

with c2:
    pred_options = ["Semua"]
    if "prediksi_ml" in data.columns:
        pred_values = data["prediksi_ml"].dropna().astype(str)
        pred_options += sorted([x for x in pred_values.unique().tolist() if x.strip() != ""])
    pred_filter = st.selectbox("Filter Prediksi ML", pred_options)

with c3:
    treatment_status_options = ["Semua"]
    if "treatment_status" in data.columns:
        ts_values = data["treatment_status"].dropna().astype(str)
        treatment_status_options += sorted([x for x in ts_values.unique().tolist() if x.strip() != ""])
    treatment_status_filter = st.selectbox("Filter Treatment Status", treatment_status_options)

with c4:
    status_options = ["Semua"]
    if "status_selesai" in data.columns:
        ss_values = data["status_selesai"].dropna().astype(str)
        status_options += sorted([x for x in ss_values.unique().tolist() if x.strip() != ""])
    status_filter = st.selectbox("Filter Status Selesai", status_options)

filtered = data.copy()

if kelas_filter != "Semua":
    filtered = filtered[filtered["kelas"].astype(str) == kelas_filter]

if pred_filter != "Semua":
    filtered = filtered[filtered["prediksi_ml"].astype(str) == pred_filter]

if treatment_status_filter != "Semua":
    filtered = filtered[filtered["treatment_status"].astype(str) == treatment_status_filter]

if status_filter != "Semua":
    filtered = filtered[filtered["status_selesai"].astype(str) == status_filter]

# =========================
# SEARCH
# =========================
st.subheader("Pencarian")
keyword = st.text_input("Cari berdasarkan NIS atau Nama", value="").strip().lower()

if keyword:
    filtered = filtered[
        filtered["NIS"].astype(str).str.lower().str.contains(keyword, na=False) |
        filtered["nama"].astype(str).str.lower().str.contains(keyword, na=False)
    ]

# =========================
# METRICS
# =========================
st.subheader("Ringkasan")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Record", len(filtered))

with m2:
    if "total_score" in filtered.columns and not filtered["total_score"].isna().all():
        st.metric("Rata-rata Pretest", round(filtered["total_score"].mean(), 2))
    else:
        st.metric("Rata-rata Pretest", "-")

with m3:
    if "posttest_score" in filtered.columns and not filtered["posttest_score"].isna().all():
        st.metric("Rata-rata Posttest", round(filtered["posttest_score"].mean(), 2))
    else:
        st.metric("Rata-rata Posttest", "-")

with m4:
    if "gain_score" in filtered.columns and not filtered["gain_score"].isna().all():
        st.metric("Rata-rata Gain", round(filtered["gain_score"].mean(), 2))
    else:
        st.metric("Rata-rata Gain", "-")

with m5:
    if "treatment_jumlah_soal" in filtered.columns and not filtered["treatment_jumlah_soal"].isna().all():
        st.metric("Rata-rata Jumlah Treatment", round(filtered["treatment_jumlah_soal"].mean(), 2))
    else:
        st.metric("Rata-rata Jumlah Treatment", "-")

# =========================
# DISTRIBUSI
# =========================
c1, c2 = st.columns(2)

with c1:
    if "prediksi_ml" in filtered.columns:
        st.subheader("Distribusi Prediksi ML")
        pred_counts = filtered["prediksi_ml"].astype(str).value_counts()
        if len(pred_counts) > 0:
            st.bar_chart(pred_counts)

with c2:
    if "weakest_indicator" in filtered.columns:
        st.subheader("Distribusi Weakest Indicator")
        weak_counts = filtered["weakest_indicator"].astype(str).value_counts()
        if len(weak_counts) > 0:
            st.bar_chart(weak_counts)

c3, c4 = st.columns(2)

with c3:
    if "treatment_status" in filtered.columns:
        st.subheader("Distribusi Treatment Status")
        treat_counts = filtered["treatment_status"].astype(str).value_counts()
        if len(treat_counts) > 0:
            st.bar_chart(treat_counts)

with c4:
    if "status_selesai" in filtered.columns:
        st.subheader("Distribusi Status Selesai")
        selesai_counts = filtered["status_selesai"].astype(str).value_counts()
        if len(selesai_counts) > 0:
            st.bar_chart(selesai_counts)

# =========================
# TABEL DATA
# =========================
st.subheader("Tabel Data Siswa")

display_columns = [
    "record_id",
    "timestamp",
    "NIS",
    "nama",
    "kelas",
    "D_score",
    "P_score",
    "A_score",
    "Alg_score",
    "total_score",
    "weakest_indicator",
    "prediksi_ml",
    "treatment_materi",
    "treatment_level",
    "treatment_jumlah_soal",
    "treatment_status",
    "posttest_score",
    "gain_score",
    "status_selesai"
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True
)

# =========================
# DATA TERAKHIR
# =========================
st.subheader("10 Data Terakhir")

if "timestamp" in filtered.columns:
    preview = filtered.sort_values("timestamp", ascending=False).head(10)
else:
    preview = filtered.head(10)

st.dataframe(
    preview[display_columns],
    use_container_width=True,
    hide_index=True
)

# =========================
# DOWNLOAD CSV
# =========================
st.subheader("Export Data")

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV Hasil Filter",
    data=csv_data,
    file_name="rekap_data_siswa.csv",
    mime="text/csv"
)