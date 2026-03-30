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


def clean_text_series(series):
    return series.fillna("").astype(str).str.strip()


def get_clean_value_counts(series):
    s = clean_text_series(series)
    s = s[s != ""]
    return s.value_counts()


try:
    data = load_data(conn)
except Exception as e:
    st.error(f"Gagal membaca Data_Siswa: {e}")
    st.stop()

if data.empty:
    st.warning("Data_Siswa masih kosong.")
    st.stop()

# =========================
# NORMALISASI DATA
# =========================
for col in [
    "record_id", "timestamp", "NIS", "nama", "kelas",
    "weakest_indicator", "prediksi_ml", "treatment_materi",
    "treatment_level", "treatment_status", "status_selesai"
]:
    if col in data.columns:
        data[col] = clean_text_series(data[col])

for col in [
    "D_score", "P_score", "A_score", "Alg_score",
    "total_score", "treatment_jumlah_soal", "posttest_score", "gain_score"
]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

if "timestamp" in data.columns:
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
else:
    data["timestamp_dt"] = pd.NaT

st.success(f"Data berhasil dimuat ✅ ({len(data)} baris)")

# =========================
# FILTER
# =========================
st.subheader("Filter Data")

c1, c2, c3, c4 = st.columns(4)

with c1:
    kelas_options = ["Semua"]
    if "kelas" in data.columns:
        kelas_values = clean_text_series(data["kelas"])
        kelas_options += sorted([x for x in kelas_values.unique().tolist() if x != ""])
    kelas_filter = st.selectbox("Filter Kelas", kelas_options)

with c2:
    pred_options = ["Semua"]
    if "prediksi_ml" in data.columns:
        pred_values = clean_text_series(data["prediksi_ml"])
        pred_options += sorted([x for x in pred_values.unique().tolist() if x != ""])
    pred_filter = st.selectbox("Filter Prediksi ML", pred_options)

with c3:
    treatment_status_options = ["Semua"]
    if "treatment_status" in data.columns:
        ts_values = clean_text_series(data["treatment_status"])
        treatment_status_options += sorted([x for x in ts_values.unique().tolist() if x != ""])
    treatment_status_filter = st.selectbox("Filter Treatment Status", treatment_status_options)

with c4:
    status_options = ["Semua"]
    if "status_selesai" in data.columns:
        ss_values = clean_text_series(data["status_selesai"])
        status_options += sorted([x for x in ss_values.unique().tolist() if x != ""])
    status_filter = st.selectbox("Filter Status Selesai", status_options)

filtered = data.copy()

if kelas_filter != "Semua":
    filtered = filtered[filtered["kelas"] == kelas_filter]

if pred_filter != "Semua":
    filtered = filtered[filtered["prediksi_ml"] == pred_filter]

if treatment_status_filter != "Semua":
    filtered = filtered[filtered["treatment_status"] == treatment_status_filter]

if status_filter != "Semua":
    filtered = filtered[filtered["status_selesai"] == status_filter]

# =========================
# SEARCH
# =========================
st.subheader("Pencarian")
keyword = st.text_input("Cari berdasarkan NIS atau Nama", value="").strip().lower()

if keyword:
    filtered = filtered[
        filtered["NIS"].str.lower().str.contains(keyword, na=False) |
        filtered["nama"].str.lower().str.contains(keyword, na=False)
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
    st.subheader("Distribusi Prediksi ML")
    pred_counts = get_clean_value_counts(filtered["prediksi_ml"])
    if len(pred_counts) > 0:
        st.bar_chart(pred_counts)
    else:
        st.info("Belum ada data prediksi ML yang bisa ditampilkan.")

with c2:
    st.subheader("Distribusi Weakest Indicator")
    weak_counts = get_clean_value_counts(filtered["weakest_indicator"])
    if len(weak_counts) > 0:
        st.bar_chart(weak_counts)
    else:
        st.info("Belum ada data weakest indicator yang bisa ditampilkan.")

c3, c4 = st.columns(2)

with c3:
    st.subheader("Distribusi Treatment Status")
    treat_counts = get_clean_value_counts(filtered["treatment_status"])
    if len(treat_counts) > 0:
        st.bar_chart(treat_counts)
    else:
        st.info("Belum ada data treatment status yang bisa ditampilkan.")

with c4:
    st.subheader("Distribusi Status Selesai")
    selesai_counts = get_clean_value_counts(filtered["status_selesai"])
    if len(selesai_counts) > 0:
        st.bar_chart(selesai_counts)
    else:
        st.info("Belum ada data status selesai yang bisa ditampilkan.")

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

try:
    if "timestamp_dt" in filtered.columns:
        preview = filtered.sort_values("timestamp_dt", ascending=False).head(10)
    else:
        preview = filtered.head(10)
except Exception as e:
    st.warning(f"Gagal mengurutkan data berdasarkan timestamp: {e}")
    preview = filtered.head(10)

st.dataframe(
    preview[display_columns],
    use_container_width=True,
    hide_index=True
)

# =========================
# EXPORT CSV
# =========================
st.subheader("Export Data")

export_df = filtered[display_columns].copy()
csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Hasil Filter",
    data=csv_data,
    file_name="rekap_data_siswa.csv",
    mime="text/csv"
)
