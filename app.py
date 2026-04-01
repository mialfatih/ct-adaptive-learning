import streamlit as st
from streamlit_gsheets import GSheetsConnection

from utils import (
    load_model,
    load_bank_soal,
    get_phase_questions,
    build_options,
    calculate_pretest_scores,
    calculate_posttest_score,
    predict_overall_level,
    init_treatment_state,
    level_target,
    advance_state,
    fetch_question,
    build_result_row,
    save_student_result,
    load_pretest_siswa,
    find_existing_pretest,
    build_pretest_row,
    save_pretest_result
)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="CT Adaptive Learning",
    layout="centered"
)

MODEL_PATH = "models/knn_ct_model_k11.pkl"

# =========================
# CONNECTION
# =========================
conn = st.connection("gsheets", type=GSheetsConnection)

# =========================
# LOAD RESOURCES
# =========================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

try:
    bank = load_bank_soal(conn)
except Exception as e:
    st.error(f"Gagal load Bank_Soal dari Google Sheets: {e}")
    st.stop()

if bank.empty:
    st.error("Bank_Soal kosong atau belum terbaca.")
    st.stop()

# =========================
# SESSION DEFAULTS
# =========================
defaults = {
    "stage": "identitas",
    "student_profile": None,
    "pretest_df": None,
    "posttest_df": None,
    "treatment_state": None,
    "current_question": None,
    "pretest_answers": {},
    "posttest_answers": {},
    "final_result": None,
    "saved_to_db": False,
    "treatment_status": "selesai"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    for k, v in defaults.items():
        st.session_state[k] = v


# =========================
# SMALL UI HELPERS
# =========================
def render_header():
    st.title("CT Adaptive Learning")
    st.caption("Sistem pembelajaran adaptif berbasis pretest, treatment, dan posttest")


def render_student_box():
    prof = st.session_state.get("student_profile")
    if not prof:
        return

    with st.container(border=True):
        st.markdown("**Data Siswa**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**NIS/ID:** {prof.get('student_id', '-')}")
        with c2:
            st.write(f"**Nama:** {prof.get('student_name', '-')}")
        with c3:
            st.write(f"**Kelas:** {prof.get('student_class', '-')}")


def render_progress(answer_dict: dict, total_questions: int, label: str):
    answered = sum(1 for v in answer_dict.values() if str(v).strip() != "")
    progress = answered / total_questions if total_questions > 0 else 0
    st.progress(progress, text=f"{label}: {answered} dari {total_questions} soal terjawab")


def render_stage_badge(stage_name: str):
    st.info(f"Tahap saat ini: **{stage_name}**")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Menu")
    st.write(f"**Tahap:** {st.session_state['stage']}")

    prof = st.session_state.get("student_profile")
    if prof:
        st.write("---")
        st.write(f"**NIS:** {prof.get('student_id', '-')}")
        st.write(f"**Nama:** {prof.get('student_name', '-')}")
        st.write(f"**Kelas:** {prof.get('student_class', '-')}")

    st.write("---")
    if st.button("Reset Sesi"):
        reset_all()
        st.rerun()


# =========================
# MAIN HEADER
# =========================
render_header()

# =========================
# STAGE 1: IDENTITAS
# =========================
if st.session_state["stage"] == "identitas":
    render_stage_badge("Identitas")
    st.subheader("Isi Identitas Siswa")

    with st.container(border=True):
        st.write("Silakan isi data diri terlebih dahulu sebelum memulai pretest.")

        with st.form("form_identitas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                student_id = st.text_input("NIS / ID")
            with c2:
                student_name = st.text_input("Nama")
            with c3:
                student_class = st.text_input("Kelas")

            submitted = st.form_submit_button("Mulai Pretest", type="primary")

    if submitted:
        if not student_id.strip():
            st.warning("NIS / ID wajib diisi.")
            st.stop()

        if not student_name.strip():
            st.warning("Nama wajib diisi.")
            st.stop()

        pretest_df = get_phase_questions(bank, "pretest")
        posttest_df = get_phase_questions(bank, "posttest")

        if pretest_df.empty:
            st.error("Soal pretest tidak ditemukan di Bank_Soal.")
            st.stop()

        if posttest_df.empty:
            st.error("Soal posttest tidak ditemukan di Bank_Soal.")
            st.stop()

        # cek apakah siswa sudah pernah menyelesaikan pretest
        existing_pretest = None
        try:
            pretest_saved_df = load_pretest_siswa(conn)
            existing_pretest = find_existing_pretest(
                pretest_saved_df,
                student_id.strip(),
                student_name.strip()
            )
        except Exception as e:
            st.warning(f"Gagal memeriksa data pretest sebelumnya: {e}")

        # kalau ditemukan pretest lama, skip pretest
        if existing_pretest is not None:
            scores = {
                "D": int(existing_pretest["D_score"]),
                "P": int(existing_pretest["P_score"]),
                "A": int(existing_pretest["A_score"]),
                "Alg": int(existing_pretest["Alg_score"])
            }

            st.session_state["student_profile"] = {
                "student_id": str(existing_pretest["NIS"]).strip(),
                "student_name": str(existing_pretest["nama"]).strip(),
                "student_class": str(existing_pretest["kelas"]).strip(),
                "scores": scores,
                "total": int(existing_pretest["total_score"]),
                "overall": str(existing_pretest["prediksi_ml"]).strip(),
                "weak_indicator": str(existing_pretest["weakest_indicator"]).strip(),
                "needs": [k for k, v in scores.items() if v < 7]
            }

            st.session_state["pretest_df"] = pretest_df
            st.session_state["posttest_df"] = posttest_df
            st.session_state["pretest_answers"] = {}
            st.session_state["posttest_answers"] = {}
            st.session_state["treatment_state"] = init_treatment_state(scores)
            st.session_state["current_question"] = None
            st.session_state["final_result"] = None
            st.session_state["saved_to_db"] = False
            st.session_state["treatment_status"] = "selesai"
            st.session_state["stage"] = "hasil_pretest"

            st.success("Data pretest sebelumnya ditemukan. Kamu bisa melanjutkan ke treatment.")
            st.rerun()

        # kalau belum ada, lanjut normal ke pretest
        st.session_state["student_profile"] = {
            "student_id": student_id.strip(),
            "student_name": student_name.strip(),
            "student_class": student_class.strip()
        }
        st.session_state["pretest_df"] = pretest_df
        st.session_state["posttest_df"] = posttest_df
        st.session_state["pretest_answers"] = {}
        st.session_state["posttest_answers"] = {}
        st.session_state["treatment_state"] = None
        st.session_state["current_question"] = None
        st.session_state["final_result"] = None
        st.session_state["saved_to_db"] = False
        st.session_state["treatment_status"] = "selesai"
        st.session_state["stage"] = "pretest"
        st.rerun()

# =========================
# STAGE 2: PRETEST
# =========================
elif st.session_state["stage"] == "pretest":
    render_stage_badge("Pretest")
    render_student_box()

    pretest_df = st.session_state["pretest_df"]
    total_questions = len(pretest_df)

    st.subheader("Pretest")
    st.write("Jawablah semua soal berikut dengan teliti.")
    render_progress(st.session_state["pretest_answers"], total_questions, "Progress Pretest")

    with st.form("form_pretest"):
        for i, (_, row) in enumerate(pretest_df.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(f"**Soal {i}**")
                st.write(row["question"])

                options, labels = build_options(row)
                current_value = st.session_state["pretest_answers"].get(row["id"], "")
                select_options = [""] + options

                choice = st.selectbox(
                    f"Pilih jawaban soal {i}",
                    select_options,
                    index=select_options.index(current_value) if current_value in select_options else 0,
                    format_func=lambda x, options=options, labels=labels: "— Pilih jawaban —" if x == "" else labels[options.index(x)],
                    key=f"pre_{row['id']}"
                )

                st.session_state["pretest_answers"][row["id"]] = choice

        submitted = st.form_submit_button("Submit Pretest", type="primary")

    if submitted:
        answered_ids = {
            qid for qid, ans in st.session_state["pretest_answers"].items()
            if str(ans).strip() != ""
        }
        all_ids = set(pretest_df["id"].astype(str).tolist())
        missing = list(all_ids - answered_ids)

        if missing:
            st.warning("Masih ada soal pretest yang belum dijawab.")
            st.stop()

        result = calculate_pretest_scores(pretest_df, st.session_state["pretest_answers"])
        scores = result["scores"]
        total_score = result["total_score"]
        weakest_indicator = result["weakest_indicator"]
        needs = result["needs"]

        overall = predict_overall_level(
            model,
            scores["D"], scores["P"], scores["A"], scores["Alg"]
        )

        st.session_state["student_profile"].update({
            "scores": scores,
            "total": total_score,
            "overall": overall,
            "weak_indicator": weakest_indicator,
            "needs": needs
        })

        # simpan hasil pretest ke sheet Pretest_Siswa
        try:
            pretest_row = build_pretest_row(st.session_state["student_profile"])
            save_pretest_result(conn, pretest_row)
        except Exception as e:
            st.warning(f"Hasil pretest belum berhasil disimpan ke Pretest_Siswa: {e}")

        st.session_state["treatment_state"] = init_treatment_state(scores)
        st.session_state["current_question"] = None
        st.session_state["treatment_status"] = "selesai"
        st.session_state["stage"] = "hasil_pretest"
        st.rerun()

# =========================
# STAGE 3: HASIL PRETEST
# =========================
elif st.session_state["stage"] == "hasil_pretest":
    render_stage_badge("Hasil Pretest")
    render_student_box()

    prof = st.session_state["student_profile"]

    st.subheader("Pretest Selesai")
    st.success("Sistem telah memproses hasil pretest kamu.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dekomposisi", prof["scores"]["D"])
    with c2:
        st.metric("Pola", prof["scores"]["P"])
    with c3:
        st.metric("Abstraksi", prof["scores"]["A"])
    with c4:
        st.metric("Algoritma", prof["scores"]["Alg"])

    c5, c6 = st.columns(2)
    with c5:
        st.metric("Total Pretest", prof["total"])
    with c6:
        st.metric("Prediksi Level", prof["overall"].capitalize())

    st.write(f"**Fokus awal pembelajaran:** {prof['weak_indicator']}")

    with st.container(border=True):
        st.write("Tahap berikutnya adalah treatment adaptif. Sistem akan menyesuaikan soal pembelajaran berdasarkan hasil pretest kamu.")

    if st.button("Lanjut ke Treatment", type="primary"):
        st.session_state["stage"] = "treatment"
        st.rerun()

# =========================
# STAGE 4: TREATMENT
# =========================
elif st.session_state["stage"] == "treatment":
    render_stage_badge("Treatment")
    render_student_box()

    state = st.session_state["treatment_state"]

    if "answered_count" not in state:
        state["answered_count"] = 0

    st.subheader("Treatment Adaptif")
    st.write("Kerjakan soal treatment berikut. Jika waktu treatment habis, guru dapat mengarahkan untuk melewati tahap ini.")

    if state["project_ready"]:
        st.success("Treatment selesai. Kamu bisa lanjut ke posttest.")
        if st.button("Lanjut ke Posttest", type="primary"):
            st.session_state["stage"] = "posttest"
            st.rerun()
        st.stop()

    if st.session_state["current_question"] is None:
        q = fetch_question(bank, state["current_ct"], state["current_level"], state["history_ids"])
        st.session_state["current_question"] = q

    q = st.session_state["current_question"]

    if q is None:
        st.error("Tidak ada soal treatment yang cocok di Bank_Soal.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fokus CT", state["current_ct"])
    with c2:
        st.metric("Level Soal", state["current_level"].capitalize())
    with c3:
        st.metric("Jumlah Soal Treatment", state.get("answered_count", 0))

    with st.container(border=True):
        st.write(f"**Materi:** {q['materi']}")
        st.write(f"**Soal:** {q['question']}")

        options, labels = build_options(q)
        treat_key = f"treat_{q['id']}"
        current_value = st.session_state.get(treat_key, "")
        select_options = [""] + options

        choice = st.selectbox(
            "Pilih jawaban",
            select_options,
            index=select_options.index(current_value) if current_value in select_options else 0,
            format_func=lambda x, options=options, labels=labels: "— Pilih jawaban —" if x == "" else labels[options.index(x)],
            key=treat_key
        )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Submit Jawaban Treatment", type="primary", use_container_width=True):
            if choice == "":
                st.warning("Pilih jawaban dulu.")
                st.stop()

            correct = (choice == q["answer"])

            state["answered_count"] = state.get("answered_count", 0) + 1

            if q["id"] not in state["history_ids"]:
                state["history_ids"].append(q["id"])

            state["served_items"].append({
                "id": q["id"],
                "materi": q["materi"],
                "ct": q["ct"],
                "level": q["level"]
            })

            if correct:
                state["points"] += 1
                st.success("Jawaban benar.")
            else:
                if state["points"] > 0:
                    state["points"] -= 1
                st.error("Jawaban belum tepat.")

            if state["points"] >= level_target(state["current_level"]):
                state, msg = advance_state(state)
                st.info(msg)

            st.session_state["treatment_state"] = state
            st.session_state["current_question"] = None
            st.rerun()

    with c2:
        if st.button("Lewati Treatment", use_container_width=True):
            st.session_state["treatment_status"] = "skip"
            st.session_state["stage"] = "posttest"
            st.rerun()

# =========================
# STAGE 5: POSTTEST
# =========================
elif st.session_state["stage"] == "posttest":
    render_stage_badge("Posttest")
    render_student_box()

    posttest_df = st.session_state["posttest_df"]
    total_questions = len(posttest_df)

    st.subheader("Posttest")
    st.write("Jawablah semua soal posttest berikut.")
    render_progress(st.session_state["posttest_answers"], total_questions, "Progress Posttest")

    if st.session_state.get("treatment_status", "selesai") == "skip":
        st.warning("Treatment sebelumnya dilewati, tetapi kamu tetap bisa melanjutkan ke posttest.")

    with st.form("form_posttest"):
        for i, (_, row) in enumerate(posttest_df.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(f"**Soal {i}**")
                st.write(row["question"])

                options, labels = build_options(row)
                current_value = st.session_state["posttest_answers"].get(row["id"], "")
                select_options = [""] + options

                choice = st.selectbox(
                    f"Pilih jawaban soal {i}",
                    select_options,
                    index=select_options.index(current_value) if current_value in select_options else 0,
                    format_func=lambda x, options=options, labels=labels: "— Pilih jawaban —" if x == "" else labels[options.index(x)],
                    key=f"post_{row['id']}"
                )

                st.session_state["posttest_answers"][row["id"]] = choice

        submitted = st.form_submit_button("Submit Posttest & Simpan Hasil", type="primary")

    if submitted:
        if st.session_state.get("saved_to_db", False):
            st.warning("Data sesi ini sudah pernah disimpan.")
            st.stop()

        answered_ids = {
            qid for qid, ans in st.session_state["posttest_answers"].items()
            if str(ans).strip() != ""
        }
        all_ids = set(posttest_df["id"].astype(str).tolist())
        missing = list(all_ids - answered_ids)

        if missing:
            st.warning("Masih ada soal posttest yang belum dijawab.")
            st.stop()

        posttest_score = calculate_posttest_score(
            posttest_df,
            st.session_state["posttest_answers"]
        )

        result_row = build_result_row(
            profile=st.session_state["student_profile"],
            treatment_state=st.session_state["treatment_state"],
            posttest_score=posttest_score,
            treatment_status=st.session_state.get("treatment_status", "selesai")
        )

        try:
            saved_row = save_student_result(conn, result_row)
            st.session_state["final_result"] = saved_row
            st.session_state["saved_to_db"] = True
            st.session_state["stage"] = "final"
            st.rerun()
        except Exception as e:
            st.error(f"Gagal menyimpan ke Data_Siswa: {e}")

# =========================
# STAGE 6: FINAL
# =========================
elif st.session_state["stage"] == "final":
    render_stage_badge("Selesai")
    render_student_box()

    result = st.session_state["final_result"]

    if not result:
        st.error("Hasil akhir tidak ditemukan.")
        st.stop()

    st.subheader("Terima Kasih")
    st.success("Data kamu berhasil disimpan.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Pretest", result["total_score"])
    with c2:
        st.metric("Posttest", result["posttest_score"])
    with c3:
        st.metric("Gain", result["gain_score"])
    with c4:
        st.metric("Treatment", str(result["treatment_status"]).capitalize())

    with st.container(border=True):
        st.markdown("**Ringkasan Hasil**")
        st.write(f"- Weakest Indicator: **{result['weakest_indicator']}**")
        st.write(f"- Prediksi ML: **{str(result['prediksi_ml']).capitalize()}**")
        st.write(f"- Materi Treatment: **{result['treatment_materi']}**")
        st.write(f"- Level Treatment: **{result['treatment_level']}**")
        st.write(f"- Jumlah Soal Treatment: **{result['treatment_jumlah_soal']}**")
        st.write(f"- Waktu Submit: **{result['timestamp']}**")

    if st.button("Mulai Siswa Baru", type="primary"):
        reset_all()
        st.rerun()
