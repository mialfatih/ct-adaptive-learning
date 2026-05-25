import streamlit as st

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
    get_or_create_siswa,
    get_active_session_by_siswa_id,
    get_or_create_active_session,
    update_session_after_pretest,
    update_session_treatment_progress,
    mark_session_skip_treatment,
    update_session_final,
    restore_student_profile_from_session,
    restore_treatment_state_from_session,
    save_phase_answers,
    save_treatment_answer,
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
# LOAD RESOURCES
# =========================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

try:
    bank = load_bank_soal()
except Exception as e:
    st.error(f"Gagal load Bank_Soal dari Supabase: {e}")
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
    "siswa_row": None,
    "session_row": None,
    "pretest_df": None,
    "posttest_df": None,
    "treatment_state": None,
    "current_question": None,
    "pretest_answers": {},
    "posttest_answers": {},
    "final_result": None,
    "saved_to_db": False,
    "treatment_status": "selesai",
    "wrong_previous": False  # <--- Variabel untuk menyimpan Hint AI
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_all():
    for k, v in defaults.items():
        st.session_state[k] = v

# =========================
# UI HELPERS & KNN HINT
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

def generate_knn_hint(ct_type, materi, knn_level):
    """Menghasilkan hint dinamis berdasarkan klasifikasi KNN siswa"""
    hints_rendah = {
        "D": f"Pecah masalah {materi} ini menjadi 3 bagian: Perintah utama, Tabel yang dituju, dan Kondisi/Syaratnya.",
        "P": f"Ingat kembali pola dasar penulisan {materi}. Format urutan syntax-nya selalu sama dan berulang.",
        "A": f"Abaikan nama data spesifiknya. Fokus pada struktur inti {materi} (seperti nama tabel atau kolom yang tepat).",
        "Alg": f"Susun langkah {materi} ini secara berurutan. Pikirkan apa syntax yang harus diketik pertama kali."
    }
    hints_sedang = {
        "D": f"Cek kembali struktur dasar (Perintah, Tabel, Kondisi) pada {materi} ini.",
        "P": f"Periksa apakah ada bagian klausa {materi} yang terlewat atau tidak sesuai urutan.",
        "A": f"Fokus pada penulisan nama tabel dan kolom yang menjadi target.",
        "Alg": f"Perhatikan urutan penulisan syntax {materi} Anda."
    }

    if knn_level == "rendah":
        return hints_rendah.get(ct_type, "Perhatikan kembali materi dan struktur soalnya.")
    elif knn_level == "sedang":
        return hints_sedang.get(ct_type, "Coba teliti kembali penulisan syntax Anda.")
    
    return None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Menu Navigasi")
    
    prof = st.session_state.get("student_profile")
    if prof:
        st.write(f"👤 **{prof.get('student_name', '-')}** ({prof.get('student_class', '-')})")
        st.write("---")

    # 1. PETA PERJALANAN (VISUAL STEPPER)
    st.markdown("**📍 Posisi Kamu Saat Ini:**")
    
    # Logika untuk menentukan icon centang berdasarkan stage
    stages_order = ["identitas", "pretest", "hasil_pretest", "treatment", "posttest", "final"]
    current_stage = st.session_state["stage"]
    current_idx = stages_order.index(current_stage) if current_stage in stages_order else 0

    def get_icon(stage_name):
        idx = stages_order.index(stage_name)
        if idx < current_idx: return "✅"
        elif idx == current_idx: return "🔵"
        else: return "⚪"

    st.write(f"{get_icon('identitas')} 1. Isi Identitas")
    st.write(f"{get_icon('pretest')} 2. Tes Awal (Pretest)")
    st.write(f"{get_icon('treatment')} 3. Latihan Adaptif (Treatment)")
    st.write(f"{get_icon('posttest')} 4. Tes Akhir (Posttest)")
    st.write(f"{get_icon('final')} 5. Selesai")

    st.write("---")

    # 2. PETUNJUK PENGGUNAAN (EXPANDER DINAMIS)
    with st.expander("📖 Buka Petunjuk Penggunaan"):
        if current_stage in ["identitas"]:
            st.write("Isi NIS, Nama, dan Kelasmu dengan benar. Data ini digunakan untuk menyimpan riwayat belajarmu.")
            
        elif current_stage in ["pretest", "hasil_pretest"]:
            st.markdown("""
            **Fase Pretest:**
            * Jawablah 32 soal ini dengan sungguh-sungguh.
            * Jangan takut salah! Hasil ini digunakan oleh AI untuk meracik menu latihan yang paling pas dengan tingkat pemahamanmu.
            """)
            
        elif current_stage == "treatment":
            st.markdown("""
            **Aturan Main Latihan Adaptif:**
            * Sitem telah memilihkan materi berdasarkan titik kelemahanmu dari hasil Pretest.
            * **Sistem Poin:** Jawab benar poin bertambah (+1), jawab salah poin berkurang (-1).
            * **Target Lulus Level:**
              - Level **Easy** : Kumpulkan **3 Poin**
              - Level **Medium** : Kumpulkan **5 Poin**
              - Level **Hard** : Kumpulkan **4 Poin**
            * Jika kamu menjawab salah, Sistem mungkin akan memunculkan petunjuk bantuan (Hint). Baca baik-baik!
            * *Tips: Ingat kembali penjelasan materi DDL, DML, DCL yang sudah dijelaskan oleh guru di kelas.*
            """)
            
        elif current_stage == "posttest":
            st.markdown("""
            **Fase Posttest:**
            * Buktikan kemampuanmu setelah melewati latihan adaptif!
            * Nilai ini akan dibandingkan dengan pretest untuk melihat seberapa jauh kamu berkembang.
            """)
            
        else:
            st.write("Selamat! Kamu telah menyelesaikan seluruh rangkaian pembelajaran.")

    st.write("---")
    if st.button("Reset Sesi", use_container_width=True):
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
    st.subheader("Selamat Datang di Sistem Adaptive Learning")

    # --- TAMBAHAN CP & ATP (Menu Lipat) ---
    with st.expander("📚 Informasi Capaian Pembelajaran (CP) & ATP"):
        st.markdown("""
        **Mata Pelajaran:** Basis Data (RPL) | **Fase:** F (Kelas XI)
        
        **Capaian Pembelajaran (CP):**
        Pada akhir fase F, peserta didik mampu memahami, menganalisis, dan menerapkan pengelolaan basis data menggunakan perintah *Data Definition Language* (DDL), *Data Manipulation Language* (DML), dan *Data Control Language* (DCL) dalam menyelesaikan masalah secara terstruktur.
        
        **Alur Tujuan Pembelajaran (ATP):**
        1. Peserta didik mampu menganalisis permasalahan data dan memecahnya menjadi komponen penyusun basis data (Dekomposisi).
        2. Peserta didik mampu merancang struktur tabel dan tipe data (DDL) dengan mengabaikan detail yang tidak perlu (Abstraksi).
        3. Peserta didik mampu mengidentifikasi pola hubungan antar data untuk melakukan manipulasi data (DML).
        4. Peserta didik mampu merumuskan langkah-langkah logis dan berurutan (Algoritma) untuk memberikan hak akses keamanan data (DCL).
        
        *(Catatan: Sistem ini menggunakan pendekatan Problem Based Learning terintegrasi Computational Thinking untuk mencapai target kurikulum di atas).*
        """)
    # --------------------------------------

    with st.container(border=True):
        st.write("Silakan isi data diri terlebih dahulu sebelum memulai tes awal.")

        with st.form("form_identitas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                student_id = st.text_input("NIS / ID")
            with c2:
                student_name = st.text_input("Nama")
            with c3:
                student_class = st.text_input("Kelas")

            submitted = st.form_submit_button("Mulai Pembelajaran", type="primary")
            
    # ... (kode submit ke bawah tetap sama) ...
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
            st.error("Soal pretest tidak ditemukan di bank_soal.")
            st.stop()
        if posttest_df.empty:
            st.error("Soal posttest tidak ditemukan di bank_soal.")
            st.stop()

        try:
            siswa_row = get_or_create_siswa(
                nis=student_id.strip(),
                nama=student_name.strip(),
                kelas=student_class.strip()
            )
            session_row = get_or_create_active_session(siswa_row["id"])

            st.session_state["siswa_row"] = siswa_row
            st.session_state["session_row"] = session_row
            st.session_state["pretest_df"] = pretest_df
            st.session_state["posttest_df"] = posttest_df
            st.session_state["pretest_answers"] = {}
            st.session_state["posttest_answers"] = {}
            st.session_state["current_question"] = None
            st.session_state["final_result"] = None
            st.session_state["saved_to_db"] = False

            status_session = session_row.get("status_session", "pretest")

            if status_session != "pretest":
                profile = restore_student_profile_from_session(siswa_row, session_row)
                treatment_state = restore_treatment_state_from_session(session_row)

                st.session_state["student_profile"] = profile
                st.session_state["treatment_state"] = treatment_state
                st.session_state["treatment_status"] = session_row.get("treatment_status", "selesai")

                if status_session == "treatment":
                    st.session_state["stage"] = "treatment"
                    st.success("Progress sebelumnya ditemukan. Kamu bisa melanjutkan treatment.")
                elif status_session == "posttest":
                    st.session_state["stage"] = "posttest"
                    st.success("Progress sebelumnya ditemukan. Kamu bisa melanjutkan posttest.")
                else:
                    st.session_state["stage"] = "pretest"

                st.rerun()

            st.session_state["student_profile"] = {
                "student_id": student_id.strip(),
                "student_name": student_name.strip(),
                "student_class": student_class.strip()
            }
            st.session_state["treatment_state"] = None
            st.session_state["treatment_status"] = "belum_mulai"
            st.session_state["stage"] = "pretest"
            st.rerun()

        except Exception as e:
            st.error(f"Gagal menyiapkan data siswa/session: {e}")

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

        treatment_state = init_treatment_state(scores)
        st.session_state["treatment_state"] = treatment_state
        st.session_state["current_question"] = None
        st.session_state["treatment_status"] = "berjalan"

        try:
            updated_session = update_session_after_pretest(
                session_id=st.session_state["session_row"]["id"],
                profile=st.session_state["student_profile"],
                treatment_state=treatment_state
            )
            st.session_state["session_row"] = updated_session

            save_phase_answers(
                df_questions=pretest_df,
                answers_dict=st.session_state["pretest_answers"],
                session_row=updated_session,
                siswa_row=st.session_state["siswa_row"],
                phase="pretest",
                replace_existing=True
            )
        except Exception as e:
            st.error(f"Gagal menyimpan hasil pretest ke Supabase: {e}")
            st.stop()

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
    st.write("Kerjakan soal treatment berikut. Progres akan tersimpan agar bisa dilanjutkan kembali.")
    
    # === TAMBAHAN UI ROADMAP CT ===
    ct_names = {"D": "Dekomposisi", "P": "Pola", "A": "Abstraksi", "Alg": "Algoritma"}
    trackers = []
    
    for ct in state.get("priority_order", []):
        nama = ct_names.get(ct, ct)
        if ct in state.get("mastered_ct", []):
            trackers.append(f"🟢 **{nama}**") 
        elif ct == state.get("current_ct", ""):
            trackers.append(f"🔵 **{nama}**") 
        else:
            trackers.append(f"⚪ {nama}") 

    st.markdown(" **Jalur Belajarmu:** &nbsp; " + " ➔ ".join(trackers))
    st.write("---")
    # ==============================

    # ==============================
    # TAMPILKAN HINT JIKA ADA
    # ==============================
    
    # ==============================

    if state.get("project_ready", False):
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
        st.error("Tidak ada soal treatment yang cocok di bank_soal.")
        st.stop()

    # ==========================================
    # TRIGGER ADAPTIVE SCAFFOLDING (FORWARD-LOOKING HINT)
    # ==========================================
    # Jika siswa salah di soal sebelumnya, berikan hint untuk soal YANG SEKARANG (q)
    if st.session_state.get("wrong_previous"):
        knn_level = st.session_state["student_profile"].get("overall", "tinggi").lower()
        soal_level = q["level"].lower()
        soal_ct = q["ct"]
        soal_materi = q["materi"]
        
        # Hint keluar jika level soal saat ini easy/medium DAN KNN bukan tinggi
        if soal_level in ["easy", "medium"] and knn_level != "tinggi":
            hint_text = generate_knn_hint(soal_ct, soal_materi, knn_level)
            if hint_text:
                st.warning(f"💡 **Petunjuk AI:** {hint_text}")
        
        # Matikan alarm setelah hint ditampilkan (agar tidak muncul terus)
        st.session_state["wrong_previous"] = False
    # ==========================================

    deskripsi_ct = {
        "D": "Dekomposisi: Memecah masalah yang rumit menjadi bagian-bagian yang lebih kecil dan mudah diselesaikan.",
        "P": "Pengenalan Pola: Mencari kesamaan atau tren dalam masalah untuk membantu menemukan solusi yang tepat.",
        "A": "Abstraksi: Fokus pada informasi utama yang penting dan mengabaikan detail yang tidak relevan.",
        "Alg": "Algoritma: Menyusun langkah-langkah logis dan berurutan untuk menyelesaikan masalah."
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Fokus CT", 
            value=state["current_ct"], 
            help=deskripsi_ct.get(state["current_ct"], "")
        )
    with c2:
        st.metric("Level Soal", state["current_level"].capitalize())
    with c3:
        target_poin = level_target(state["current_level"])
        poin_sekarang = state.get("points", 0)
        st.metric("Progres Mastery", f"{poin_sekarang} / {target_poin}")

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

            try:
                save_treatment_answer(
                    session_row=st.session_state["session_row"],
                    siswa_row=st.session_state["siswa_row"],
                    question_row=q,
                    selected_answer=choice,
                    attempt_order=state.get("answered_count", 0)
                )
            except Exception as e:
                st.error(f"Gagal menyimpan jawaban treatment: {e}")
                st.stop()

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
                st.session_state["wrong_previous"] = False # Jawaban benar, tidak butuh hint di soal selanjutnya
            else:
                if state["points"] > 0:
                    state["points"] -= 1
                st.error("Jawaban belum tepat.")
                st.session_state["wrong_previous"] = True  # Nyalakan alarm! Siswa butuh hint di soal selanjutnya

            if state["points"] >= level_target(state["current_level"]):
                state, msg = advance_state(state)
                st.info(msg)
                # Jika dia naik level karena poinnya cukup, matikan alarm hint agar di level baru dia mulai fresh
                st.session_state["wrong_previous"] = False

                # ==========================================
                # TRIGGER ADAPTIVE SCAFFOLDING (KNN HINT)
                # ==========================================
                knn_level = st.session_state["student_profile"].get("overall", "tinggi").lower()
                soal_level = q["level"].lower()
                soal_ct = q["ct"]
                soal_materi = q["materi"]
                
                if soal_level in ["easy", "medium"] and knn_level != "tinggi":
                    hint_text = generate_knn_hint(soal_ct, soal_materi, knn_level)
                    if hint_text:
                        st.session_state["last_hint"] = hint_text
                else:
                    st.session_state["last_hint"] = None
                # ==========================================

            if state["points"] >= level_target(state["current_level"]):
                state, msg = advance_state(state)
                st.info(msg)

            st.session_state["treatment_state"] = state
            st.session_state["current_question"] = None
            st.session_state["treatment_status"] = "berjalan"

            try:
                updated_session = update_session_treatment_progress(
                    session_id=st.session_state["session_row"]["id"],
                    state=state,
                    treatment_status="berjalan"
                )
                st.session_state["session_row"] = updated_session
            except Exception as e:
                st.error(f"Gagal menyimpan progress treatment: {e}")
                st.stop()

            st.rerun()

    with c2:
        # Tombol awal hanya memicu munculnya area konfirmasi
        if st.button("Lewati Treatment", use_container_width=True):
            st.session_state["confirm_skip"] = True

        # Jika tombol awal sudah diklik, munculkan dialog konfirmasi
        if st.session_state.get("confirm_skip", False):
            with st.container(border=True):
                st.warning("⚠️ **Yakin ingin melewati latihan?** Progres di tahap ini akan dihentikan.")
                
                # Checkbox persetujuan
                izin_guru = st.checkbox("Saya sudah mendapat arahan/izin dari guru untuk melewati latihan ini.")
                
                # Tombol eksekusi hanya aktif jika checkbox dicentang
                if izin_guru:
                    if st.button("Ya, Lewati Sekarang", type="primary", use_container_width=True):
                        st.session_state["treatment_status"] = "skip"

                        try:
                            updated_session = mark_session_skip_treatment(
                                session_id=st.session_state["session_row"]["id"],
                                state=state
                            )
                            st.session_state["session_row"] = updated_session
                        except Exception as e:
                            st.error(f"Gagal menandai skip treatment: {e}")
                            st.stop()

                        # Bersihkan state konfirmasi dan pindah ke posttest
                        st.session_state["confirm_skip"] = False
                        st.session_state["stage"] = "posttest"
                        st.rerun()
                
                # Tombol batal untuk menutup area konfirmasi
                if st.button("Batal", use_container_width=True):
                    st.session_state["confirm_skip"] = False
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

        try:
            save_phase_answers(
                df_questions=posttest_df,
                answers_dict=st.session_state["posttest_answers"],
                session_row=st.session_state["session_row"],
                siswa_row=st.session_state["siswa_row"],
                phase="posttest",
                replace_existing=True
            )
            
            saved_row = update_session_final(
                session_id=st.session_state["session_row"]["id"],
                profile=st.session_state["student_profile"],
                treatment_state=st.session_state["treatment_state"],
                posttest_score=posttest_score,
                treatment_status=st.session_state.get("treatment_status", "selesai")
            )
            
            st.session_state["final_result"] = saved_row
            st.session_state["saved_to_db"] = True
            st.session_state["stage"] = "final"
            st.rerun()
            
        except Exception as e:
            st.error(f"Gagal menyimpan hasil akhir ke Supabase: {e}")

# =========================
# STAGE 6: FINAL
# =========================
elif st.session_state["stage"] == "final":
    render_stage_badge("Selesai")
    render_student_box()

    result = st.session_state.get("final_result")

    if not result:
        st.error("Hasil akhir tidak ditemukan.")
        st.stop()

    st.subheader("Terima Kasih")
    st.success("Data kamu berhasil disimpan.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Pretest", result.get("total_score", 0))
    with c2:
        st.metric("Posttest", result.get("posttest_score", 0))
    with c3:
        st.metric("Gain", result.get("gain_score", 0))
    with c4:
        st.metric("Treatment", str(result.get("treatment_status", "selesai")).capitalize())

    with st.container(border=True):
        st.markdown("**Ringkasan Hasil**")
        st.write(f"- Weakest Indicator: **{result.get('weakest_indicator', '-')}**")
        st.write(f"- Prediksi ML: **{str(result.get('prediksi_ml', '-')).capitalize()}**")
        st.write(f"- Materi Treatment: **{result.get('treatment_materi', '-')}**")
        st.write(f"- Level Treatment: **{result.get('treatment_level', '-')}**")
        st.write(f"- Jumlah Soal Treatment: **{result.get('treatment_jumlah_soal', 0)}**")
        st.write(f"- Waktu Submit: **{result.get('timestamp', '-')}**")

    if st.button("Mulai Siswa Baru", type="primary"):
        reset_all()
        st.rerun()
