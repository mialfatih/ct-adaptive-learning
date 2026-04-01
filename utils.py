import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

INT_TO_LABEL = {0: "rendah", 1: "sedang", 2: "tinggi"}
THRESHOLD_LOW_DEFAULT = 7

DATA_SISWA_COLUMNS = [
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

PRETEST_SISWA_COLUMNS = [
    "pretest_id",
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
    "status_pretest"
]


# =========================
# LOADERS
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(ttl=60)
def load_bank_soal(_conn) -> pd.DataFrame:
    df = _conn.read(worksheet="Bank_Soal")

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = [
        "id",
        "phase",
        "materi",
        "ct",
        "level",
        "question",
        "opt_a",
        "opt_b",
        "opt_c",
        "opt_d",
        "opt_e",
        "answer"
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    for col in ["id", "phase", "materi", "ct", "level", "question", "answer"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["opt_a", "opt_b", "opt_c", "opt_d", "opt_e"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["phase"] = df["phase"].str.lower()
    df["level"] = df["level"].str.lower()
    df["answer"] = df["answer"].str.upper()
    df["materi"] = df["materi"].str.upper()

    def normalize_ct(x):
        x = str(x).strip()
        if x.lower() == "alg":
            return "Alg"
        if x.lower() == "d":
            return "D"
        if x.lower() == "p":
            return "P"
        if x.lower() == "a":
            return "A"
        return x

    df["ct"] = df["ct"].apply(normalize_ct)

    return df


@st.cache_data(ttl=30)
def load_data_siswa(_conn) -> pd.DataFrame:
    last_error = None

    for _ in range(3):
        try:
            df = _conn.read(worksheet="Data_Siswa")

            if df is None or len(df) == 0:
                return pd.DataFrame(columns=DATA_SISWA_COLUMNS)

            df.columns = [str(c).strip() for c in df.columns]

            for col in DATA_SISWA_COLUMNS:
                if col not in df.columns:
                    df[col] = ""

            return df[DATA_SISWA_COLUMNS].copy()

        except Exception as e:
            last_error = e
            time.sleep(1)

    raise last_error


@st.cache_data(ttl=30)
def load_pretest_siswa(_conn) -> pd.DataFrame:
    last_error = None

    for _ in range(3):
        try:
            df = _conn.read(worksheet="Pretest_Siswa")

            if df is None or len(df) == 0:
                return pd.DataFrame(columns=PRETEST_SISWA_COLUMNS)

            df.columns = [str(c).strip() for c in df.columns]

            for col in PRETEST_SISWA_COLUMNS:
                if col not in df.columns:
                    df[col] = ""

            return df[PRETEST_SISWA_COLUMNS].copy()

        except Exception as e:
            last_error = e
            time.sleep(1)

    raise last_error


# =========================
# DATA HELPERS
# =========================
def get_phase_questions(bank: pd.DataFrame, phase: str) -> pd.DataFrame:
    phase = str(phase).strip().lower()
    df = bank[bank["phase"] == phase].copy()
    return df.reset_index(drop=True)


def get_weight(level: str) -> int:
    level = str(level).strip().lower()
    weight_map = {
        "easy": 1,
        "medium": 2,
        "hard": 3
    }
    return weight_map.get(level, 0)


def build_options(row: pd.Series):
    mapping = [
        ("A", "opt_a"),
        ("B", "opt_b"),
        ("C", "opt_c"),
        ("D", "opt_d"),
        ("E", "opt_e")
    ]

    options_letters = []
    labels = []

    for letter, col in mapping:
        txt = str(row.get(col, "")).strip()
        if txt != "":
            options_letters.append(letter)
            labels.append(f"{letter}. {txt}")

    return options_letters, labels


def calculate_pretest_scores(pretest_df: pd.DataFrame, answers_dict: dict) -> dict:
    scores = {"D": 0, "P": 0, "A": 0, "Alg": 0}

    for _, row in pretest_df.iterrows():
        qid = str(row["id"]).strip()
        correct_answer = str(row["answer"]).strip().upper()
        chosen = str(answers_dict.get(qid, "")).strip().upper()

        if chosen == correct_answer:
            ct = str(row["ct"]).strip()
            weight = get_weight(row["level"])
            if ct in scores:
                scores[ct] += weight

    total_score = sum(scores.values())
    weakest_indicator = min(scores, key=scores.get)
    needs = [k for k, v in scores.items() if v < THRESHOLD_LOW_DEFAULT]

    return {
        "scores": scores,
        "total_score": total_score,
        "weakest_indicator": weakest_indicator,
        "needs": needs
    }


def calculate_posttest_score(posttest_df: pd.DataFrame, answers_dict: dict) -> int:
    score = 0

    for _, row in posttest_df.iterrows():
        qid = str(row["id"]).strip()
        correct_answer = str(row["answer"]).strip().upper()
        chosen = str(answers_dict.get(qid, "")).strip().upper()

        if chosen == correct_answer:
            score += get_weight(row["level"])

    return int(score)


def predict_overall_level(model, D: int, P: int, A: int, Alg: int) -> str:
    X_new = np.array([[D, P, A, Alg]], dtype=int)
    pred_int = int(model.predict(X_new)[0])
    return INT_TO_LABEL[pred_int]


# =========================
# TREATMENT HELPERS
# =========================
def compute_priority_order(scores: dict):
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1])]


def compute_start_level(score: int):
    if score < 7:
        return "easy"
    elif score <= 9:
        return "medium"
    return "hard"


def init_treatment_state(scores: dict):
    priority_order = compute_priority_order(scores)
    start_level_map = {
        ct: compute_start_level(score)
        for ct, score in scores.items()
    }

    return {
        "priority_order": priority_order,
        "start_level_map": start_level_map,
        "current_ct_idx": 0,
        "current_ct": priority_order[0],
        "current_level": start_level_map[priority_order[0]],
        "points": 0,
        "mastered_ct": [],
        "history_ids": [],
        "project_ready": False,
        "served_items": [],
        "answered_count": 0
    }


def level_target(level: str):
    if level == "easy":
        return 3
    elif level == "medium":
        return 5
    elif level == "hard":
        return 2
    return 999


def get_next_level(level: str):
    if level == "easy":
        return "medium"
    elif level == "medium":
        return "hard"
    return None


def advance_state(state: dict):
    current_ct = state["current_ct"]
    current_level = state["current_level"]

    next_level = get_next_level(current_level)

    if next_level is not None:
        state["current_level"] = next_level
        state["points"] = 0
        return state, f"Naik ke level {next_level.upper()} untuk CT {current_ct}"

    if current_ct not in state["mastered_ct"]:
        state["mastered_ct"].append(current_ct)

    state["points"] = 0
    next_idx = state["current_ct_idx"] + 1

    if next_idx < len(state["priority_order"]):
        next_ct = state["priority_order"][next_idx]
        state["current_ct_idx"] = next_idx
        state["current_ct"] = next_ct
        state["current_level"] = state["start_level_map"][next_ct]
        return state, f"CT {current_ct} selesai. Lanjut ke CT {next_ct} level {state['current_level'].upper()}"

    state["project_ready"] = True
    return state, "Semua indikator treatment selesai. Lanjut ke posttest ✅"


def fetch_question(bank: pd.DataFrame, ct: str, level: str, history_ids: list):
    pool = bank[
        (bank["phase"] == "treatment") &
        (bank["ct"] == ct) &
        (bank["level"] == level)
    ].copy()

    if len(history_ids) > 0:
        pool = pool[~pool["id"].isin(history_ids)]

    if len(pool) == 0:
        pool = bank[
            (bank["phase"] == "treatment") &
            (bank["ct"] == ct) &
            (bank["level"] == level)
        ].copy()

    if len(pool) == 0:
        return None

    return pool.sample(1).iloc[0]


# =========================
# PRETEST_SISWA HELPERS
# =========================
def generate_pretest_id() -> str:
    return f"PT{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def find_existing_pretest(pretest_df: pd.DataFrame, nis: str, nama: str):
    if pretest_df.empty:
        return None

    temp = pretest_df.copy()
    temp["NIS"] = temp["NIS"].astype(str).str.strip()
    temp["nama"] = temp["nama"].astype(str).str.strip().str.lower()

    nis = str(nis).strip()
    nama = str(nama).strip().lower()

    matched = temp[
        (temp["NIS"] == nis) &
        (temp["nama"] == nama)
    ].copy()

    if matched.empty:
        return None

    matched["timestamp_dt"] = pd.to_datetime(matched["timestamp"], errors="coerce")
    matched = matched.sort_values("timestamp_dt", ascending=False)

    return matched.iloc[0].to_dict()


def build_pretest_row(profile: dict) -> dict:
    return {
        "pretest_id": generate_pretest_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "NIS": profile["student_id"],
        "nama": profile["student_name"],
        "kelas": profile["student_class"],
        "D_score": profile["scores"]["D"],
        "P_score": profile["scores"]["P"],
        "A_score": profile["scores"]["A"],
        "Alg_score": profile["scores"]["Alg"],
        "total_score": profile["total"],
        "weakest_indicator": profile["weak_indicator"],
        "prediksi_ml": profile["overall"],
        "status_pretest": "selesai"
    }


def save_pretest_result(conn, pretest_row: dict):
    row_to_save = pretest_row.copy()
    row_df = pd.DataFrame([row_to_save])

    conn.write(
        worksheet="Pretest_Siswa",
        data=row_df,
        append=True
    )

    load_pretest_siswa.clear()
    return row_to_save


# =========================
# SAVE HELPERS
# =========================
def generate_record_id() -> str:
    return f"R{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def summarize_treatment(served_items: list):
    if not served_items:
        return "-", "-"

    materi_list = []
    level_list = []

    for item in served_items:
        materi = str(item.get("materi", "")).strip()
        level = str(item.get("level", "")).strip().lower()

        if materi and materi not in materi_list:
            materi_list.append(materi)

        if level and level not in level_list:
            level_list.append(level)

    treatment_materi = ", ".join(materi_list) if materi_list else "-"
    treatment_level = ", ".join(level_list) if level_list else "-"

    return treatment_materi, treatment_level


def build_result_row(
    profile: dict,
    treatment_state: dict,
    posttest_score: int,
    treatment_status: str = "selesai"
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    treatment_materi, treatment_level = summarize_treatment(
        treatment_state.get("served_items", [])
    )

    treatment_jumlah_soal = int(treatment_state.get("answered_count", 0))
    gain_score = int(posttest_score) - int(profile["total"])

    return {
        "record_id": "",
        "timestamp": timestamp,
        "NIS": profile["student_id"],
        "nama": profile["student_name"],
        "kelas": profile["student_class"],
        "D_score": profile["scores"]["D"],
        "P_score": profile["scores"]["P"],
        "A_score": profile["scores"]["A"],
        "Alg_score": profile["scores"]["Alg"],
        "total_score": profile["total"],
        "weakest_indicator": profile["weak_indicator"],
        "prediksi_ml": profile["overall"],
        "treatment_materi": treatment_materi,
        "treatment_level": treatment_level,
        "treatment_jumlah_soal": treatment_jumlah_soal,
        "treatment_status": treatment_status,
        "posttest_score": int(posttest_score),
        "gain_score": int(gain_score),
        "status_selesai": "selesai"
    }


def save_student_result(conn, result_row: dict):
    row_to_save = result_row.copy()
    row_to_save["record_id"] = generate_record_id()

    new_row_df = pd.DataFrame([row_to_save])

    conn.write(
        worksheet="Data_Siswa",
        data=new_row_df,
        append=True
    )

    load_data_siswa.clear()
    return row_to_save
