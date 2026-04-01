import json
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import Client, create_client

INT_TO_LABEL = {0: "rendah", 1: "sedang", 2: "tinggi"}
THRESHOLD_LOW_DEFAULT = 7


# =========================
# SUPABASE
# =========================
@st.cache_resource
def get_supabase() -> Client:
    """
    Inisialisasi client Supabase dari Streamlit secrets.
    Pastikan secrets berisi:
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "xxxx"
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


# =========================
# BANK SOAL
# =========================
@st.cache_data(ttl=60)
def load_bank_soal() -> pd.DataFrame:
    supabase = get_supabase()

    response = (
        supabase
        .table("bank_soal")
        .select("*")
        .execute()
    )

    data = response.data or []
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame()

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
# SISWA HELPERS
# =========================
def get_siswa_by_nis(nis: str) -> Optional[Dict[str, Any]]:
    supabase = get_supabase()

    response = (
        supabase
        .table("siswa")
        .select("*")
        .eq("nis", str(nis).strip())
        .limit(1)
        .execute()
    )

    rows = response.data or []
    return rows[0] if rows else None


def create_siswa(nis: str, nama: str, kelas: str) -> Dict[str, Any]:
    supabase = get_supabase()

    payload = {
        "nis": str(nis).strip(),
        "nama": str(nama).strip(),
        "kelas": str(kelas).strip()
    }

    response = (
        supabase
        .table("siswa")
        .insert(payload)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal membuat data siswa.")
    return rows[0]


def get_or_create_siswa(nis: str, nama: str, kelas: str) -> Dict[str, Any]:
    siswa = get_siswa_by_nis(nis)

    if siswa:
        return siswa

    return create_siswa(nis, nama, kelas)


# =========================
# SESSION HELPERS
# =========================
def get_active_session_by_siswa_id(siswa_id: str) -> Optional[Dict[str, Any]]:
    supabase = get_supabase()

    response = (
        supabase
        .table("session_pembelajaran")
        .select("*")
        .eq("siswa_id", siswa_id)
        .neq("status_session", "selesai")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    return rows[0] if rows else None


def create_new_session(siswa_id: str) -> Dict[str, Any]:
    supabase = get_supabase()

    payload = {
        "siswa_id": siswa_id,
        "status_session": "pretest",
        "pretest_selesai": False,
        "posttest_selesai": False,
        "treatment_status": "belum_mulai",
        "d_score": 0,
        "p_score": 0,
        "a_score": 0,
        "alg_score": 0,
        "total_score": 0,
        "answered_count": 0,
        "points": 0,
        "current_ct_idx": 0,
        "priority_order_json": [],
        "start_level_map_json": {},
        "mastered_ct_json": [],
        "history_ids_json": [],
        "served_items_json": [],
        "project_ready": False
    }

    response = (
        supabase
        .table("session_pembelajaran")
        .insert(payload)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal membuat session pembelajaran baru.")
    return rows[0]


def get_or_create_active_session(siswa_id: str) -> Dict[str, Any]:
    session_row = get_active_session_by_siswa_id(siswa_id)

    if session_row:
        return session_row

    return create_new_session(siswa_id)


def update_session_after_pretest(
    session_id: str,
    profile: dict,
    treatment_state: dict
) -> Dict[str, Any]:
    supabase = get_supabase()

    payload = {
        "status_session": "treatment",
        "pretest_selesai": True,
        "treatment_status": "berjalan",
        "d_score": int(profile["scores"]["D"]),
        "p_score": int(profile["scores"]["P"]),
        "a_score": int(profile["scores"]["A"]),
        "alg_score": int(profile["scores"]["Alg"]),
        "total_score": int(profile["total"]),
        "weakest_indicator": profile["weak_indicator"],
        "prediksi_ml": profile["overall"],
        "answered_count": int(treatment_state.get("answered_count", 0)),
        "current_ct": treatment_state.get("current_ct"),
        "current_level": treatment_state.get("current_level"),
        "points": int(treatment_state.get("points", 0)),
        "current_ct_idx": int(treatment_state.get("current_ct_idx", 0)),
        "priority_order_json": treatment_state.get("priority_order", []),
        "start_level_map_json": treatment_state.get("start_level_map", {}),
        "mastered_ct_json": treatment_state.get("mastered_ct", []),
        "history_ids_json": treatment_state.get("history_ids", []),
        "served_items_json": treatment_state.get("served_items", []),
        "project_ready": bool(treatment_state.get("project_ready", False)),
        "updated_at": now_iso()
    }

    response = (
        supabase
        .table("session_pembelajaran")
        .update(payload)
        .eq("id", session_id)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal update session setelah pretest.")
    return rows[0]


def update_session_treatment_progress(
    session_id: str,
    state: dict,
    treatment_status: str = "berjalan"
) -> Dict[str, Any]:
    supabase = get_supabase()

    payload = {
        "status_session": "treatment" if not state.get("project_ready", False) else "posttest",
        "treatment_status": treatment_status if not state.get("project_ready", False) else "selesai",
        "answered_count": int(state.get("answered_count", 0)),
        "current_ct": state.get("current_ct"),
        "current_level": state.get("current_level"),
        "points": int(state.get("points", 0)),
        "current_ct_idx": int(state.get("current_ct_idx", 0)),
        "priority_order_json": state.get("priority_order", []),
        "start_level_map_json": state.get("start_level_map", {}),
        "mastered_ct_json": state.get("mastered_ct", []),
        "history_ids_json": state.get("history_ids", []),
        "served_items_json": state.get("served_items", []),
        "project_ready": bool(state.get("project_ready", False)),
        "updated_at": now_iso()
    }

    response = (
        supabase
        .table("session_pembelajaran")
        .update(payload)
        .eq("id", session_id)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal update progress treatment.")
    return rows[0]


def mark_session_skip_treatment(session_id: str, state: dict) -> Dict[str, Any]:
    supabase = get_supabase()

    payload = {
        "status_session": "posttest",
        "treatment_status": "skip",
        "answered_count": int(state.get("answered_count", 0)),
        "current_ct": state.get("current_ct"),
        "current_level": state.get("current_level"),
        "points": int(state.get("points", 0)),
        "current_ct_idx": int(state.get("current_ct_idx", 0)),
        "priority_order_json": state.get("priority_order", []),
        "start_level_map_json": state.get("start_level_map", {}),
        "mastered_ct_json": state.get("mastered_ct", []),
        "history_ids_json": state.get("history_ids", []),
        "served_items_json": state.get("served_items", []),
        "project_ready": bool(state.get("project_ready", False)),
        "updated_at": now_iso()
    }

    response = (
        supabase
        .table("session_pembelajaran")
        .update(payload)
        .eq("id", session_id)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal menandai skip treatment.")
    return rows[0]


def update_session_final(
    session_id: str,
    profile: dict,
    treatment_state: dict,
    posttest_score: int,
    treatment_status: str = "selesai"
) -> Dict[str, Any]:
    supabase = get_supabase()

    treatment_materi, treatment_level = summarize_treatment(
        treatment_state.get("served_items", [])
    )
    treatment_jumlah_soal = int(treatment_state.get("answered_count", 0))
    gain_score = int(posttest_score) - int(profile["total"])

    payload = {
        "status_session": "selesai",
        "posttest_selesai": True,
        "treatment_status": treatment_status,
        "answered_count": treatment_jumlah_soal,
        "current_ct": treatment_state.get("current_ct"),
        "current_level": treatment_state.get("current_level"),
        "points": int(treatment_state.get("points", 0)),
        "current_ct_idx": int(treatment_state.get("current_ct_idx", 0)),
        "priority_order_json": treatment_state.get("priority_order", []),
        "start_level_map_json": treatment_state.get("start_level_map", {}),
        "mastered_ct_json": treatment_state.get("mastered_ct", []),
        "history_ids_json": treatment_state.get("history_ids", []),
        "served_items_json": treatment_state.get("served_items", []),
        "project_ready": bool(treatment_state.get("project_ready", False)),
        "posttest_score": int(posttest_score),
        "gain_score": int(gain_score),
        "finished_at": now_iso(),
        "updated_at": now_iso()
    }

    response = (
        supabase
        .table("session_pembelajaran")
        .update(payload)
        .eq("id", session_id)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ValueError("Gagal update hasil akhir session.")
    session_row = rows[0]

    # kembalikan ringkasan hasil untuk ditampilkan di app
    result_row = {
        "record_id": session_row["id"],
        "timestamp": session_row.get("finished_at") or session_row.get("updated_at"),
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

    return result_row


# =========================
# RESTORE HELPERS
# =========================
def restore_student_profile_from_session(siswa_row: dict, session_row: dict) -> dict:
    scores = {
        "D": int(session_row.get("d_score") or 0),
        "P": int(session_row.get("p_score") or 0),
        "A": int(session_row.get("a_score") or 0),
        "Alg": int(session_row.get("alg_score") or 0)
    }

    return {
        "student_id": str(siswa_row.get("nis", "")).strip(),
        "student_name": str(siswa_row.get("nama", "")).strip(),
        "student_class": str(siswa_row.get("kelas", "")).strip(),
        "scores": scores,
        "total": int(session_row.get("total_score") or 0),
        "overall": str(session_row.get("prediksi_ml") or "").strip(),
        "weak_indicator": str(session_row.get("weakest_indicator") or "").strip(),
        "needs": [k for k, v in scores.items() if v < THRESHOLD_LOW_DEFAULT]
    }


def restore_treatment_state_from_session(session_row: dict) -> dict:
    return {
        "priority_order": session_row.get("priority_order_json") or [],
        "start_level_map": session_row.get("start_level_map_json") or {},
        "current_ct_idx": int(session_row.get("current_ct_idx") or 0),
        "current_ct": session_row.get("current_ct"),
        "current_level": session_row.get("current_level"),
        "points": int(session_row.get("points") or 0),
        "mastered_ct": session_row.get("mastered_ct_json") or [],
        "history_ids": session_row.get("history_ids_json") or [],
        "project_ready": bool(session_row.get("project_ready") or False),
        "served_items": session_row.get("served_items_json") or [],
        "answered_count": int(session_row.get("answered_count") or 0)
    }


# =========================
# LEGACY-COMPATIBLE RESULT BUILDER
# =========================
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
    """
    Dipertahankan untuk kompatibilitas jika masih dipakai di app lama,
    tapi untuk Supabase sebaiknya pakai update_session_final().
    """
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


def save_student_result(*args, **kwargs):
    """
    Dummy compatibility wrapper.
    Di arsitektur Supabase baru, jangan pakai ini lagi.
    Pakai update_session_final(...).
    """
    raise NotImplementedError(
        "Untuk Supabase, gunakan update_session_final(session_id, profile, treatment_state, posttest_score, treatment_status)."
    )
