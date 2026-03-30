# 🚀 CT Adaptive Learning System

### *Machine Learning-Based Adaptive Learning for Computational Thinking (CT)*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

**CT Adaptive Learning System** adalah aplikasi berbasis web yang dirancang untuk mengukur dan meningkatkan kemampuan **Computational Thinking (CT)** siswa melalui pendekatan **Adaptive Learning** yang dikombinasikan dengan **Machine Learning (KNN)**.

Sistem ini dikembangkan sebagai bagian dari penelitian skripsi pada bidang **Pendidikan Ilmu Komputer**, dengan tujuan untuk menciptakan lingkungan belajar yang **personalized, adaptif, dan berbasis data**.

---

## 🎯 Key Features

* 🧠 **Computational Thinking Assessment**

  * Dekomposisi
  * Pengenalan Pola
  * Abstraksi
  * Algoritma

* 🤖 **Machine Learning Classification (KNN)**

  * Klasifikasi level CT siswa: *rendah, sedang, tinggi*

* 🎯 **Adaptive Learning Engine**

  * Treatment berdasarkan:

    * Needs (threshold)
    * Weakest Indicator

* 📊 **Pretest – Treatment – Posttest Flow**

  * Pengukuran peningkatan kemampuan siswa (gain score)

* ☁️ **Cloud-Based Deployment**

  * Akses real-time melalui Streamlit Cloud

---

## 🧠 System Architecture

```text
User (Student)
      ↓
Pretest (32 Questions)
      ↓
Scoring per CT Indicator
      ↓
Machine Learning (KNN)
      ↓
Level CT (Low / Medium / High)
      ↓
Rule-Based Treatment
(Needs + Weak Indicator)
      ↓
Adaptive Questions (80 Bank)
      ↓
Posttest
      ↓
Gain Score & Analysis
```

---

## 🛠️ Tech Stack

| Layer            | Technology                |
| ---------------- | ------------------------- |
| Frontend         | Streamlit                 |
| Backend Logic    | Python                    |
| Machine Learning | Scikit-learn (KNN)        |
| Database         | Google Sheets             |
| Deployment       | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
ct-adaptive-learning/
│
├── app.py                  # Main Streamlit app
├── app2.py                 # Alternative / extended app logic
├── utils.py                # Helper functions
│
├── dataset_ct_240_balanced.csv
├── knn_ct_meta.json
│
├── requirements.txt
├── README.md
│
└── mlctskripsi.ipynb       # Experiment notebook
```

---

## 🚀 Live Demo

👉 https://ct-adaptive-learning.streamlit.app/

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/mialfatih/ct-adaptive-learning.git
cd ct-adaptive-learning
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔐 Configuration

> ⚠️ File credential (Google Sheets API) tidak disertakan demi keamanan.

Untuk menjalankan sistem:

1. Tambahkan file credential sendiri
2. Atur koneksi Google Sheets
3. Sesuaikan konfigurasi di `secrets.toml`

---

## 📊 Machine Learning Details

* Algorithm: **K-Nearest Neighbors (KNN)**
* Features:

  * D_score
  * P_score
  * A_score
  * Alg_score
* Range: 0 – 14 per indicator
* Label:

  * 0 = Rendah
  * 1 = Sedang
  * 2 = Tinggi

---

## 🎓 Research Context

Sistem ini dikembangkan menggunakan pendekatan:

* **R&D (Research and Development)**
* Model: **SLEEG + ADDIE**
* Standar: **ISO 21001:2018**

Digunakan untuk siswa tingkat **SMK (Informatika)**.

---

## 📈 Future Improvements

* 🔐 Authentication system (student login)
* 📊 Dashboard analitik untuk guru
* 🧠 Model ML yang lebih advanced (Random Forest / XGBoost)
* 🗄️ Migrasi database ke SQL

---

## 🤝 Contributing

Open for feedback & collaboration.

---

## 📜 License

For educational and research purposes.

---

## 👨‍💻 Author

**Muhammad Izzuddin Al Fatih**
Undergraduate Student – Pendidikan Ilmu Komputer
Universitas Pendidikan Indonesia (UPI)

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!

---
