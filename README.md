# 🫀 Heart Disease Risk Prediction Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)

Proyek ini bertujuan untuk membangun sistem prediksi risiko penyakit jantung menggunakan pendekatan *Machine Learning*. Pipeline ini mencakup pembersihan data (*Data Preprocessing*), rekayasa fitur (*Feature Engineering*), serta perbandingan strategi validasi data menggunakan **K-Fold** dan **Stratified K-Fold Cross Validation** yang dikombinasikan dengan optimasi hiperparameter (**GridSearchCV / RandomizedSearchCV**).

## 📌 Permasalahan & Tujuan
* **Masalah:** Penyakit jantung merupakan salah satu penyebab kematian tertinggi global. Deteksi dini melalui indikator kesehatan non-invasif dapat membantu intervensi medis yang cepat tepat.
* **Tujuan:** Membangun model klasifikasi biner yang mampu memprediksi secara akurat apakah seorang pasien berisiko terkena penyakit jantung (`target = 1`) atau tidak berisiko (`target = 0`).

## 📊 Dataset
Dataset yang digunakan merupakan gabungan data dari rekam medis **Cleveland** dan **Hungary** (`heart_statlog_cleveland_hungary_final.csv`). Fitur-fitur utama meliputi:
* **Fitur Numerik:** Usia (`age`), Tekanan Darah (`resting bp s`), Kolesterol (`cholesterol`), Detak Jantung Maksimal (`max heart rate`), dan ST Depression (`oldpeak`).
* **Fitur Kategorikal:** Jenis Kelamin (`sex`), Tipe Nyeri Dada (`chest pain type`), Gula Darah Puasa (`fasting blood sugar`), EKG Istirahat (`resting ecg`), Angina Akibat Olahraga (`exercise angina`), dan `ST slope`.

## ⚙️ Alur Kerja (Pipeline) Proyek

### 1. Data Preprocessing & Feature Engineering
Proses pembersihan data dilakukan di notebook `Data Preprocessing (Data Cleaning, Feature Engineering).ipynb`:
* **Penanganan Outlier:** Mendeteksi dan menangani data ekstrem pada fitur numerik seperti kolesterol menggunakan metode *Interquartile Range* (IQR).
* **One-Hot Encoding:** Mengonversi fitur kategorikal multinomial (seperti `chest pain type` dan `ST slope`) menjadi bentuk biner numerik.
* **Feature Scaling:** Melakukan standardisasi semua variabel numerik menggunakan `StandardScaler` agar memiliki skala rata-rata 0 dan varians 1.

### 2. Model Klasifikasi & Strategi Validasi
Eksperimen pemodelan dilakukan di notebook `Modeling.ipynb` dengan fokus pengujian validasi:
* **Validasi Silang:** Membandingkan **K-Fold** standar dengan **Stratified K-Fold** (5-Fold) untuk menguji stabilitas model.
* **Hyperparameter Tuning:** Mengoptimalkan performa model menggunakan **GridSearchCV** dan **RandomizedSearchCV**.

## 📈 Hasil Eksperimen & Analisis
* **Stabilitas Validasi:** **Stratified K-Fold** terbukti menghasilkan performa metrik yang jauh lebih stabil di setiap lipatan (*fold*) dibandingkan K-Fold biasa.
* **Fenomena K-Fold Biasa:** Terjadi penurunan akurasi yang cukup drastis secara konsisten pada **Fold 4** dan **Fold 5** saat menggunakan K-Fold biasa. Hal ini berhasil diatasi oleh Stratified K-Fold yang menjaga proporsi sebaran target tetap seimbang di setiap pecahan data.
* **Stabilitas Performa:** **Fold 1** dan **Fold 2** secara umum mencatatkan performa akurasi tertinggi dan paling stabil di seluruh skenario pengujian.
* **Metode Optimal:** Kombinasi **Stratified K-Fold + Hyperparameter Tuning (GridSearchCV/RandomizedSearchCV)** menghasilkan model dengan akurasi terbaik dan varians performa paling rendah (*highly robust model*).

## 📂 Struktur Repositori
```text
├── Data Preprocessing (Data Cleaning, Feature Engineering).ipynb  # Notebook Preprocessing
├── Modeling.ipynb                                                 # Notebook Pemodelan & Validasi
├── heart_statlog_cleveland_hungary_final.csv                     # Dataset Mentah (Raw)
├── heart_statlog_cleveland_hungary_final_cleaned.csv             # Dataset Hasil Preprocessing
└── README.md                                                      # Dokumentasi Proyek
