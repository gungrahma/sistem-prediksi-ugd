# Emergency Room (UGD) Triage AI System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

Sistem pendukung keputusan (*Decision Support System*) untuk memprediksi serta mengategorikan tingkat kegawatdaruratan pasien UGD berdasarkan teks keluhan medis awal.

---

## Latar Belakang & Masalah

Proses Triase di Unit Gawat Darurat (UGD) merupakan tahap kritis di mana keselamatan pasien ditentukan oleh kecepatan dan ketepatan pemilahan kondisi gawat darurat (*True Emergency*) vs non-darurat (*False Alarm*). 

Sistem ini dibangun untuk membantu tenaga medis mengurangi bias subjektif serta memberikan *second opinion* yang terkalibrasi secara probabilitas saat proses *screening* awal.

---

## Arsitektur Model & Eksperimen

Proyek ini membandingkan dan mengevaluasi tiga pendekatan algoritma yang berbeda:

1. **Multinomial Naive Bayes (Baseline Probabilistik):** Mengukur baseline performa menggunakan asumsi independensi fitur.
2. **Random Forest Classifier (Ensemble Approach):** Digunakan sebagai otak utama pada antarmuka prototipe karena stabilitasnya terhadap data tabular/sparse teks.
3. **Artificial Neural Network / Multi-Layer Perceptron (Deep Learning):**
   - **Arsitektur:** Dense (128 units, ReLU) $\rightarrow$ Dropout (0.5) $\rightarrow$ Dense (64 units, ReLU) $\rightarrow$ Dropout (0.3) $\rightarrow$ Output (1 unit, Sigmoid).
   - **Optimasi:** Adam Optimizer dengan *Binary Crossentropy Loss*.
   - **Regularisasi & Stabilitas:** Menggunakan *Early Stopping* (`monitor='val_loss'`, `patience=5`) untuk mencegah *overfitting* dan menghasilkan output probabilitas yang realistis.