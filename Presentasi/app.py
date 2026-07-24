import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(
    page_title="Emergency (UGD) Triage AI -s Prototype dan Roadmap",
    layout="centered"
)

@st.cache_resource
def load_triage_resources():
    # Menggunakan cache agar model hanya di-load sekali ke memori
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('model_triase_rf.pkl')
    return vectorizer, model

try:
    tfidf, model = load_triage_resources()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .pkl ada di folder yang sama. Error: {e}")
    st.stop()


st.title("Sistem Prediksi Triase UGD (Prototype)")
st.subheader("Fase Awal: Algoritma Klasifikasi Teks Keluhan & Diagnosa Awal")

st.markdown("""
Aplikasi ini adalah bentuk **Interactive UI Prototype** dari *codebase* model klasifikasi triase UGD.
Sistem memproses teks diagnosa masuk menggunakan **TF-IDF + Random Forest** untuk menentukan prioritas penanganan.
""")

# Informasi Status Sistem saat ini
st.info("**Informasi:** Sistem ini berjalan di atas *codebase* eksperimental. Belum terintegrasi dengan SIMRS dan belum melalui validasi klinis menyeluruh oleh Dokter Spesialis Emergensitas.")

st.divider()


st.write("### Simulasi Input Diagnosa Pasien")

# Opsi A: Input Teks Manual
diagnosa_input = st.text_area(
    "Masukkan Keluhan Awal Pasien:",
    placeholder="Contoh: Pasien datang dengan trauma kepala akibat kecelakaan, kesadaran menurun dan pendarahan aktif..."
)

# Opsi B: Contoh Cepat untuk Pengujian Singkat (Sangat berguna saat presentasi!)
st.write("*Atau pilih contoh kasus di bawah ini untuk simulasi cepat:*")
contoh_kasus = {
    "Pilih contoh kasus...": "",
    "Kasus Gawat Darurat (True Emergency)": "Fraktur terbuka regio femur dextra dengan perdarahan aktif post kll",
    "Kasus Non-Darurat (False Alarm)": "Pasien mengeluhkan batuk pilek ringan sejak 3 hari lalu tanpa disertai sesak napas"
}
pilihan = st.selectbox("Gunakan template kasus:", list(contoh_kasus.keys()))

# Jika template dipilih, timpa input manual
if pilihan != "Pilih contoh kasus...":
    diagnosa_input = contoh_kasus[pilihan]
    st.text_area("Masukkan Teks Diagnosa / Keluhan Awal Pasien: (Diisi otomatis)", value=diagnosa_input, key="disabled_input", disabled=True)


if st.button("Jalankan Prediksi Triase", type="primary"):
    if diagnosa_input.strip() == "":
        st.warning("Silakan masukkan teks diagnosa terlebih dahulu.")
    else:
        with st.spinner("Model AI sedang menganalisis teks klinis..."):
            # Simulasi delay pemrosesan agar UI terasa responsif dan nyata
            time.sleep(0.6)
            
            # --- Tahap Pra-pemrosesan (Sesuaikan jika Anda punya fungsi custom cleansing) ---
            # Catatan: Teks input harus dibersihkan sama persis dengan proses 'Diagnose_Bersih' di Jupyter
            diagnosa_clean = [diagnosa_input.lower()] 
            
            # --- Tahap Transformasi TF-IDF ---
            X_vectorized = tfidf.transform(diagnosa_clean)
            
            # --- Tahap Prediksi Model ---
            prediksi = model.predict(X_vectorized)[0]
            
            # Probabilitas untuk memperkuat keyakinan hasil (Decision Support)
            probabilitas = model.predict_proba(X_vectorized)[0]
            prob_emergency = probabilitas[1] * 100 # Indeks 1 biasanya untuk True Emergency
            
        # Tampilan Hasil berdasarkan Hasil Klasifikasi
        st.success("Analisis Selesai!")
        
        st.write("### Hasil Prediksi Sistem:")
        
        if prediksi == 1: # Asumsi 1 = True Emergency
            st.error("## TRUE EMERGENCY")
            st.metric(label="Tingkat Keyakinan Model AI", value=f"{prob_emergency:.2f}%")
            st.markdown("""
            **Rekomendasi Tindakan Internal:**
            *   Segera arahkan pasien ke Bed Resusitasi / Trauma.
            *   Notifikasi otomatis dikirimkan ke Dokter Jaga UGD.
            """)
        else: # Asumsi 0 = False Alarm / Non-Emergency
            st.warning("## FALSE ALARM (NON-EMERGENCY)")
            st.metric(label="Tingkat Keyakinan Model AI", value=f"{(100 - prob_emergency):.2f}%")
            st.markdown("""
            **Rekomendasi Tindakan Internal:**
            *   Pasien dapat diarahkan ke area triase hijau atau poliklinik rawat jalan.
            *   Optimalisasi antrean untuk memprioritaskan pasien kritis.
            """)

st.divider()

st.write("### Rencana Masa Depan Produk (Roadmap)")
col1, col2 = st.columns(2)
with col1:
    st.checkbox("Integrasi Validasi Manual oleh Dokter (Human-in-the-Loop)", value=False, disabled=True)
    st.checkbox("Penyempurnaan Model dengan Text Preprocessing yang Lebih Kompleks", value=False, disabled=True)
with col2:
    st.checkbox("Pembuatan API Backend menggunakan Framework Skala Produksi", value=False, disabled=True)