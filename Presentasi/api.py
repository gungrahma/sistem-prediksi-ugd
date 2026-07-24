from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="API Triase UGD AI",
    description="RESTful API untuk mendeteksi status kegawatdarurat pasien berdasarkan teks keluhan",
    version="1.0"
)

try:
    print("Memuat model Random Forest...")
    model_rf = joblib.load('model_triase_rf.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model dan vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model atau vectorizer: {e}")

class DataPasien(BaseModel):
    keluhan: str

@app.post("/predict")
def prediksi_triase(data: DataPasien):
    if not data.keluhan.strip():
        raise HTTPException(status_code=400, detail="Teks Keluhan tidak boleh kosong")
    
    vektor_teks = vectorizer.transform([data.keluhan]).toarray()

    kelas_prediksi = model_rf.predict(vektor_teks)[0]
    probabilitas = model_rf.predict_proba(vektor_teks)[0]

    status_hasil = "True Emergency" if kelas_prediksi == 1 else "False Alarm"

    keyakinan = probabilitas[1] if kelas_prediksi == 1 else probabilitas[0]

    return {
        "pesan_status": "Berhasil memproses data",
        "input_keluhan": data.keluhan,
        "hasil_prediksi": status_hasil,
        "tingkat_keyakinan": f"{keyakinan * 100:.1f}%",
        "probabilitas_gawat_darurat": float(probabilitas[1])
    }