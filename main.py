import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. SETUP LOGIKA SISTEM REKOMENDASI (AI)
# ==========================================

# Membaca dataset wisata Indonesia yang Anda miliki
try:
    df = pd.read_csv('list wisata.csv')
except FileNotFoundError:
    raise FileNotFoundError("Pastikan file 'dataset_wisata.csv' berada di folder yang sama dengan skrip ini.")

# Menggabungkan kolom tekstual untuk dianalisis oleh algoritma Content-Based Filtering
# Menyatukan Kategori, Deskripsi Fitur, dan Tags menjadi satu korpus teks (Metadata)
df['metadata'] = df['Kategori'].fillna('') + " " + df['Deskripsi_Fitur'].fillna('') + " " + df['Fitur_Kunci (Tags)'].fillna('')
df['metadata'] = df['metadata'].str.lower()

# Inisialisasi TF-IDF Vectorizer dengan menyisipkan kata henti (stop words) bahasa Indonesia umum
# Ini mencegah kata hubung mempengaruhi bobot perhitungan kemiripan teks
stop_words_id = ['yang', 'di', 'dan', 'dengan', 'untuk', 'ke', 'buat', 'ada', 'dong', 'saya', 'ingin', 'mau', 'bisa', 'kamu']
tfidf = TfidfVectorizer(stop_words=stop_words_id)

# Mentransformasikan teks metadata menjadi matriks angka (TF-IDF Matrix)
tfidf_matrix = tfidf.fit_transform(df['metadata'])

def dapatkan_rekomendasi(user_query: str, top_n: int = 3) -> List[dict]:
    """
    Fungsi utama untuk menghitung Cosine Similarity antara preferensi teks user
    dengan kolom metadata seluruh tempat wisata di dataset.
    """
    # Mengubah input chat user menjadi vektor TF-IDF
    query_vec = tfidf.transform([user_query.lower()])
    
    # Menghitung skor kemiripan menggunakan rumus Cosine Similarity
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Membuat salinan dataframe agar tidak merusak data asli dan menyisipkan skor kemiripan
    df_result = df.copy()
    df_result['score'] = similarity_scores
    
    # Mengurutkan destinasi berdasarkan skor kemiripan tertinggi
    rekomendasi = df_result.sort_values(by='score', ascending=False)
    
    # Mengambil objek dengan skor kecocokan di atas 0 (relevan) sebanyak top_n hasil
    rekomendasi_valid = rekomendasi[rekomendasi['score'] > 0].head(top_n)
    
    # Mengonversi hasil dataframe Pandas menjadi format List of Dictionary (siap jadi JSON)
    daftar_rekomendasi = []
    for _, row in rekomendasi_valid.iterrows():
        daftar_rekomendasi.append({
            "id": int(row['ID']),
            "nama_wisata": row['Nama_Wisata'],
            "provinsi": row['Provinsi'],
            "kota_kabupaten": row['Kota_Kabupaten'],
            "kategori": row['Kategori'],
            "harga_tiket": int(row['Harga_Tiket']),
            "deskripsi": row['Deskripsi_Fitur'],
            "tags": row['Fitur_Kunci (Tags)'],
            "skor_kemiripan": float(row['score'])
        })
        
    return daftar_rekomendasi


# ==========================================
# 2. SETUP INTERFACES API (FASTAPI)
# ==========================================

app = FastAPI(title="Travel Recommendation Engine API", version="1.0")

# Mengaktifkan CORS agar teman bagian Frontend Web bisa menembak API ini walau berbeda port/domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Di fase produksi, ini bisa diganti dengan URL Web tim Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menentukan struktur data input JSON yang wajib dikirim oleh Frontend
class ChatInput(BaseModel):
    pesan: str
    jumlah_rekomendasi: int = 3

@app.post("/api/chat")
async def chat_bot_endpoint(input_data: ChatInput):
    """
    Endpoint POST yang akan dipanggil oleh bagian Website Frontend.
    Menerima input teks chat dan membalas dengan teks AI serta data rekomendasi terstruktur.
    """
    user_message = input_data.pesan
    n_results = input_data.jumlah_rekomendasi
    
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong")
        
    # Memproses rekomendasi berdasarkan input pesan pengguna
    list_destinasi = dapatkan_rekomendasi(user_message, top_n=n_results)
    
    # Membuat variasi teks jawaban AI pembuka yang dinamis
    if list_destinasi:
        teks_balasan = f"Halo! Berdasarkan keinginan kamu yang mencari '{user_message}', berikut adalah {len(list_destinasi)} rekomendasi tempat wisata terbaik di Indonesia yang sangat cocok buat kamu:"
    else:
        teks_balasan = f"Maaf, aku belum menemukan tempat wisata di Indonesia yang spesifik cocok dengan kata kunci '{user_message}'. Coba ketik preferensi lain seperti 'pantai sepi' atau 'gunung dingin berkabut'!"
        
    # Mengembalikan response terstruktur JSON ke Frontend Website
    return {
        "jawaban_bot": teks_balasan,
        "rekomendasi_wisata": list_destinasi
    }

# Untuk menjalankan server secara lokal langsung dari file ini
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)