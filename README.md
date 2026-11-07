# Music Recommendation System 🎵

### Penyusun Proyek (Big Data Analytics):
1. Ricky Hardian (NRP 6032252067)  
2. Mulyana (NRP 6032252055)  

---

## 📝 Deskripsi Proyek
Proyek ini adalah purwarupa sistem rekomendasi musik berbasis konten (Content-Based Filtering) yang dikombinasikan dengan pembobotan linier tahun rilis sebagai tie-breaker untuk mengatasi kesamaan representasi genre (*ties*). Sistem ini juga mengintegrasikan perluasan genre (*genre expansion*) menggunakan pola aturan asosiasi (*association rules*) yang diperoleh dari algoritma **Apriori**.

## 🧪 Metode dan Algoritma
1. **Content-Based Filtering (Utama):** Representasi TF-IDF pada penandaan genre multi-label dan perhitungan Cosine Similarity.
2. **Algoritma Apriori (Pendukung):** Menemukan korelasi asosiasi antargenre untuk memperluas pencarian (*genre expansion*).
3. **Knowledge-Based Lexicon (Mood):** Pemetaan manual berbasis asumsi dari 6 kategori suasana hati (*mood*) ke bobot genre terkait.
4. **Unified Query Engine:** Penyatuan seluruh kueri ke dalam representasi vektor gabungan terpadu.

## 🔍 Empat Mode Pencarian Dashboard
- **Lagu:** Mencari lagu serupa berdasarkan kemiripan representasi genre dan kedekatan tahun rilis.
- **Genre:** Rekomendasi lagu berdasarkan masukan genre dengan chip pintasan asosiasi Apriori.
- **Mood:** Rekomendasi lagu adaptif menyesuaikan 6 tombol emoji suasana hati.
- **Perpaduan:** Fusi fleksibel multi-input dengan slider pembobotan kustom.

## 📁 Struktur Folder Proyek
```
|-- notebooks/           # Notebook analisis .ipynb
|-- data/                # Dataset mentah dan bersih
|-- models/              # Model serializer & data konfigurasi JSON
|-- .streamlit/          # Konfigurasi visual Streamlit
|-- app.py               # Kode utama antarmuka Streamlit
|-- recommender.py       # Logika backend mesin kueri terpadu
|-- requirements.txt     # Daftar dependensi library
```

## 🚀 Cara Menjalankan Secara Lokal
1. Pastikan Anda telah menginstal Python 3.10 atau versi di atasnya.
2. Kloning repositori ini dan masuk ke dalam foldernya:
   ```bash
   git clonehttps://github.com/rickhardy12/rekomendasi_musik_kamu.git
   cd music-recommendation-system
   ```
3. Pasang semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

## 🌐 Link Dashboard Aktif (Cloud)
Anda dapat mengakses aplikasi web yang telah dideploy secara publik pada tautan berikut:  
👉 **https://rekomendasimusikkamu.streamlit.app/#temukan-rekomendasi-lagumu**

## ⚠️ Keterbatasan
Sistem ini tidak merepresentasikan aspek psikologis selera musik manusia secara mutlak dan belum didukung data interaksi dinamis pengguna (*user-item rating*).
