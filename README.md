# AI Analyst — Dashboard Dark Lavender & Pink

Tampilan diperbarui mengikuti referensi dark dashboard yang diunggah: sidebar navy, kartu KPI sederhana, aksen ungu/pink, tata letak lapang, dan grafik gelap. Analisis satu dan dua periode tetap saling eksklusif, serta integrasi GroqCloud dipertahankan.

## Menjalankan
1. Buka folder ini di VS Code.
2. `python -m pip install -r requirements.txt`
3. Salin `.streamlit/secrets.toml.example` ke `.streamlit/secrets.toml`, isi `GROQ_API_KEY` **baru** milik Anda. Jangan bagikan kunci tersebut.
4. `python -m streamlit run app.py`

## Penting
- Tema visual terinspirasi gambar, bukan replika persis layout aplikasi manajemen tugas (karena kebutuhan dashboard analitik berbeda).
- Hasil analisis model, kategorisasi keluhan, GroqCloud, dan ekspor mengikuti aplikasi sebelumnya.
- Jalankan pengujian visual di browser setelah instalasi; koneksi GroqCloud perlu API key aktif.
