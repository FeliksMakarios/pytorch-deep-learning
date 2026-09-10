# Catatan verifikasi

Pengujian dilakukan pada 10 September 2026 dengan Python 3.12 dan CPU.

## 1. Notebook inti 00–09

**Lulus:** seluruh 72 sel kode dijalankan berurutan, dengan proses Python baru untuk setiap notebook. Keluaran teks dan grafik disimpan di notebook. Hasil per notebook tersedia pada [execution_report.json](execution_report.json).

Pelaksana menggunakan eksekusi Python langsung dan backend Matplotlib Agg. Kernel Jupyter berbasis soket tidak dapat digunakan di lingkungan pengujian ini. Karena itu hasil ini tidak diklaim sebagai pengujian antarmuka Jupyter atau Colab.

Pemeriksaan yang berhasil mencakup:

1. Pembaruan embedding setara dengan rumus SGD dan gradien PAD tetap nol.
2. Keluaran satu langkah RNN manual cocok dengan nn.RNN.
3. Penambahan PAD tidak mengubah prediksi MeanClassifier, RecurrentClassifier, atau TinyTransformer di luar toleransi numerik.
4. Attention manual cocok dengan scaled_dot_product_attention menggunakan mask yang benar.
5. Bobot frozen tidak berubah selama pelatihan kepala klasifikasi.
6. Model yang dimuat ulang menghasilkan logits yang sama.
7. Program pelatihan CLI selesai dan menulis checkpoint yang dapat digunakan.
8. Data tidak mengandung duplikat persis atau kelompok template yang melintasi split.

## 2. Pengujian mekanisme

**Lulus:** enam pengujian unittest pada `tests/test_nlp.py`, termasuk subpengujian tiga arsitektur. Pengujian mencakup OOV dan teks kosong, pembagian data, padding dan gradien, metrik dengan hasil yang diketahui, checkpoint dan kontrak masukan, serta freezing.

Tidak digunakan ambang akurasi sintetis sebagai syarat kelulusan. Tujuan pengujian adalah memeriksa mekanisme yang benar.

## 3. Notebook 10

**Lulus:** tujuh sel kode dieksekusi dengan dataset SST-2 dan bobot BERT-Tiny yang benar-benar diunduh. Pemuatan tokenizer, pembagian data, dua epoch fine-tuning, evaluasi tahanan lokal, save_pretrained, dan kesamaan logits sesudah pemuatan ulang berhasil.

Pengujian memakai 512 contoh latih, 128 validasi lokal, dan 128 tahanan lokal. Hasil demonstrasi tahanan lokal: accuracy 0.578125 dan macro-F1 sekitar 0.527223. Ukuran kecil serta dua epoch sengaja membatasi komputasi. Hasil ini bukan skor benchmark resmi dan belum menunjukkan model siap digunakan pada pengguna nyata.

Dua penyesuaian checkpoint lama diperlukan pada Transformers 5: vocabulary dimuat secara eksplisit melalui BertTokenizer, dan jenis model dinyatakan melalui BertConfig serta BertForSequenceClassification. Konversi bobot melalui layanan eksternal dinonaktifkan. Model hasil penyimpanan memakai konfigurasi baru yang dapat dibuka melalui AutoModel.

## 4. Lingkungan dependensi

| Paket | Versi terpasang pada pengujian |
|---|---|
| Python | 3.12 |
| torch | 2.14.0 |
| matplotlib | 3.10.8 |
| nbformat | 5.11.1 |
| transformers | 5.17.0 |
| datasets | 5.0.1 |
| tokenizers | 0.23.2 |
| huggingface-hub | 1.31.0 |

Dependensi dasar dan tambahan dipisahkan agar notebook inti tidak memerlukan Transformers atau unduhan model. Lingkungan dengan proksi SOCKS mungkin memerlukan `python -m pip install 'httpx[socks]'` untuk unduhan Hugging Face. Paket tersebut digunakan pada lingkungan pengujian.

## 5. Batas verifikasi

1. GPU, Apple MPS, dan runtime Google Colab belum diuji langsung.
2. Sintaks app.py diperiksa. Fungsi pemuatan dan prediksi yang dipakai aplikasi telah diuji. Server dan antarmuka Gradio belum diuji secara interaktif.
3. Aplikasi belum diterbitkan daring.
4. Kode dan metrik data sintetis tidak memvalidasi kinerja pada data pengguna nyata.
5. Materi ini berfokus pada klasifikasi teks. Tidak ada klaim bahwa semua tugas NLP sudah tercakup.
