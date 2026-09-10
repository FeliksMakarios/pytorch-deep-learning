# Belajar PyTorch melalui NLP

Jalur pembelajaran berbahasa Indonesia dengan seluruh studi kasus berbasis teks. Mulai dari tensor dan pembaruan embedding, lanjut ke klasifikasi sentimen, model urutan, pembelajaran transfer, attention, dan aplikasi prediksi.

Materi ini merupakan edisi ringkas yang mengikuti alur pedagogis kursus [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning). Kedalaman dan jumlah contohnya belum menyamai seluruh buku, slide, dan video sumber. Notebook NLP ditulis sebagai materi pendamping mandiri. Studi kasus inti menggunakan ulasan sintetis berbahasa Indonesia, sedangkan notebook tambahan memakai SST-2 berbahasa Inggris dengan BERT kecil.

## 1. Mulai belajar

1. Kuasai dasar Python, fungsi, kelas sederhana, serta pengertian data latih dan uji.
2. Mulai dari notebook 00. Baca penjelasan sebelum menjalankan kode.
3. Periksa bentuk tensor dan keluaran setiap tahap. Kerjakan latihan sebelum membaca pembahasan.
4. Jalankan notebook secara berurutan. Setiap notebook juga dapat dijalankan mandiri setelah paket tersedia.
5. Gunakan notebook 10 setelah memahami data, pelatihan, dan pembelajaran transfer.

Penjelasan memakai Bahasa Indonesia. Nama API Python dan istilah teknis yang membantu pencarian dokumentasi tetap dicantumkan. Tidak diperlukan GPU untuk rangkaian inti.

## 2. Daftar notebook

| Modul | Materi dan studi kasus | Buka |
|---|---|---|
| 00 | Tensor teks, tokenisasi, embedding, transpose, mask, gradien, pembaruan SGD | [Notebook](notebooks/00_tensor_teks.ipynb) |
| 01 | Klasifikasi sentimen bag-of-words, loop pelatihan manual, validasi, checkpoint | [Notebook](notebooks/01_alur_pelatihan.ipynb) |
| 02 | Rata-rata embedding, mini-batch, baseline, macro-F1, analisis kesalahan | [Notebook](notebooks/02_klasifikasi_teks.ipynb) |
| 03 | RNN manual, nn.RNN, LSTM, GRU, panjang urutan dan packed sequence | [Notebook](notebooks/03_rnn_lstm_gru.ipynb) |
| 04 | CSV khusus, audit duplikasi, OOV, pemotongan teks, Dataset dan DataLoader | [Notebook](notebooks/04_dataset_teks.ipynb) |
| 05 | Kode modular, pelatihan melalui terminal, konfigurasi, pemuatan model | [Notebook](notebooks/05_kode_modular.ipynb) |
| 06 | Transfer embedding dari ulasan produk ke layanan, frozen encoder, fine-tuning | [Notebook](notebooks/06_transfer_learning.ipynb) |
| 07 | Proyek 1: perbandingan beberapa model dan seed, log JSON, kurva dan evaluasi | [Notebook](notebooks/07_pencatatan_eksperimen.ipynb) |
| 08 | Proyek 2: attention manual, pengujian terhadap PyTorch, Transformer kecil | [Notebook](notebooks/08_attention_transformer.ipynb) |
| 09 | Proyek 3: checkpoint prediksi, kontrak masukan, aplikasi sentimen lokal | [Notebook](notebooks/09_aplikasi_sentimen.ipynb) |
| 10 | Tambahan: SST-2, tokenizer subword, fine-tuning BERT kecil, save_pretrained | [Notebook](notebooks/10_bert_data_nyata.ipynb) |

## 3. Hubungan dengan materi sumber

| Materi sumber | Adaptasi NLP |
|---|---|
| Dasar PyTorch | Operasi tensor dengan indeks token dan embedding |
| Alur PyTorch | Alur pelatihan klasifikasi sentimen |
| Klasifikasi jaringan saraf | Bag-of-words dan representasi embedding |
| Computer vision | RNN, LSTM, dan GRU untuk urutan teks |
| Dataset khusus | Dataset CSV teks, vocabulary, padding, OOV |
| Kode modular | Modul data, model, pelatihan, dan inferensi NLP |
| Pembelajaran transfer | Transfer embedding dan tambahan fine-tuning BERT |
| Pencatatan eksperimen | Perbandingan classifier teks dengan beberapa seed |
| Replikasi makalah | Implementasi dan verifikasi mekanisme attention, bukan klaim replikasi eksperimen makalah |
| Penerapan model | Aplikasi prediksi sentimen lokal dan paket checkpoint |

Fokus proyek berkelanjutan adalah **klasifikasi sentimen** agar perhatian tetap pada cara kerja PyTorch. NER, peringkasan, penerjemahan, dan tanya jawab belum menjadi modul terimplementasi dalam edisi ini.

## 4. Menjalankan secara lokal

1. Ambil hanya direktori NLP agar tidak mengunduh seluruh aset gambar repositori sumber.

   ```bash
   git clone --depth 1 --filter=blob:none --sparse --branch nlp-learning-path https://github.com/FeliksMakarios/pytorch-deep-learning.git
   cd pytorch-deep-learning
   git sparse-checkout set nlp
   cd nlp
   ```

2. Buat lingkungan Python. Pengujian paket dilakukan pada Python 3.12.

   ```bash
   python -m venv .venv
   ```

   Aktifkan melalui `source .venv/bin/activate` pada Linux/macOS atau `.venv\Scripts\Activate.ps1` pada PowerShell.

3. Instal dependensi dan buka notebook.

   ```bash
   python -m pip install -r requirements.txt
   python -m jupyterlab
   ```

4. Untuk notebook 10, pasang tambahan berikut. Data dan model akan diunduh pada eksekusi pertama.

   ```bash
   python -m pip install -r requirements-hf.txt
   ```

## 5. Google Colab

1. Buka [notebook 00 di Colab](https://colab.research.google.com/github/FeliksMakarios/pytorch-deep-learning/blob/nlp-learning-path/nlp/notebooks/00_tensor_teks.ipynb).
2. Jalankan sel penyiapan. Pada Colab, sel tersebut mengambil direktori NLP dari cabang `nlp-learning-path` jika paket belum tersedia.
3. Untuk notebook lain, ganti nama berkas pada URL Colab sesuai tabel modul.
4. Untuk notebook 10, jalankan `%pip install -r /content/pytorch-deep-learning-nlp/nlp/requirements-hf.txt` setelah sel penyiapan. Mulai ulang sesi bila Colab meminta setelah perubahan dependensi.

Jalur Colab disediakan untuk digunakan, tetapi status verifikasi lingkungan Colab dicatat terpisah dari pengujian Python lokal pada [VALIDATION.md](VALIDATION.md).

## 6. Data pembelajaran

[reviews.csv](data/reviews.csv) berisi **480 ulasan sintetis** yang disusun untuk kursus ini. Label 0 berarti negatif dan 1 positif. Pembagian berisi 288 data latih, 96 validasi, dan 96 uji. Setiap kelompok pola kalimat hanya berada pada satu split. Lihat [kartu data](data/README.md).

Data ini berguna untuk memeriksa mekanisme dan bentuk tensor. Polanya sederhana dan tidak mewakili variasi bahasa nyata. Akurasi tinggi pada data ini tidak membuktikan kemampuan aplikasi menghadapi ulasan pengguna.

Untuk data sendiri, siapkan CSV dengan kolom `text,label,split`. Pertahankan label 0/1 atau sesuaikan seluruh kontrak model dan evaluasi bila menambah kelas. Pembentukan kosakata dilakukan dari data latih saja.

Notebook 10 memakai 512 contoh latih SST-2. Sebanyak 256 contoh dari validation resmi dibagi menjadi 128 validasi lokal dan 128 data tahanan lokal. Ini bukan evaluasi test resmi SST-2. Dataset dan bobot eksternal tidak disertakan dalam repositori.

## 7. Pelatihan dan aplikasi

1. Latih classifier ringan dari direktori `nlp`.

   ```bash
   python -m nlp_course.train --epochs 12 --seed 42
   ```

2. Gunakan data sendiri jika diperlukan.

   ```bash
   python -m nlp_course.train --csv data/ulasan_sendiri.csv --epochs 20
   ```

3. Jalankan aplikasi lokal.

   ```bash
   python -m pip install -r requirements-app.txt
   python app.py
   ```

Checkpoint dan log yang dihasilkan berada di `artifacts/` dan dikecualikan dari Git. Aplikasi memakai `share=False`. Paket ini belum diterbitkan sebagai layanan daring.

## 8. Verifikasi

Lihat [VALIDATION.md](VALIDATION.md) untuk hasil, versi dependensi, dan batas pengujian.

1. Jalankan pengujian mekanisme.

   ```bash
   python -m unittest discover -s tests -v
   ```

2. Jalankan ulang notebook inti dalam proses Python terpisah.

   ```bash
   python -m pip install -r requirements-test.txt
   python scripts/run_notebooks.py
   ```

3. Tambahkan notebook eksternal jika dependensi serta jaringan tersedia.

   ```bash
   python scripts/run_notebooks.py --with-hf
   ```

Pelaksana notebook menyimpan keluaran teks dan grafik ke notebook. Ia menjalankan sel kode berurutan di proses Python baru per notebook, tanpa server kernel Jupyter. Perintah `--with-hf` membutuhkan unduhan dataset serta model pada pemakaian pertama.

## 9. Atribusi dan rujukan

Struktur belajar terinspirasi kursus Daniel Bourke. Lisensi MIT dan atribusi sumber tetap tersedia pada [LICENSE repositori](../LICENSE). Kode dan contoh sintetis tambahan ini menggunakan lisensi MIT yang sama.

- [Kursus sumber](https://github.com/mrdbourke/pytorch-deep-learning)
- [Dokumentasi PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Implementasi Embedding PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/sparse.py)
- [Implementasi Transformer PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)
- [Panduan klasifikasi teks Transformers](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Kartu model BERT-Tiny](https://huggingface.co/prajjwal1/bert-tiny)
- [Kartu dataset SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)
