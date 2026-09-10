# Kartu data: ulasan sintetis

1. **Asal:** contoh buatan untuk kursus PyTorch NLP ini. Tidak dikumpulkan dari pengguna, situs ulasan, atau data pribadi. Lisensi MIT mengikuti repositori.
2. **Ukuran:** 480 baris, masing-masing 240 label positif dan 240 negatif. Domain produk dan layanan masing-masing 240 baris.
3. **Kolom:** text berisi ulasan, label berisi 0=negatif atau 1=positif, split berisi train/val/test, domain berisi produk/layanan, template_id mengidentifikasi kelompok pola kalimat.
4. **Pembagian:** 288 latih, 96 validasi, 96 uji. Pola kalimat disisihkan menurut kelompok sebelum diperluas dengan nama barang atau layanan serta frasa opini. Setiap split seimbang menurut label.
5. **Tujuan:** memeriksa tokenisasi, embedding, padding, pelatihan, dan evaluasi secara cepat. Contoh negasi sederhana membantu menunjukkan batas model rata-rata embedding.
6. **Batas:** kosakata opini sengaja berulang antarsplit. Bahasa bersifat templatis, tanpa ironi, sarkasme, ragam dialek, atau ambiguitas label. Hasil tidak dapat dipakai sebagai ukuran kemampuan sentimen dunia nyata. Pembagian berdasarkan template tidak menghilangkan seluruh kemiripan semantik.
7. **Transfer domain:** kosakata dibangun dari data latih domain sumber. Nama layanan yang tidak ada dalam kosakata sumber menjadi UNK. Demonstrasi ini mengajarkan pemindahan bobot, bukan mengklaim manfaat transfer lintas domain yang kuat.
8. **Data eksternal:** notebook 10 mengunduh SST-2 secara terpisah. Kartu dataset tersebut mencantumkan lisensi sebagai unknown pada saat diperiksa. Tinjau ketentuan sumber sebelum penggunaan atau redistribusi lebih lanjut. Data SST-2 dan bobot BERT tidak dibundel dalam kursus.
