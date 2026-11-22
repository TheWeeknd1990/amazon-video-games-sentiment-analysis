# amazon-video-games-sentiment-analysis
Sentiment analysis of Amazon Video Games reviews using TF-IDF and Logistic Regression.
# Analisis Sentimen Ulasan Amazon Video Games

## 1. Deskripsi Proyek

Proyek ini bertujuan membangun model Machine Learning untuk mengklasifikasikan sentimen ulasan pelanggan Amazon pada kategori Video Games menjadi dua kelas, yaitu positif dan negatif.

Model menggunakan kombinasi judul ulasan (*review title*) dan isi ulasan (*review text*) sebagai input untuk memprediksi sentimen pelanggan.

## 2. Dataset

Dataset yang digunakan berasal dari **Amazon Reviews 2023** dengan kategori `Video_Games`.

**Kriteria pengolahan dataset:**
- Periode ulasan: sampai dengan 31 Desember 2019.
- Input: kombinasi kolom `title` dan `text`.
- Sentimen negatif (label 0): rating 1 dan 2.
- Sentimen positif (label 1): rating 4 dan 5.
- Rating 3 (netral): dihapus dari proses pemodelan.

**Jumlah data yang digunakan:**

| Pembagian Data | Jumlah |
|---|---:|
| Training | 30.000 |
| Validation | 10.000 |
| Total | 40.000 |

Data training menggunakan ulasan sebelum 1 Januari 2019, sedangkan data validation menggunakan ulasan sepanjang tahun 2019.

Pengambilan sampel dilakukan dari 300.000 baris pertama yang dipindai menggunakan metode reservoir sampling. Dengan demikian, data yang digunakan merupakan subset dataset, bukan keseluruhan data Amazon Reviews 2023.

## 3. Metodologi dan Pemodelan

Proses pengembangan model dilakukan melalui beberapa tahapan:

1. **Data Loading:** Memuat dataset ulasan Amazon kategori Video Games.
2. **Data Preprocessing:** Menggabungkan judul dan isi ulasan serta menangani nilai kosong.
3. **Label Encoding:** Mengubah rating menjadi label sentimen positif dan negatif.
4. **Data Splitting:** Membagi data berdasarkan periode waktu (*temporal split*).
5. **Feature Extraction:** Menggunakan TF-IDF untuk mengubah teks menjadi representasi numerik.
6. **Model Training:** Melatih model klasifikasi menggunakan Logistic Regression.
7. **Model Evaluation:** Mengevaluasi performa menggunakan Accuracy, Macro Precision, Macro Recall, dan Macro-F1.

### Konfigurasi Model

| Komponen | Konfigurasi |
|---|---|
| Feature Extraction | TF-IDF |
| N-gram Range | (1, 2) |
| Max Features | 50.000 |
| Minimum Document Frequency | 2 |
| Classifier | Logistic Regression |
| Class Weight | Balanced |
| Solver | Liblinear |
| Maximum Iterations | 1.000 |

### Alasan Pemilihan Metode

TF-IDF digunakan untuk memberikan bobot pada kata berdasarkan tingkat kepentingannya dalam dokumen. Penggunaan unigram dan bigram membantu model mengenali kata maupun kombinasi dua kata yang memiliki makna sentimen.

Logistic Regression dipilih karena efisien, relatif ringan, dan cocok sebagai model baseline untuk klasifikasi teks.

Parameter `class_weight="balanced"` digunakan untuk membantu menangani ketidakseimbangan distribusi kelas pada dataset.

## 4. Hasil Evaluasi Model

Evaluasi dilakukan menggunakan data validation yang berasal dari periode tahun 2019.

| Metrik | Hasil |
|---|---:|
| Accuracy | 94,51% |
| Macro Precision | 85,66% |
| Macro Recall | 93,60% |
| **Macro-F1** | **89,01%** |

Macro-F1 digunakan sebagai metrik evaluasi utama karena memperhitungkan performa kedua kelas sentimen secara seimbang.

Berdasarkan hasil evaluasi, model memperoleh Macro-F1 sebesar **89,01%**, yang menunjukkan kemampuan klasifikasi sentimen yang baik pada data validation yang digunakan.

## 5. Struktur Repository

```text
amazon-video-games-sentiment/
│
├── notebooks/
│   └── amazon_sentiment_analysis.ipynb
│
├── models/
│   └── amazon_sentiment_model.joblib
│
├── results/
│   └── evaluation_results.csv
│
├── README.md
│
└── requirements.txt
```

## 6. Cara Menjalankan Proyek

1. Buka notebook menggunakan Google Colab.
2. Pastikan file `train.csv` dan `validation.csv` tersedia pada direktori `/content/data/`.
3. Jalankan seluruh cell notebook secara berurutan.
4. Model akan dilatih dan dievaluasi menggunakan data validation.
5. Model hasil training disimpan dalam format `.joblib` agar dapat digunakan kembali tanpa perlu melakukan training ulang.

## 7. Kesimpulan

Proyek ini berhasil mengembangkan model klasifikasi sentimen ulasan Amazon Video Games menggunakan TF-IDF dan Logistic Regression.

Dengan hasil Macro-F1 sebesar 89,01%, model menunjukkan performa yang baik dalam membedakan ulasan positif dan negatif pada subset data validation yang digunakan.

Pengembangan selanjutnya dapat dilakukan melalui hyperparameter tuning, eksplorasi model alternatif, serta penggunaan sampel dataset yang lebih luas untuk menguji kemampuan generalisasi model.
