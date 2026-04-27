# SMS SPAM PROJECT

Project ini saya buat untuk belajar tentang text mining dan klasifikasi teks menggunakan Python. Program ini digunakan untuk mendeteksi apakah sebuah pesan termasuk spam atau pesan normal.

## Tujuan Project

- Belajar mengolah data teks
- Memahami proses klasifikasi pesan
- Menerapkan TF-IDF pada data teks
- Mencoba algoritma Naive Bayes
- Membuat aplikasi sederhana berbasis Streamlit

## Dataset

Dataset yang digunakan adalah:

`sms_spam_indo.csv`

Berisi data pesan dengan kategori:

- spam
- ham

## Metode

- TF-IDF
- Multinomial Naive Bayes

## Hasil

Model berhasil dijalankan dengan akurasi sebesar **96.5%** dalam mendeteksi pesan spam dan pesan normal.

## Struktur Folder

```bash
SMS SPAM PROJECT/
│── app/
│   └── app.py
│── data/
│   └── sms_spam_indo.csv
│── models/
│   ├── model.pkl
│   └── tfidf.pkl
│── src/
│   ├── train.py
│   └── predict.py
│── README.md
│── requirements.txt