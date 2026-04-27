import joblib
import re

# load model dan tfidf
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# bersihin teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# input dari user
pesan = input("Masukkan pesan: ")

# preprocess
pesan_bersih = clean_text(pesan)

# ubah ke tfidf
data = tfidf.transform([pesan_bersih])

# prediksi
hasil = model.predict(data)

print("Hasil:", hasil[0])