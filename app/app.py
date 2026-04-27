import streamlit as st
import joblib
import re

# load model
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# bersihin teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# judul
st.title("SpamShield ID")
st.subheader("Deteksi Spam SMS Bahasa Indonesia")

# input user
pesan = st.text_area("Masukkan pesan")

if st.button("Cek Pesan"):
    teks = clean_text(pesan)
    data = tfidf.transform([teks])
    hasil = model.predict(data)[0]

    if hasil == "spam":
        st.error("Pesan termasuk SPAM")
    else:
        st.success("Pesan termasuk HAM / Normal")