import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# baca file csv
df = pd.read_csv("data/sms_spam_indo.csv")

print(df.head())

# samakan nama kolom
df.columns = ['label', 'text']

# bersihin teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# pisah data x dan y
X = df['text']
y = df['label']

# bagi data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ubah teks jadi angka
tfidf = TfidfVectorizer(max_features=3000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# training model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# prediksi data test
y_pred = model.predict(X_test_tfidf)

# cek hasil
acc = accuracy_score(y_test, y_pred)

print("Akurasi:", acc)
print(classification_report(y_test, y_pred))

# simpan model
joblib.dump(model, "models/model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")

print("model berhasil disimpan")