import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Metin için daha uygun
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 0) NLTK verilerini indir ===
# nltk.download("stopwords")
# nltk.download("punkt")        # klasik tokenizer
# nltk.download("punkt_tab")    # yeni sürümlerde gerekiyor
# nltk.download("wordnet")      # lemmatizer
# nltk.download("omw-1.4")      # wordnet yardımcı paket

# === 1) Veri ===
data = pd.read_csv(r"excelller/gender_classifier.csv", encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)
data["gender"] = data["gender"].apply(lambda g: 1 if g == "female" else 0)

# === 2) Ön işleme fonksiyonu ===
lemma = nltk.WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = re.sub("[^a-zA-Z]", " ", str(text))   # sadece harfler
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemma.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Tüm açıklamaları işle
description_list = [preprocess(t) for t in data.description]

# === 3) Bag of Words ===
max_features = 5000
vectorizer = CountVectorizer(max_features=max_features)
X = vectorizer.fit_transform(description_list).toarray()
y = data["gender"].values

# (İstersen en sık kelimeleri görebilirsin)
# print(vectorizer.get_feature_names_out()[:50])

# === 4) Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 5) Model ===
nb = MultinomialNB()
nb.fit(X_train, y_train)

# === 6) Değerlendirme ===
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"accuracy: {acc:.4f}")
print("confusion matrix:\n", cm)
print("\nclassification report:\n", classification_report(y_test, y_pred, target_names=["male(0)", "female(1)"]))
