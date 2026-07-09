import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

news = pd.read_csv("data/preprocessed_news.csv")
print("Missing values in clean_text:", news["clean_text"].isnull().sum())

# Remove rows where clean_text is missing
news = news.dropna(subset=["clean_text"])

# Remove empty strings
news = news[news["clean_text"].str.strip() != ""]

# Features and labels
X = news["clean_text"]
y = news["label"]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Convert text into numerical features
X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF Completed Successfully!")
print("Feature Matrix Shape:", X_tfidf.shape)
print("Labels:", len(y))

import os

# Get the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create models folder if it doesn't exist
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save vectorizer
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, MODEL_PATH)

print(f"\n✅ Vectorizer saved successfully!")
print(f"Location: {MODEL_PATH}")

print("\nVectorizer saved successfully!")