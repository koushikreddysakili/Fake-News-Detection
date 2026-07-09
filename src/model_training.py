import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

news = pd.read_csv("data/preprocessed_news.csv")

news = news.dropna(subset=["clean_text"])

# Remove empty text
news = news[news["clean_text"].str.strip() != ""]

# Features and Labels
X = news["clean_text"]
y = news["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n✅ Model Training Completed")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# Save Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "fake_news_model.pkl"))

print("\n✅ Model Saved Successfully")

import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "fake_news_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("✅ Model and Vectorizer saved successfully!")