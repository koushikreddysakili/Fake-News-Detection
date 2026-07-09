import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load model
model = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.pkl"))

# Load vectorizer
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("=" * 60)
print("FAKE NEWS DETECTOR")
print("=" * 60)

while True:

    news = input("\nEnter News Article:\n")

    if news.lower() == "exit":
        print("Goodbye!")
        break

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    print("\nPrediction:", prediction[0])