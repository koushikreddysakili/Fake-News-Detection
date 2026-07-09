import streamlit as st
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 Fake News Detection using Machine Learning")

st.write(
    "Paste any news article below and the trained Machine Learning model "
    "will predict whether it is **Fake** or **Real**."
)

news = st.text_area("Enter News Article")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:

        vector = vectorizer.transform([news])

        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vector).max() * 100
            st.info(f"Confidence: {confidence:.2f}%")

        if prediction == "Fake":
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

st.markdown("---")

st.subheader("Project Information")

st.markdown("""
**Technologies Used**

- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Scikit-learn

This project was developed as an internship project for Fake News Detection.
""")