#Fake News Detection using Machine Learning

## Project Overview

The Fake News Detection System is a Machine Learning-based web application that classifies news articles as **Fake** or **Real**. The project uses **Natural Language Processing (NLP)** techniques to preprocess text data and **TF-IDF** for feature extraction. A **Logistic Regression** model is trained on the processed dataset and deployed using **Streamlit** to provide real-time predictions.

---

## Project Objectives

- Detect whether a news article is Fake or Real.
- Apply Natural Language Processing (NLP) techniques to clean text.
- Convert text into numerical features using TF-IDF.
- Train and evaluate a Machine Learning model.
- Build an interactive web application using Streamlit.
- Deploy the project for public access.

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Plotly
- Joblib
- SQLite
- Git & GitHub

---

## Project Structure

```
Fake-News-Detection/
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── app.py
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── visualization.py
│   ├── sql_analysis.py
│   ├── advanced_eda.py
│   ├── statistical_summary.py
│   └── kpi_analysis.py
│
├── requirements.txt
├── main.py
└── README.md
```

---

## Project Workflow

```
Dataset
    ↓
Data Collection
    ↓
Data Cleaning
    ↓
Text Preprocessing (NLP)
    ↓
Feature Engineering (TF-IDF)
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Prediction
    ↓
Streamlit Dashboard
```

---

## Machine Learning Model

**Algorithm Used:**

- Logistic Regression

**Feature Extraction:**

- TF-IDF Vectorizer

**Natural Language Processing:**

- Lowercase Conversion
- Tokenization
- Stopword Removal
- Punctuation Removal
- Stemming

---

## Features

- Predicts Fake or Real news articles.
- Interactive Streamlit web application.
- Dashboard with KPIs.
- Data visualization using Plotly.
- Statistical summary.
- SQL-based dataset analysis.
- Machine Learning model evaluation.
- Real-time prediction.

---

## Dashboard Features

- Total Articles KPI
- Fake News Count
- Real News Count
- Model Accuracy
- News Distribution Charts
- Subject Distribution
- Statistical Summary
- Business Insights

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/koushikreddysakili/Fake-News-Detection.git
```

### 2. Open Project

```bash
cd Fake-News-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run src/app.py
```

---

## Model Performance

| Metric | Value |
|---------|-------|
| Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Prediction Type | Binary Classification |
| Output | Fake / Real |

---

## Sample Workflow

User enters news article
          ↓
NLP Preprocessing
          ↓
TF-IDF Vectorization
          ↓
Logistic Regression Model
          ↓
Prediction
          ↓
Display Fake or Real

---

## Business Insights

- Political news dominates the dataset.
- NLP preprocessing improves text quality.
- TF-IDF extracts meaningful text features.
- Logistic Regression performs efficiently for binary text classification.
- Interactive dashboard provides quick insights into the dataset.

---

## Future Enhancements

- Deep Learning (LSTM/BERT)
- Multi-language Fake News Detection
- Live News API Integration
- User Authentication
- News Source Credibility Analysis
- Explainable AI (XAI)
- Mobile Application

---


**Sakili Koushik reddy**

B.Tech (CSE - AI & ML)

R.V.R. & J.C. College of Engineering, Guntur
