import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Load dataset
news = pd.read_csv("data/preprocessed_news.csv")

# Remove missing values
news = news.dropna(subset=["clean_text"])
news = news[news["clean_text"].str.strip() != ""]

# Features and Labels
X = news["clean_text"]
y = news["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Linear SVM": LinearSVC(random_state=42)
}

print("="*70)
print("MODEL COMPARISON")
print("="*70)

results = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    end = time.time()

    results.append([name, accuracy, end-start])

# Display results
print("\n{:<25} {:<15} {:<15}".format("Model", "Accuracy", "Time (sec)"))
print("-"*60)

for result in results:
    print("{:<25} {:<15.4f} {:<15.2f}".format(result[0], result[1], result[2]))