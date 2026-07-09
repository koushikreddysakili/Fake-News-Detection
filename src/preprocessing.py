import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords(only first time)
nltk.download('stopwords')

# Load cleaned dataset
news = pd.read_csv("data/clean_news.csv")

# Initialize
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    # Remove stop words and stem
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
news["clean_text"] = news["text"].apply(clean_text)

print(news[["text", "clean_text"]].head())

news.to_csv("data/preprocessed_news.csv", index=False)

print("\n✅ Preprocessed dataset saved!")