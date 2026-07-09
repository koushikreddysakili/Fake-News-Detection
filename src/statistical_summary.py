import pandas as pd

# Load dataset
news = pd.read_csv("data/preprocessed_news.csv")

# Remove missing values
news = news.dropna(subset=["clean_text"])

# Article statistics
news["article_length"] = news["text"].str.len()
news["word_count"] = news["text"].str.split().apply(len)

print("="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print(f"Total Records           : {len(news)}")

print(f"Missing Values          : {news.isnull().sum().sum()}")

print(f"Duplicate Rows          : {news.duplicated().sum()}")

print(f"Average Article Length  : {news['article_length'].mean():.2f}")

print(f"Median Article Length   : {news['article_length'].median():.2f}")

print(f"Minimum Length          : {news['article_length'].min()}")

print(f"Maximum Length          : {news['article_length'].max()}")

print(f"Standard Deviation      : {news['article_length'].std():.2f}")

print(f"Average Words           : {news['word_count'].mean():.2f}")

print("\n")

print("="*70)
print("SUBJECT DISTRIBUTION")
print("="*70)

print(news["subject"].value_counts())

print("\n")

print("="*70)
print("FAKE vs REAL")
print("="*70)

print(news["label"].value_counts(normalize=True)*100)