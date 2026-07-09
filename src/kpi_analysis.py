import pandas as pd

# Load dataset
news = pd.read_csv("data/preprocessed_news.csv")

# Remove missing values
news = news.dropna(subset=["clean_text"])

# Total Articles
total_articles = len(news)

# Fake & Real Count
fake_count = (news["label"] == "Fake").sum()
real_count = (news["label"] == "Real").sum()

# Percentages
fake_percentage = (fake_count / total_articles) * 100
real_percentage = (real_count / total_articles) * 100

# Subjects
subjects = news["subject"].nunique()

# Average article length
news["article_length"] = news["text"].str.len()

average_length = news["article_length"].mean()

print("="*60)
print("PROJECT KPIs")
print("="*60)

print(f"Total Articles       : {total_articles}")
print(f"Fake Articles        : {fake_count}")
print(f"Real Articles        : {real_count}")

print(f"Fake Percentage      : {fake_percentage:.2f}%")
print(f"Real Percentage      : {real_percentage:.2f}%")

print(f"Unique Subjects      : {subjects}")

print(f"Average Article Size : {average_length:.2f} characters")