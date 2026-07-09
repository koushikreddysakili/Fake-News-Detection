import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
news = pd.read_csv("data/preprocessed_news.csv")

# Remove missing values
news = news.dropna(subset=["clean_text"])

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Chart 1: Fake vs Real Count
# -----------------------------

counts = news["label"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(counts.index, counts.values)
plt.title("Fake vs Real News")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")

plt.savefig("outputs/fake_vs_real.png")
plt.show()

# -----------------------------
# Chart 2: Pie Chart
# -----------------------------

plt.figure(figsize=(6,6))

plt.pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("News Distribution")

plt.savefig("outputs/news_pie.png")

plt.show()

# -----------------------------
# Chart 3: Subject Distribution
# -----------------------------

subject_counts = news["subject"].value_counts()

plt.figure(figsize=(10,5))

subject_counts.plot(kind="bar")

plt.title("Subject Distribution")

plt.ylabel("Articles")

plt.savefig("outputs/subjects.png")

plt.show()

# -----------------------------
# Chart 4: Article Length
# -----------------------------

news["article_length"] = news["text"].str.len()

plt.figure(figsize=(8,5))

plt.hist(news["article_length"], bins=40)

plt.title("Article Length Distribution")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.savefig("outputs/article_length.png")

plt.show()

# -----------------------------
# Word Cloud - Fake
# -----------------------------

fake_text = " ".join(news[news["label"]=="Fake"]["clean_text"])

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(fake_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Fake News Word Cloud")

plt.savefig("outputs/fake_wordcloud.png")

plt.show()

# -----------------------------
# Word Cloud - Real
# -----------------------------

real_text = " ".join(news[news["label"]=="Real"]["clean_text"])

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(real_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Real News Word Cloud")

plt.savefig("outputs/real_wordcloud.png")

plt.show()

print("\nAll visualizations saved successfully!")