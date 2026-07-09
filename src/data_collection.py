import pandas as pd

# -----------------------------
# Source 1: Fake News Dataset
# -----------------------------
fake = pd.read_csv("data/Fake.csv.zip")

# -----------------------------
# Source 2: Real News Dataset
# -----------------------------
true = pd.read_csv("data/True.csv.zip")

# Add labels
fake["label"] = "Fake"
true["label"] = "Real"

# Merge datasets
news = pd.concat([fake, true], ignore_index=True)

print("=" * 60)
print("DATA COLLECTION REPORT")
print("=" * 60)

print(f"Fake Dataset Records : {len(fake)}")
print(f"Real Dataset Records : {len(true)}")
print(f"Total Records        : {len(news)}")

print("\nColumns Available:")
print(news.columns.tolist())

print("\nFirst Five Records")
print(news.head())

# Save combined dataset
news.to_csv("data/news_dataset.csv", index=False)

print("\nCombined dataset saved successfully!")