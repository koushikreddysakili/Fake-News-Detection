import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv.zip")
true = pd.read_csv("data/True.csv.zip")

# Add labels
fake["label"] = "Fake"
true["label"] = "Real"

# Merge datasets
news = pd.concat([fake, true], ignore_index=True)

print("=" * 50)
print("Before Cleaning")
print("=" * 50)

print("Shape :", news.shape)
print("Duplicates :", news.duplicated().sum())
print("Missing Values:\n")
print(news.isnull().sum())

# Remove duplicate rows
news = news.drop_duplicates()

print("\n" + "=" * 50)
print("After Removing Duplicates")
print("=" * 50)

print("Shape :", news.shape)

# Save cleaned dataset
news.to_csv("data/clean_news.csv", index=False)

print("\n✅ Cleaned dataset saved as data/clean_news.csv")