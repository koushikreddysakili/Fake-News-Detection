import sqlite3
import pandas as pd

# Load dataset
news = pd.read_csv("data/news_dataset.csv")

# Create SQLite database
conn = sqlite3.connect("data/news.db")

# Store dataset into SQL table
news.to_sql("news", conn, if_exists="replace", index=False)

print("Database created successfully!")

# Query 1
query1 = """
SELECT label,
COUNT(*) AS Total_Articles
FROM news
GROUP BY label;
"""

print("\nFake vs Real News")
print(pd.read_sql(query1, conn))

# Query 2
query2 = """
SELECT subject,
COUNT(*) AS Total
FROM news
GROUP BY subject
ORDER BY Total DESC;
"""

print("\nArticles by Subject")
print(pd.read_sql(query2, conn))

# Query 3
query3 = """
SELECT
AVG(LENGTH(text)) AS Average_Length
FROM news;
"""

print("\nAverage Article Length")
print(pd.read_sql(query3, conn))

conn.close()