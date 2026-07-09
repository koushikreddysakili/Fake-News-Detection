import pandas as pd
import matplotlib.pyplot as plt

fake = pd.read_csv("data/Fake.csv.zip")
true = pd.read_csv("data/True.csv.zip")

fake["label"] = "Fake"
true["label"] = "Real"

news = pd.concat([fake, true], ignore_index=True)

counts = news["label"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(counts.index, counts.values)

plt.title("Fake vs Real News")

plt.xlabel("News Type")

plt.ylabel("Number of Articles")

plt.show()