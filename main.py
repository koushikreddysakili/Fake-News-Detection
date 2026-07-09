import pandas as pd

print("=" * 60)
print("FAKE NEWS DETECTION PROJECT")
print("=" * 60)

fake = pd.read_csv(r"C:\Users\sakil\OneDrive\เอกสาร\fack news detection\data\Fake.csv.zip")
true = pd.read_csv(r"C:\Users\sakil\OneDrive\เอกสาร\fack news detection\data\True.csv.zip")

print("\nDatasets Loaded Successfully!\n")

print("Fake Dataset Shape :", fake.shape)
print("True Dataset Shape :", true.shape)

print("\nFake Dataset Columns")
print(fake.columns)

print("\nTrue Dataset Columns")
print(true.columns)

print("\nFirst Five Rows of Fake Dataset")
print(fake.head())

print("\nFirst Five Rows of True Dataset")
print(true.head())