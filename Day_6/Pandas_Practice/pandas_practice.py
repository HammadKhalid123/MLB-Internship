import pandas as pd

# Load the dataset.
df = pd.read_csv("titanic.csv")

print("===== First 5 Rows =====")
print(df.head())

print("\n===== Last 5 Rows =====")
print(df.tail())

print("\n===== Dataset Information =====")
df.info()

print("\n===== Missing Values =====")
missing_values = df.isnull().sum()
print(missing_values)

print("\n===== Passengers with Age > 30 =====")
age_filtered = df[df["Age"] > 30]
print(age_filtered)

print("\n===== Female Passengers =====")
female_passengers = df[df["Sex"] == "female"]
print(female_passengers)


print("\n===== Summary Statistics =====")
summary_statistics = df.describe()
print(summary_statistics)