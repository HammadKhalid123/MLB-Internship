import pandas as pd
from sklearn.datasets import load_breast_cancer


def get_dataset():
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df["target"] = cancer.target

    return df, cancer


if __name__ == "__main__":
    df, cancer = get_dataset()

    print("Features shape:", cancer.data.shape)
    print("Target shape:", cancer.target.shape)
    print("Feature names:\n", cancer.feature_names)
    print("Target names:", cancer.target_names)

    print("\nPehle 5 rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nTypes of data in each column:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\n Descriptive statistics of the dataset:")
    print(df.describe())

    print("\n Information about the dataset:")
    print(df.info())