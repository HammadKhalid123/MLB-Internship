from sklearn.datasets import load_iris
import pandas as pd

def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names

def explore_dataset(df):
    print("Dataset Information:")
    print(df.info())
    
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nClass Distribution:")
    print(df['target'].value_counts())


def main():
    df, target_names = get_dataset()
    explore_dataset(df)

if __name__ == "__main__":
    main()