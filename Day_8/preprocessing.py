import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# current working directory is — important for cloud deployment).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_student_performance.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def encode_data(df):
    le = LabelEncoder()
    df["Degree_Program"] = le.fit_transform(df["Degree_Program"])

    print("Degree Program (Encoded):")
    print(df["Degree_Program"].head())

    ohe = OneHotEncoder(sparse_output=False)
    performance_encoded = ohe.fit_transform(df[["Performance"]])

    performance_df = pd.DataFrame(
        performance_encoded,
        columns=ohe.get_feature_names_out(["Performance"])
    )

    print("\nPerformance (One-Hot Encoded):")
    print(performance_df.head())

    return df, le


def split_and_scale(df):
    X = df[
        [
            "Python_Score",
            "ML_Score",
            "Attendance_Percentage",
            "Degree_Program",
            "Age",
        ]
    ]

    y = df["Average_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTraining Set Shape:", X_train.shape)
    print("Testing Set Shape:", X_test.shape)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining Set After Scaling:")
    print(X_train[:5])

    print("\nTesting Set After Scaling:")
    print(X_test[:5])

    return X_train, X_test, y_train, y_test, scaler


def preprocess_pipeline():
    df = load_data()
    df, encoder = encode_data(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    return {
        "df": df,
        "encoder": encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def main():
    preprocess_pipeline()


if __name__ == "__main__":
    main()