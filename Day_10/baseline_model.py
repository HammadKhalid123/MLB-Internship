import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset_exploration import get_dataset


def run_baseline_model():
    df, cancer = get_dataset()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    results = run_baseline_model()

    print("Training Set Size:", results["X_test"].shape)

    print("\nModel Training...")

    print("\nModel Accuracy:", results["accuracy"])

    print("\nClassification Report:")
    print(results["report"])

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title("Baseline Model - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()