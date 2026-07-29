from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train_decision_tree_model():
    # Loading Dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    print("Features Names:", df.columns.tolist())
    print("Target Names:", iris.target_names)

    # Exploring Dataset
    print("\nPehle 5 rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nTypes of data in each column:")
    print(df.dtypes)

    print("\nMissing values:", df.isnull().sum().sum())

    # Feature Selection
    mi_scores = mutual_info_classif(df.drop("target", axis=1), df["target"])

    feature_importance = pd.DataFrame({
        "feature": iris.feature_names,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False)

    print("\nFeatures sorted by Mutual Information:")
    print(feature_importance)

    # Correlation Analysis
    correlation_with_target = df.corr()["target"].sort_values(ascending=False)

    print("\nCorrelation with target variable:")
    print(correlation_with_target)

    # Splitting Dataset
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    # Training Model
    model = DecisionTreeClassifier(random_state=42)

    print("\nModel training...")
    model.fit(X_train, y_train)

    # Making Predictions
    predictions = model.predict(X_test)

    # Evaluating Model
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    # Visualizing Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig = plt.gcf()

    return {
        "df": df,
        "iris": iris,
        "feature_importance": feature_importance,
        "correlation_with_target": correlation_with_target,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "cm": cm,
        "fig": fig,
    }

if __name__ == "__main__":
    train_decision_tree_model()
    plt.show()