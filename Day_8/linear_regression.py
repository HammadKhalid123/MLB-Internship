import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_pipeline

GRAPH_PATH = "./graphs/actual_vs_predicted.png"


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def plot_actual_vs_predicted(y_test, y_pred, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(x=y_test, y=y_pred, s=80, ax=ax)

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax.set_title("Actual vs Predicted Average Scores")
    ax.set_xlabel("Actual Average Score")
    ax.set_ylabel("Predicted Average Score")
    ax.legend()
    ax.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def main():
    data = preprocess_pipeline()

    correlation = data["df"].corr(numeric_only=True)

    print("\nCorrelation with Average Score:")
    print(correlation["Average_Score"].sort_values(ascending=False))

    model = train_model(data["X_train"], data["y_train"])

    y_test = data["y_test"]
    y_pred = model.predict(data["X_test"])

    metrics = evaluate_model(y_test, y_pred)

    print("\n" + "=" * 50)
    print("Model Evaluation Metrics")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"R² Score: {metrics['R2']:.4f}")

    comparison_df = pd.DataFrame(
        {
            "Actual Score": y_test.values,
            "Predicted Score": y_pred.round(2),
        }
    )

    print("\n" + "=" * 50)
    print("Actual vs Predicted Scores")
    print("=" * 50)
    print(comparison_df)

    print("\n" + "=" * 50)
    print("Predicted Student Average Scores")
    print("=" * 50)

    for i, prediction in enumerate(y_pred, start=1):
        print(f"Student {i}: {prediction:.2f}")

    fig = plot_actual_vs_predicted(
        y_test,
        y_pred,
        save_path=GRAPH_PATH,
    )

    plt.show()


if __name__ == "__main__":
    main()