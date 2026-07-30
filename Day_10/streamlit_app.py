import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset_exploration import get_dataset
from baseline_model import run_baseline_model
from hyperparameter_tuning import run_hyperparameter_tuning

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")


@st.cache_data
def cached_dataset():
    return get_dataset()


@st.cache_resource
def cached_baseline():
    return run_baseline_model()


@st.cache_resource
def cached_tuning():
    return run_hyperparameter_tuning()


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)


df, cancer = cached_dataset()

st.title("Breast Cancer Classification Pipeline")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Dataset Exploration",
        "Baseline Model",
        "Hyperparameter Tuning",
        "Prediction Pipeline",
        "Comparison",
    ]
)

with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", cancer.data.shape[0])
    col2.metric("Features", cancer.data.shape[1])
    col3.metric("Classes", len(cancer.target_names))

    st.write("Target classes:", ", ".join(cancer.target_names))

    st.write("First 5 rows")
    st.dataframe(df.head())

    left, right = st.columns(2)
    with left:
        st.write("Target distribution")
        st.bar_chart(df["target"].value_counts())
    with right:
        st.write("Missing values")
        st.dataframe(df.isnull().sum().rename("Missing Values"))

    st.write("Descriptive statistics")
    st.dataframe(df.describe())

with tab2:
    st.subheader("Baseline Logistic Regression Model")

    baseline_results = cached_baseline()

    st.metric("Accuracy", f"{baseline_results['accuracy']:.4f}")

    left, right = st.columns(2)
    with left:
        st.write("Classification Report")
        st.text(baseline_results["report"])
    with right:
        st.write("Confusion Matrix")
        plot_confusion_matrix(
            baseline_results["confusion_matrix"], "Baseline Model - Confusion Matrix"
        )

with tab3:
    st.subheader("Hyperparameter Tuning (GridSearchCV)")

    if st.button("Run Hyperparameter Tuning"):
        with st.spinner("Training..."):
            st.session_state["tuning_results"] = cached_tuning()

    if "tuning_results" in st.session_state:
        tuning_results = st.session_state["tuning_results"]

        st.write("Best Parameters")
        st.json(tuning_results["best_params"])

        col1, col2 = st.columns(2)
        col1.metric("Best CV Score", f"{tuning_results['best_score']:.4f}")
        col2.metric("Test Accuracy", f"{tuning_results['accuracy']:.4f}")

        left, right = st.columns(2)
        with left:
            st.write("Classification Report")
            st.text(tuning_results["report"])
        with right:
            st.write("Confusion Matrix")
            plot_confusion_matrix(
                tuning_results["confusion_matrix"], "Tuned Model - Confusion Matrix"
            )
    else:
        st.info("Click the button above to run GridSearchCV.")

with tab4:
    st.subheader("Final Prediction Pipeline")

    if "tuning_results" in st.session_state:
        tuning_results = st.session_state["tuning_results"]
        X_test = tuning_results["X_test"]
        y_test = tuning_results["y_test"]
        grid_search = tuning_results["grid_search"]

        index = st.slider("Select a test sample index", 0, len(X_test) - 1, 0)

        sample_scaled = X_test[index].reshape(1, -1)
        actual_label = cancer.target_names[y_test.iloc[index]]

        st.write("Selected Sample (scaled features)")
        st.dataframe(pd.DataFrame(sample_scaled, columns=cancer.feature_names))

        prediction = grid_search.predict(sample_scaled)[0]
        predicted_label = cancer.target_names[prediction]

        col1, col2 = st.columns(2)
        col1.metric("Actual Label", actual_label)
        col2.metric("Predicted Label", predicted_label)

        if predicted_label == actual_label:
            st.success("Prediction matches the actual label.")
        else:
            st.error("Prediction does not match the actual label.")
    else:
        st.warning("Run Hyperparameter Tuning first to use the prediction pipeline.")

with tab5:
    st.subheader("Baseline vs Tuned Model Comparison")

    if "tuning_results" in st.session_state:
        baseline_results = cached_baseline()
        tuning_results = st.session_state["tuning_results"]

        comparison_df = pd.DataFrame(
            {
                "Model": ["Baseline", "Tuned"],
                "Accuracy": [baseline_results["accuracy"], tuning_results["accuracy"]],
            }
        )

        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("Model"))

        improvement = tuning_results["accuracy"] - baseline_results["accuracy"]
        st.metric("Improvement", f"{improvement:.4f}")
    else:
        st.warning("Run Hyperparameter Tuning first in the previous tab to see comparison.")