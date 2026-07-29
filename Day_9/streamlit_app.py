import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from iris_classification import train_logistic_model
from decision_tree_comparison import train_decision_tree_model

st.set_page_config(
    page_title="Iris Flower Classification Dashboard",
    page_icon="🌸",
    layout="wide",
)

@st.cache_resource
def get_results():
    log_results = train_logistic_model()
    tree_results = train_decision_tree_model()
    return log_results, tree_results

log_results, tree_results = get_results()

df = log_results["df"]
iris = log_results["iris"]

st.sidebar.title("🌸 Iris Dashboard")
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Data Exploration", "Model Comparison", "Try Your Own Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Mini Project — Iris Flower Classification System")

if page == "Overview":
    st.title("Iris Flower Classification System")
    st.write(
        "This dashboard reuses the results from iris_classification.py and "
        "decision_tree_comparison.py to compare a Logistic Regression model "
        "with a Decision Tree model on the Iris dataset."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(iris.feature_names))
    col3.metric("Classes", len(iris.target_names))
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Sample of the Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Target Class Distribution")
    dist = df["target"].value_counts()
    dist.index = [iris.target_names[i] for i in dist.index]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(dist.index, dist.values)
    ax.set_ylabel("Count")
    ax.set_xlabel("Species")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(fig)

elif page == "Data Exploration":
    st.title("Data Exploration")

    st.markdown("### Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### Feature Importance (Mutual Information)")
    st.dataframe(log_results["feature_importance"], use_container_width=True)

    st.markdown("### Correlation with Target")
    st.dataframe(
        log_results["correlation_with_target"].to_frame("correlation"),
        use_container_width=True,
    )

elif page == "Model Comparison":
    st.title("Model Comparison")

    log_acc = log_results["accuracy"]
    tree_acc = tree_results["accuracy"]

    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression Accuracy", f"{log_acc*100:.2f}%")
    col2.metric("Decision Tree Accuracy", f"{tree_acc*100:.2f}%")

    tab1, tab2 = st.tabs(["Logistic Regression", "Decision Tree"])

    with tab1:
        st.markdown("#### Confusion Matrix")
        log_fig = log_results["fig"]
        log_fig.set_size_inches(4, 3)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(log_fig)

    with tab2:
        st.markdown("#### Confusion Matrix")
        tree_fig = tree_results["fig"]
        tree_fig.set_size_inches(4, 3)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(tree_fig)

    st.markdown("### Sample Predictions vs Actual Values")
    y_test = tree_results["y_test"]
    sample_df = tree_results["X_test"].copy()
    sample_df["actual"] = y_test.map(lambda x: iris.target_names[x]).values
    sample_df["logistic_pred"] = pd.Series(
        log_results["predictions"], index=y_test.index
    ).map(lambda x: iris.target_names[x])
    sample_df["tree_pred"] = pd.Series(
        tree_results["predictions"], index=y_test.index
    ).map(lambda x: iris.target_names[x])
    st.dataframe(sample_df.head(15), use_container_width=True)

elif page == "Try Your Own Prediction":
    st.title("Try Your Own Prediction")
    st.write("Values select karo aur model choose karke predict button dabao.")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            float(df["sepal length (cm)"].min()),
            float(df["sepal length (cm)"].max()),
            float(df["sepal length (cm)"].mean()),
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            float(df["sepal width (cm)"].min()),
            float(df["sepal width (cm)"].max()),
            float(df["sepal width (cm)"].mean()),
        )

    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            float(df["petal length (cm)"].min()),
            float(df["petal length (cm)"].max()),
            float(df["petal length (cm)"].mean()),
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            float(df["petal width (cm)"].min()),
            float(df["petal width (cm)"].max()),
            float(df["petal width (cm)"].mean()),
        )

    model_choice = st.radio(
        "Model choose karo:",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True,
    )

    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names,
    )

    if st.button("Predict Species"):
        if model_choice == "Logistic Regression":
            scaled_input = log_results["scaler"].transform(input_data)
            pred = log_results["model"].predict(scaled_input)[0]
            proba = log_results["model"].predict_proba(scaled_input)[0]
        else:
            pred = tree_results["model"].predict(input_data)[0]
            proba = tree_results["model"].predict_proba(input_data)[0]

        st.success(f"Predicted Species: **{iris.target_names[pred]}**")

        proba_df = pd.DataFrame({
            "species": iris.target_names,
            "probability": proba
        }).set_index("species")
        st.markdown("#### Prediction Confidence")
        st.bar_chart(proba_df)