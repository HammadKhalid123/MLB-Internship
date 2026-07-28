import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing import preprocess_pipeline
from linear_regression import train_model, evaluate_model

# Must match the exact columns used inside preprocessing.split_and_scale()
FEATURE_COLUMNS = [
    "Python_Score",
    "ML_Score",
    "Attendance_Percentage",
    "Degree_Program",
    "Age",
]

DATA_PATH = "./data/cleaned_student_performance.csv"

st.set_page_config(
    page_title="Student Score Prediction System",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { color: #1f2937; }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px 8px;
        }
        .stDataFrame { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Score Prediction System")
st.caption("Predicting a student's **Average Score** using Linear Regression")

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset not found at `{DATA_PATH}`. Please add the CSV file there.")
    st.stop()

data = preprocess_pipeline()

df = data["df"]
encoder = data["encoder"]
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
scaler = data["scaler"]

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"Total rows: {len(df)} | Total columns: {len(df.columns)}")

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
metrics = evaluate_model(y_test, y_pred)

st.subheader("Model Evaluation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{metrics['MAE']:.2f}")
c2.metric("MSE", f"{metrics['MSE']:.2f}")
c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
c4.metric("R² Score", f"{metrics['R2']:.4f}")

st.subheader("Actual vs Predicted")

comparison = pd.DataFrame(
    {
        "Actual Score": y_test.values,
        "Predicted Score": np.round(y_pred, 2),
        "Difference": np.round(y_test.values - y_pred, 2),
    }
)
st.dataframe(comparison, use_container_width=True)

# ---------------------------------------------------------
# Graphs — Scatter Plot & Residuals side-by-side (small, tidy)
# ---------------------------------------------------------
st.subheader("Visualizations")

g1, g2 = st.columns(2)

with g1:
    st.markdown("**Actual vs Predicted**")
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
    ax1.scatter(y_test, y_pred, color="#2563eb", alpha=0.7, edgecolors="white", s=35)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        color="#ef4444",
        linewidth=1.5,
    )
    ax1.set_xlabel("Actual Score", fontsize=9)
    ax1.set_ylabel("Predicted Score", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=False)

    os.makedirs("./graphs", exist_ok=True)
    fig1.savefig("./graphs/actual_vs_predicted.png", dpi=300, bbox_inches="tight")

with g2:
    st.markdown("**Residuals Plot** (errors vs predicted)")
    residuals = y_test.values - y_pred
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
    ax2.scatter(y_pred, residuals, color="#f59e0b", alpha=0.7, edgecolors="white", s=35)
    ax2.axhline(0, linestyle="--", color="#ef4444", linewidth=1.5)
    ax2.set_xlabel("Predicted Score", fontsize=9)
    ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=False)

    fig2.savefig("./graphs/residuals_plot.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------
# Feature Coefficients (own row, compact size)
# ---------------------------------------------------------
st.markdown("**Feature Coefficients**")

coef_df = pd.DataFrame(
    {"Feature": FEATURE_COLUMNS, "Coefficient": model.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.barh(coef_df["Feature"], coef_df["Coefficient"], color="#10b981")
ax3.set_xlabel("Coefficient Value", fontsize=9)
ax3.tick_params(labelsize=8)
ax3.grid(alpha=0.2, axis="x")
fig3.tight_layout()
st.pyplot(fig3, use_container_width=False)

# ---------------------------------------------------------
# Try It Yourself - Live Prediction
# ---------------------------------------------------------
st.subheader("Try It Yourself")
st.write("Enter a student's details below to predict their Average Score.")

degree_options = list(encoder.classes_)

col1, col2, col3 = st.columns(3)
with col1:
    python_score = st.slider("Python Score", 0, 100, 70)
    ml_score = st.slider("ML Score", 0, 100, 70)
with col2:
    attendance = st.slider("Attendance %", 0, 100, 80)
    age = st.number_input("Age", min_value=15, max_value=60, value=20)
with col3:
    degree = st.selectbox("Degree Program", degree_options)

if st.button("Predict Average Score"):
    degree_encoded = encoder.transform([degree])[0]
    input_data = np.array([[python_score, ml_score, attendance, degree_encoded, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Average Score: **{prediction:.2f}**")