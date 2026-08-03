import io

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from fashion_mnist_project import train_ann_model

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.6rem;
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 0.4rem;
        }
        div[data-testid="stMetric"] {
            background-color: #f8f9fb;
            border: 1px solid #eaecef;
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
defaults = {
    "model": None,
    "history": None,
    "X_train": None,
    "y_train": None,
    "X_test": None,
    "y_test": None,
    "test_loss": None,
    "test_acc": None,
    "trained": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Model Configuration")

epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=5)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
hidden_units = st.sidebar.select_slider("Hidden Layer Neurons", options=[32, 64, 128, 256, 512], value=128)
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh", "sigmoid"], index=0)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)

st.sidebar.markdown("---")
train_clicked = st.sidebar.button("Train Model", use_container_width=True, type="primary")
reset_clicked = st.sidebar.button("Reset", use_container_width=True)

if reset_clicked:
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Adjust the settings above, then click **Train Model** to build and evaluate "
    "the network. Use **Reset** to clear results and start over."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="main-title">Fashion MNIST Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train and evaluate a simple Artificial Neural Network '
    'on the Fashion MNIST dataset — configure, train, and analyze results all in one place.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Training trigger
# ----------------------------------------------------------------------------
if train_clicked:
    with st.spinner("Training the model, please wait..."):
        (model, history, X_train, y_train, X_test, y_test,
         test_loss, test_acc) = train_ann_model(
            epochs=epochs,
            batch_size=batch_size,
            hidden_units=hidden_units,
            activation=activation,
            optimizer=optimizer,
        )

        st.session_state.model = model
        st.session_state.history = history
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.test_loss = test_loss
        st.session_state.test_acc = test_acc
        st.session_state.trained = True

    st.success("Training complete.")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_dataset, tab_summary, tab_performance, tab_predictions = st.tabs(
    ["Dataset", "Model Summary", "Performance", "Predictions"]
)

# ---- Dataset tab ----
with tab_dataset:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.write(
        "Fashion MNIST contains 70,000 grayscale images (28x28 pixels) across "
        "10 clothing categories. Below is a quick preview of sample images."
    )

    if st.button("Load Sample Images"):
        from keras.datasets import fashion_mnist
        (X_sample, y_sample), (_, _) = fashion_mnist.load_data()

        st.info(f"Training set shape: {X_sample.shape}  |  Labels shape: {y_sample.shape}")

        cols = st.columns(8)
        for i in range(8):
            with cols[i]:
                st.image(X_sample[i], caption=CLASS_NAMES[y_sample[i]], use_container_width=True)

# ---- Model Summary tab ----
with tab_summary:
    st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
    if st.session_state.trained:
        summary_buffer = io.StringIO()
        st.session_state.model.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))
        st.code(summary_buffer.getvalue(), language="text")

        total_params = st.session_state.model.count_params()
        st.caption(f"Total trainable parameters: {total_params:,}")
    else:
        st.warning("Train a model first to view its architecture summary.")

# ---- Performance tab ----
with tab_performance:
    if st.session_state.trained:
        st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{st.session_state.test_acc * 100:.2f}%")
        col2.metric("Test Loss", f"{st.session_state.test_loss:.4f}")

        st.markdown('<div class="section-title">Accuracy Curve</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 2.8))
        ax1.plot(st.session_state.history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(st.session_state.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=9)
        ax1.set_ylabel('Accuracy', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        fig1.tight_layout()

        st.markdown('<div class="section-title">Loss Curve</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        ax2.plot(st.session_state.history.history['loss'], label='Training', linewidth=2, color='tomato')
        ax2.plot(st.session_state.history.history['val_loss'], label='Validation', linewidth=2, color='goldenrod')
        ax2.set_xlabel('Epoch', fontsize=9)
        ax2.set_ylabel('Loss', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.pyplot(fig1, use_container_width=False)
        with graph_col2:
            st.pyplot(fig2, use_container_width=False)

        st.markdown('<div class="section-title">Per-Class Accuracy</div>', unsafe_allow_html=True)
        y_pred_probs = st.session_state.model.predict(st.session_state.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = st.session_state.y_test

        per_class_acc = []
        for class_idx in range(10):
            mask = y_true == class_idx
            acc = (y_pred[mask] == y_true[mask]).mean() if mask.sum() > 0 else 0
            per_class_acc.append(acc)

        fig3, ax3 = plt.subplots(figsize=(6, 2.8))
        bars = ax3.bar(CLASS_NAMES, per_class_acc, color="#4C72B0")
        ax3.set_ylabel("Accuracy", fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.tick_params(axis='y', labelsize=8)
        ax3.grid(alpha=0.3, axis='y')
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=False)

    else:
        st.warning("Train a model first to view performance metrics.")

# ---- Predictions tab ----
with tab_predictions:
    st.markdown('<div class="section-title">Sample Predictions</div>', unsafe_allow_html=True)
    if st.session_state.trained:
        num_samples = st.slider("Number of samples to display", min_value=4, max_value=16, value=8)
        filter_option = st.radio(
            "Filter results",
            ["All", "Correct only", "Incorrect only"],
            horizontal=True,
        )

        if st.button("Generate Predictions"):
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            model = st.session_state.model

            preds = model.predict(X_test, verbose=0)
            predicted_labels = np.argmax(preds, axis=1)
            correct_mask = predicted_labels == y_test

            if filter_option == "Correct only":
                candidate_indices = np.where(correct_mask)[0]
            elif filter_option == "Incorrect only":
                candidate_indices = np.where(~correct_mask)[0]
            else:
                candidate_indices = np.arange(len(X_test))

            if len(candidate_indices) == 0:
                st.info("No samples match this filter.")
            else:
                chosen = np.random.choice(
                    candidate_indices, size=min(num_samples, len(candidate_indices)), replace=False
                )

                cols = st.columns(4)
                for i, idx in enumerate(chosen):
                    with cols[i % 4]:
                        true_label = CLASS_NAMES[y_test[idx]]
                        pred_label = CLASS_NAMES[predicted_labels[idx]]
                        confidence = preds[idx][predicted_labels[idx]] * 100
                        st.image(X_test[idx], use_container_width=True)
                        caption = f"True: {true_label}\nPredicted: {pred_label} ({confidence:.1f}%)"
                        if true_label == pred_label:
                            st.success(caption)
                        else:
                            st.error(caption)
    else:
        st.warning("Train a model first to generate predictions.")