import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from fashion_mnist_cnn import (
    CLASS_NAMES,
    BEST_MODEL_PATH,
    load_dataset,
    normalize,
    reshape_images,
    build_cnn,
    train_model,
    evaluate_model,
    predict,
    load_best_model,
)

st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="👕", layout="wide")

GRAPHS_DIR = "Graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)


def save_and_show(fig, name):
    fig.savefig(os.path.join(GRAPHS_DIR, f"{name}.png"), bbox_inches="tight", dpi=150)
    st.pyplot(fig)
    plt.close(fig)


st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(90deg, #4C72B0, #55A868);
        padding: 22px 20px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .main-title {
        font-size: 100px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #f0f0f0;
        font-size: 16px;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-box">
        <p class="main-title">👕 Fashion MNIST CNN Classifier</p>
        <p class="sub-title">Explore, Train, Evaluate and Predict Fashion Items using Deep Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    X_train, y_train, X_test, y_test = load_dataset()
    X_train_n, X_test_n = normalize(X_train, X_test)
    X_train_r, X_test_r = reshape_images(X_train_n, X_test_n)
    return X_train, y_train, X_test, y_test, X_train_r, X_test_r


@st.cache_resource
def get_model_if_exists():
    if os.path.exists(BEST_MODEL_PATH):
        return load_best_model()
    return None


X_train, y_train, X_test, y_test, X_train_r, X_test_r = get_data()

if "model" not in st.session_state:
    st.session_state.model = get_model_if_exists()
if "history" not in st.session_state:
    st.session_state.history = None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Explore Data", "🧠 Train Model", "📈 Performance", "🔍 Predictions", "🖼️ Try Your Own Image"]
)

with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", X_train.shape[0])
    col2.metric("Test Samples", X_test.shape[0])
    col3.metric("Classes", len(CLASS_NAMES))

    st.write("")
    st.subheader("Sample Images")

    n_samples = st.slider("Images to display", min_value=5, max_value=20, value=10)
    cols = st.columns(5)
    for i in range(n_samples):
        with cols[i % 5]:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(X_train[i], cmap="gray")
            ax.set_title(CLASS_NAMES[y_train[i]], fontsize=9)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

    st.write("")
    st.subheader("Class Distribution")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        fig, ax = plt.subplots(figsize=(4.5, 2.2))
        unique, counts = np.unique(y_train, return_counts=True)
        ax.bar([CLASS_NAMES[i] for i in unique], counts, color="#4C72B0")
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        save_and_show(fig, "class_distribution")

with tab2:
    st.subheader("Train the CNN Model")
    st.write("Configure the training settings below and start training. The best performing model is saved automatically to the **Graphs** and model files during training.")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
    with col2:
        st.metric("Architecture", "2 Conv + 2 Dense")
    with col3:
        status = "Loaded ✅" if st.session_state.model is not None else "Not Trained ❌"
        st.metric("Current Model Status", status)

    with st.expander("View Model Architecture"):
        summary_model = build_cnn()
        summary_lines = []
        summary_model.summary(print_fn=lambda line: summary_lines.append(line))
        st.text("\n".join(summary_lines))

    st.write("")
    train_button = st.button("🚀 Start Training", use_container_width=True, type="primary")

    if train_button:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.write(f"Preparing model for {epochs} epochs...")

        model = build_cnn()
        progress_bar.progress(20)

        progress_text.write("Training in progress, please wait...")
        history = train_model(model, X_train_r, y_train, X_test_r, y_test, epochs=epochs)
        progress_bar.progress(90)

        st.session_state.model = model
        st.session_state.history = history.history
        progress_bar.progress(100)
        progress_text.empty()

        final_acc = history.history["val_accuracy"][-1] * 100
        st.success(f"Training completed! Final validation accuracy: {final_acc:.2f}%. Best model saved automatically.")

    st.write("")
    if st.session_state.model is not None:
        st.info("A trained model is currently loaded and ready to use in the Performance, Predictions and Try Your Own Image tabs.")
    else:
        st.warning("No model available yet. Train a model above, or make sure a saved best model file exists.")

with tab3:
    st.subheader("Model Performance")

    if st.session_state.model is None:
        st.warning("Please train a model first from the Train Model tab.")
    else:
        model = st.session_state.model
        loss, accuracy = evaluate_model(model, X_test_r, y_test)

        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        col2.metric("Test Loss", f"{loss:.4f}")

        if st.session_state.history is not None:
            h = st.session_state.history
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(3.2, 2.2))
                ax.plot(h["accuracy"], label="Train")
                ax.plot(h["val_accuracy"], label="Validation")
                ax.set_title("Accuracy", fontsize=9)
                ax.set_xlabel("Epoch", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                save_and_show(fig, "accuracy_graph")

            with col2:
                fig, ax = plt.subplots(figsize=(3.2, 2.2))
                ax.plot(h["loss"], label="Train")
                ax.plot(h["val_loss"], label="Validation")
                ax.set_title("Loss", fontsize=9)
                ax.set_xlabel("Epoch", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                save_and_show(fig, "loss_graph")

        predicted_labels = predict(model, X_test_r)

        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predicted_labels)
            fig, ax = plt.subplots(figsize=(4, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
            disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", colorbar=False)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)
            save_and_show(fig, "confusion_matrix")

        with col2:
            st.subheader("Per-Class Accuracy")
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.barh(CLASS_NAMES, per_class_acc, color="#C44E52")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Accuracy", fontsize=8)
            ax.tick_params(labelsize=7)
            save_and_show(fig, "per_class_accuracy")

        st.write("")
        st.subheader("Predicted vs Actual Label Distribution")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        width = 0.4
        x = np.arange(len(CLASS_NAMES))
        actual_counts = np.bincount(y_test, minlength=len(CLASS_NAMES))
        pred_counts = np.bincount(predicted_labels, minlength=len(CLASS_NAMES))
        ax.bar(x - width / 2, actual_counts, width, label="Actual", color="#4C72B0")
        ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=8)
        save_and_show(fig, "predicted_vs_actual_distribution")

with tab4:
    st.subheader("Correct and Incorrect Predictions")

    if st.session_state.model is None:
        st.warning("Please train a model first from the Train Model tab.")
    else:
        model = st.session_state.model
        predicted_labels = predict(model, X_test_r)

        correct_idx = np.where(predicted_labels == y_test)[0]
        incorrect_idx = np.where(predicted_labels != y_test)[0]

        col1, col2 = st.columns(2)
        col1.metric("Correct Predictions", len(correct_idx))
        col2.metric("Incorrect Predictions", len(incorrect_idx))

        st.write("")
        pred_type = st.radio("Show", ["Correct Predictions", "Incorrect Predictions"], horizontal=True)
        indices = correct_idx if pred_type == "Correct Predictions" else incorrect_idx

        n_show = min(10, len(indices))
        cols = st.columns(5)
        for i in range(n_show):
            idx = indices[i]
            with cols[i % 5]:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(X_test[idx], cmap="gray")
                ax.set_title(
                    f"T:{CLASS_NAMES[y_test[idx]]}\nP:{CLASS_NAMES[predicted_labels[idx]]}",
                    fontsize=8,
                )
                ax.axis("off")
                if pred_type == "Incorrect Predictions":
                    save_and_show(fig, f"incorrect_prediction_{i}")
                else:
                    st.pyplot(fig)
                    plt.close(fig)

with tab5:
    st.subheader("Upload Your Own Image")

    if st.session_state.model is None:
        st.warning("Please train a model first from the Train Model tab.")
    else:
        uploaded_file = st.file_uploader("Choose a grayscale clothing image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            img_array = np.array(image).astype("float32") / 255.0
            img_input = img_array.reshape(1, 28, 28, 1)

            model = st.session_state.model
            prediction = model.predict(img_input)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            col1, col2 = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img_array, cmap="gray")
                ax.axis("off")
                save_and_show(fig, "uploaded_image")

            with col2:
                st.success(f"Predicted Class: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}%")
                fig, ax = plt.subplots(figsize=(4.5, 2.5))
                ax.bar(CLASS_NAMES, prediction[0], color="#55A868")
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                ax.set_ylabel("Probability", fontsize=8)
                save_and_show(fig, "uploaded_image_prediction")