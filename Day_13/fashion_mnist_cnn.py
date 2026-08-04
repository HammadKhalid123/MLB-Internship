"""
Fashion MNIST - CNN Classification Project
Complete pipeline: Load -> Explore -> Preprocess -> Build -> Train -> Evaluate -> Predict -> Save Best Model
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

BEST_MODEL_PATH = "best_fashion_mnist_model.h5"


# 1. Load Dataset
def load_dataset():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    return X_train, y_train, X_test, y_test


# 2. Explore Dataset
def explore_dataset(X_train, y_train):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.xlabel(f"Label: {CLASS_NAMES[y_train[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 3. Normalize
def normalize(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test


# 4. Reshape (add channel dimension)
def reshape_images(X_train, X_test):
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    print("X_train shape after reshape:", X_train.shape)
    print("X_test shape after reshape:", X_test.shape)
    return X_train, X_test


# 5. Build CNN
def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


# 6. Train Model (with Best Model Save)
def train_model(model, X_train, y_train, X_test, y_test, epochs=10):
    # ModelCheckpoint saves the BEST model automatically during training
    # "best" = lowest validation loss (aap 'val_accuracy' bhi use kar saktay hain)
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_loss',      # kis metric ko dekh kar best decide karay
        save_best_only=True,     # sirf best wala save hoga, har epoch nahi
        mode='min',              # val_loss ka kam hona acha hai
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    return history


# 7. Evaluate
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy


# 8. Predict
def predict(model, X_test):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels


# 9. Accuracy Graph
def plot_accuracy(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 10. Loss Graph
def plot_loss(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 11. Confusion Matrix
def plot_confusion_matrix(y_test, predicted_labels):
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# 12. Correct Predictions
def show_correct_predictions(X_test, y_test, predicted_labels, num=10):
    correct_idx = np.where(predicted_labels == y_test)[0]
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(correct_idx[:num]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')
        plt.xlabel(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[predicted_labels[idx]]}")
        plt.axis('off')
    plt.suptitle("Correct Predictions")
    plt.tight_layout()
    plt.show()


# 13. Incorrect Predictions
def show_incorrect_predictions(X_test, y_test, predicted_labels, num=10):
    incorrect_idx = np.where(predicted_labels != y_test)[0]
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(incorrect_idx[:num]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')
        plt.xlabel(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[predicted_labels[idx]]}")
        plt.axis('off')
    plt.suptitle("Incorrect Predictions")
    plt.tight_layout()
    plt.show()


# 14. Load the saved BEST model later (example function)
def load_best_model():
    model = load_model(BEST_MODEL_PATH)
    print(f"Best model loaded from: {BEST_MODEL_PATH}")
    return model


# ---------------- MAIN PIPELINE ----------------
def main():
    X_train, y_train, X_test, y_test = load_dataset()
    explore_dataset(X_train, y_train)

    X_train, X_test = normalize(X_train, X_test)
    X_train, X_test = reshape_images(X_train, X_test)

    model = build_cnn()
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=10)

    evaluate_model(model, X_test, y_test)
    predicted_labels = predict(model, X_test)

    plot_accuracy(history)
    plot_loss(history)
    plot_confusion_matrix(y_test, predicted_labels)

    show_correct_predictions(X_test, y_test, predicted_labels)
    show_incorrect_predictions(X_test, y_test, predicted_labels)

    print(f"\nTraining complete. Best model already saved at: {BEST_MODEL_PATH}")

    # Best model dobara load karne ka tareeqa:
    # best_model = load_best_model()


if __name__ == "__main__":
    main()