import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras import layers

# ==========================================================
# Paths
# ==========================================================

BEST_MODEL_PATH = "best_cats_vs_dogs_model.keras"
GRAPH_FOLDER = "Graphs"

os.makedirs(GRAPH_FOLDER, exist_ok=True)

CLASS_NAMES = ["Cat", "Dog"]


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    dataset, info = tfds.load(
        "cats_vs_dogs",
        split="train",
        as_supervised=True,
        with_info=True
    )

    total_images = info.splits["train"].num_examples
    train_size = int(0.8 * total_images)
    train_data = dataset.take(train_size)
    validation_data = dataset.skip(train_size)

    print("Total Images :", total_images)
    print("Train Images :", train_size)
    print("Validation Images :", total_images - train_size)

    return train_data, validation_data


# ==========================================================
# Preprocessing
# ==========================================================

def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    return image, label


def prepare_dataset(dataset, batch_size=32):

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ==========================================================
# Build Model
# ==========================================================

def build_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==========================================================
# Train Model
# ==========================================================

def train_model(model, train_data, validation_data, epochs=5):

    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    return history


# ==========================================================
# Evaluate
# ==========================================================

def evaluate_model(model, validation_data):

    loss, accuracy = model.evaluate(validation_data, verbose=1)
    print(f"Loss : {loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    return loss, accuracy


# ==========================================================
# Prediction
# ==========================================================

def predict(model, validation_data):

    y_true = []
    y_pred = []
    predictions = model.predict(validation_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    for _, labels in validation_data:
        y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)
    return np.array(y_true), np.array(y_pred)


# ==========================================================
# Accuracy Graph
# ==========================================================

def plot_accuracy(history):

    plt.figure(figsize=(7,5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Accuracy.png"))

    plt.show()


# ==========================================================
# Loss Graph
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Loss.png"))

    plt.show()


# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap( cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Confusion_Matrix.png"))

    plt.show()


# ==========================================================
# Additional Functions for UI
# ==========================================================

def load_best_model():
    """Load the best saved model"""
    if os.path.exists(BEST_MODEL_PATH):
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        print(f"Model loaded from {BEST_MODEL_PATH}")
        return model
    else:
        print(f"No model found at {BEST_MODEL_PATH}")
        return None

def get_model_summary(model):
    """Get model summary as string"""
    from io import StringIO
    import sys
    
    # Capture the summary
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    summary_str = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return summary_str

def get_sample_predictions(model, validation_data, num_samples=5):
    """Get sample predictions with images"""
    samples = []
    
    # Take a few batches
    for images, labels in validation_data.take(1):
        # Get predictions
        predictions = model.predict(images)
        pred_labels = (predictions > 0.5).astype(int).flatten()
        
        # Get actual labels
        actual_labels = labels.numpy()
        
        # Store first num_samples
        for i in range(min(num_samples, len(images))):
            samples.append({
                'image': images[i],
                'actual': actual_labels[i],
                'predicted': pred_labels[i],
                'confidence': float(predictions[i][0])
            })
        break
    
    return samples

def get_dataset_info():
    """Get dataset overview information"""
    dataset, info = tfds.load(
        "cats_vs_dogs",
        split="train",
        as_supervised=True,
        with_info=True
    )
    
    total_images = info.splits["train"].num_examples
    train_size = int(0.8 * total_images)
    val_size = total_images - train_size
    
    return {
        'total_images': total_images,
        'train_size': train_size,
        'val_size': val_size,
        'num_classes': 2,
        'class_names': CLASS_NAMES,
        'image_shape': (224, 224, 3)
    }

def get_model_architecture_info():
    """Get model architecture details"""
    model = build_model()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Get layer info
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'params': layer.count_params(),
            'trainable': layer.trainable
        })
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'layers': layer_info
    }

def get_validation_data_samples():
    """Get validation data for testing"""
    train_data, validation_data = load_dataset()
    validation_data = prepare_dataset(validation_data)
    return validation_data

def get_training_history(history):
    """Get training history as dictionary"""
    if history is None:
        return None
    
    return {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }

def load_graphs():
    """Load all saved graph images"""
    graph_files = []
    if os.path.exists(GRAPH_FOLDER):
        for file in os.listdir(GRAPH_FOLDER):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                graph_files.append({
                    'name': file,
                    'path': os.path.join(GRAPH_FOLDER, file)
                })
    return graph_files


# ==========================================================
# Main
# ==========================================================

def main():

    train_data, validation_data = load_dataset()
    train_data = prepare_dataset(train_data)
    validation_data = prepare_dataset(validation_data)
    model = build_model()
    history = train_model(
        model,
        train_data,
        validation_data,
        epochs=5
    )
    evaluate_model(model, validation_data)
    y_true, y_pred = predict(model, validation_data)
    plot_accuracy(history)
    plot_loss(history)
    plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main()