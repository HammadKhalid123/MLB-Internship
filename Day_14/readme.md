# Day 14 – Cats vs Dogs Image Classification using CNN

## Objective

Build a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow and TensorFlow Datasets.

---

## Topics Covered

- Image Classification
- Convolutional Neural Networks (CNN)
- TensorFlow Datasets (TFDS)
- Image Preprocessing
- Data Augmentation
- Model Training
- Model Evaluation
- Model Checkpoint
- Accuracy and Loss Visualization

---

## Dataset

**Dataset:** Cats vs Dogs (TensorFlow Datasets)

- Total Images: **23,262**
- Training Images: **18,609**
- Validation Images: **4,653**

> Note: Some corrupted images were automatically skipped by TensorFlow.

---

## Model Architecture

- Input Layer
- Data Augmentation
- Convolution Layers
- MaxPooling Layers
- Dropout Layers
- Dense Layers
- Softmax Output Layer

---

## Training Configuration

- Epochs: **5**
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**
- Metric: **Accuracy**

---

## Model Performance

| Metric | Value |
|---------|-------|
| Best Validation Accuracy | **98.99%** |
| Final Validation Accuracy | **98.75%** |
| Test Accuracy | **98.75%** |
| Test Loss | **0.0404** |

---

## Performance Target

- Minimum Validation Accuracy: **90%** ✅
- Target Validation Accuracy: **93%+** ✅ Achieved

---

## Observations

- Successfully trained a CNN for binary image classification.
- Achieved validation accuracy significantly higher than the target.
- TensorFlow automatically skipped corrupted images in the dataset.
- Model checkpoint saved the best-performing model during training.

---

## Files

- `cats_vs_dogs_classifier.py`
- `best_cats_vs_dogs_model.keras`

---

## Libraries Used

- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib