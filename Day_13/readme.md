# Day 13 – Convolutional Neural Networks (CNN)

## Overview

In this task, I learned the fundamentals of Convolutional Neural Networks (CNNs) and built an image classification model using TensorFlow/Keras. I trained the CNN on the Fashion MNIST dataset to classify clothing images into different categories. I also evaluated the model using multiple performance metrics, generated a confusion matrix, visualized predictions, and saved the best trained model for deployment in a Streamlit application.

---

# Why CNNs are Better than ANNs for Image Data?

Artificial Neural Networks (ANNs) treat every pixel independently, which results in a large number of parameters and loss of spatial information.

Convolutional Neural Networks (CNNs) automatically learn important image features such as edges, textures, and shapes by using convolution filters. This makes CNNs more efficient, accurate, and suitable for image classification tasks.

---

# Purpose of Convolution and Pooling Layers

## Convolution Layer

- Applies filters (kernels) to extract important image features.
- Learns edges, corners, textures, and object patterns.
- Reduces the need for manual feature engineering.

## Max Pooling Layer

- Reduces the spatial dimensions of feature maps.
- Keeps the most important information.
- Reduces computation and helps prevent overfitting.

---

# Model Architecture

- Input Layer (28 × 28 × 1)
- Conv2D (32 Filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Conv2D (64 Filters, 3×3, ReLU)
- MaxPooling2D (2×2)
- Flatten Layer
- Dense Layer (128 Neurons, ReLU)
- Output Layer (10 Neurons, Softmax)

---

# Model Performance

| Metric | Result |
|---------|--------|
| Training Accuracy | *(Your Final Accuracy)* |
| Validation Accuracy | *(Your Final Accuracy)* |
| Testing Accuracy | *(Your Final Accuracy)* |

---

# Visualizations Generated

- Class Distribution
- Training Accuracy Curve
- Training Loss Curve
- Confusion Matrix
- Per-Class Accuracy
- Predicted vs Actual Distribution
- Correctly Classified Images
- Incorrectly Classified Images

---

# Challenges Faced

- Understanding how CNN layers process image data.
- Choosing an appropriate CNN architecture.
- Interpreting the confusion matrix and prediction results.
- Managing model training time and preventing overfitting.
- Saving the trained model for later deployment.

---

# Solution

- Normalized image pixel values before training.
- Used Convolution and MaxPooling layers for feature extraction.
- Evaluated the model using accuracy, loss, and confusion matrix.
- Visualized correct and incorrect predictions for better analysis.
- Saved the best trained model (`best_fashion_mnist_model.h5`) for use in the Streamlit application.

---

# Key Learning

- CNNs automatically learn image features and outperform traditional ANNs on image classification tasks.
- Convolution and pooling layers help extract meaningful patterns while reducing computational complexity.
- Confusion matrices and prediction visualizations provide deeper insights into model performance.
- Saving trained models makes deployment and inference much easier.

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

# Files Included

- `cnn_practice.py`
- `fashion_mnist_cnn.py`
- `app.py`
- `best_fashion_mnist_model.h5`
- `Graphs/`
- `README.md`
- `requirements.txt`