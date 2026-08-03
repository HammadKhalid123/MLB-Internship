# Day 12 - Deep Learning with Artificial Neural Networks (ANN)

## Overview

In this task, I learned the fundamentals of **Deep Learning** and built my first **Artificial Neural Network (ANN)** using **TensorFlow/Keras**. I trained the model on the **Fashion MNIST** dataset to classify different types of clothing images. I also explored different activation functions and evaluated the model using training, validation, and testing accuracy.

---

## What is Deep Learning?

Deep Learning is a branch of Machine Learning that uses **Artificial Neural Networks (ANNs)** with multiple layers to automatically learn patterns from data. It is widely used in image classification, object detection, natural language processing, speech recognition, and many other AI applications.

---

## Machine Learning vs Deep Learning

| Machine Learning | Deep Learning |
|------------------|---------------|
| Requires manual feature engineering. | Learns features automatically from data. |
| Works well on small to medium datasets. | Performs better with large datasets. |
| Uses simpler algorithms such as Linear Regression and Decision Trees. | Uses multi-layer Artificial Neural Networks. |
| Faster to train. | Requires more computational power and training time. |

---

## What is a Perceptron?

A **Perceptron** is the basic building block of an Artificial Neural Network. It receives input values, applies weights and bias, calculates a weighted sum, and passes the result through an activation function to produce the final output.

---

## Activation Functions Explored

### ReLU (Rectified Linear Unit)
- Returns `max(0, x)`.
- Commonly used in hidden layers.
- Helps the model learn faster and reduces the vanishing gradient problem.

### Sigmoid
- Produces values between **0 and 1**.
- Commonly used in binary classification output layers.

### Tanh
- Produces values between **-1 and 1**.
- Often used in hidden layers and generally performs better than Sigmoid.

### Softmax
- Converts outputs into probability values.
- Commonly used in the output layer for multi-class classification problems such as Fashion MNIST.

---

## Model Architecture

- Input Layer (28 × 28 Images)
- Flatten Layer
- Hidden Layer (128 Neurons, ReLU)
- Output Layer (10 Neurons, Softmax)

---

## Model Performance

| Metric | Result |
|---------|--------|
| Training Accuracy | **88.9%** |
| Validation Accuracy | **87.3%** |
| Testing Accuracy | **86.6%** |

---

## Observations

- The Fashion MNIST dataset was normalized before training.
- The ANN successfully classified different clothing categories.
- Training and validation accuracy improved with each epoch.
- Fashion MNIST is more challenging than the handwritten MNIST dataset because several clothing categories have similar visual features.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Files Included

- `tensorflow_practice.py`
- `ann_model.py`
- `fashion_mnist_project.py`
- `training_accuracy.png`
- `sample_predictions.png`
- `README.md`