# Day 10 - Model Evaluation & Hyperparameter Tuning

## Overview

This project focuses on evaluating a Logistic Regression model and improving its performance using **GridSearchCV** for hyperparameter tuning.

The goal was to compare a baseline model with an optimized model and understand how selecting the right hyperparameters can improve overall performance.

---

## Dataset

- Dataset: Breast Cancer Wisconsin Dataset
- Total Features: 30
- Training Samples: 114
- Model Used: Logistic Regression

---

## Objectives

- Train a baseline Logistic Regression model.
- Evaluate the model using different classification metrics.
- Understand the importance of model evaluation.
- Perform hyperparameter tuning using GridSearchCV.
- Compare the performance of the baseline and tuned models.

---

# What I Learned About Model Evaluation

Model evaluation is the process of measuring how well a machine learning model performs on unseen data. Instead of relying only on accuracy, multiple evaluation metrics provide a better understanding of the model's strengths and weaknesses.

The main evaluation metrics learned include:

- **Accuracy** – Measures the percentage of correct predictions.
- **Precision** – Indicates how many predicted positive cases were actually positive.
- **Recall** – Measures how many actual positive cases were correctly identified.
- **F1-Score** – Balances Precision and Recall, making it useful for imbalanced datasets.
- **Confusion Matrix** – Shows the number of correct and incorrect predictions for each class.

Using multiple metrics provides a more reliable evaluation than accuracy alone.

---

# What is Hyperparameter Tuning?

Hyperparameter tuning is the process of finding the best combination of model parameters before training.

Unlike model parameters that are learned during training, hyperparameters are set by the user and directly affect the learning process.

For Logistic Regression, important hyperparameters include:

- Regularization strength (**C**)
- Solver
- Maximum iterations (**max_iter**)
- Class weight

Finding the right combination of these values helps improve model performance and generalization.

---

# Why Hyperparameter Tuning Matters

Hyperparameter tuning helps to:

- Improve prediction accuracy.
- Reduce overfitting and underfitting.
- Find the most suitable model configuration.
- Increase model reliability on unseen data.
- Achieve better overall classification performance.

In this project, **GridSearchCV** was used to automatically test multiple parameter combinations and select the best one using cross-validation.

---

# Best Parameters Found by GridSearchCV

```python
{
    'C': 0.1,
    'class_weight': None,
    'max_iter': 100,
    'solver': 'liblinear'
}
```
# Key Observations

- The tuned model achieved higher accuracy than the baseline model.
- GridSearchCV successfully identified the optimal hyperparameters.
- Precision, Recall, and F1-Score improved after tuning.
- The confusion matrix showed fewer classification errors.
- Cross-validation helped select a model that generalizes better to unseen data.
- Even a simple model like Logistic Regression can achieve excellent performance when properly tuned.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# Conclusion

This project demonstrated the importance of both **model evaluation** and **hyperparameter tuning** in machine learning.

By evaluating the model with multiple performance metrics and optimizing it using **GridSearchCV**, the Logistic Regression model achieved a noticeable improvement in prediction performance, increasing accuracy from **97.37%** to **99.12%** while reducing classification errors.

This exercise highlighted that selecting appropriate hyperparameters is an essential step in building reliable and high-performing machine learning models.