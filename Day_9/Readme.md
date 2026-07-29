# Day 11 – Classification & Model Evaluation

## 📌 Objective

The goal of this task was to understand Classification problems, learn how to evaluate Machine Learning models, and build a classification model using Logistic Regression. As a bonus, a Decision Tree model was also implemented for comparison.

---

# Topics Covered

- Model Evaluation
- Training vs Testing Performance
- Overfitting & Underfitting
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Logistic Regression
- Decision Tree Classification

---

# 📂 Files Included

- `classification_practice.py`
- `iris_classification.py`
- `decision_tree_comparison.py` *(Bonus)*
- `confusion_matrix.png`

---

# Datasets Used

### Breast Cancer Dataset
Used for classification practice and model evaluation.

### Iris Dataset
Used for building the Iris Flower Classification System.

---

# What is Classification?

Classification is a supervised Machine Learning task in which the model predicts a category or class label instead of a continuous numerical value.

### Examples

- Spam or Not Spam
- Fraud or Not Fraud
- Benign or Malignant Tumor
- Iris Flower Species Prediction

---

# Difference Between Regression and Classification

| Regression | Classification |
|------------|---------------|
| Predicts continuous values | Predicts categories or classes |
| Output is a number | Output is a label |
| Example: House Price Prediction | Example: Email Spam Detection |
| Common Metrics: MAE, MSE, RMSE, R² Score | Common Metrics: Accuracy, Precision, Recall, F1-Score |

---

# Evaluation Metrics Used

## Accuracy
Measures the overall percentage of correct predictions made by the model.

## Precision
Measures how many predicted positive samples are actually positive.

## Recall
Measures how many actual positive samples are correctly identified.

## F1-Score
The harmonic mean of Precision and Recall. It provides a balanced evaluation when both metrics are important.

## Confusion Matrix
Shows the number of correct and incorrect predictions for each class and helps understand model performance in detail.

---

# Models Used

## Logistic Regression

- Trained on the Iris Dataset.
- StandardScaler was applied before training.
- Evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Decision Tree (Bonus)

- Trained on the same Iris Dataset.
- Used to compare its performance with Logistic Regression.
- Evaluated using the same metrics.

---

# Model Performance and Observations

### Logistic Regression

- Successfully classified Iris flower species.
- Achieved very high accuracy on the test dataset.
- Precision, Recall, and F1-Score were excellent for all classes.
- The Confusion Matrix showed very few or no misclassifications.

### Decision Tree

- Successfully classified the Iris dataset.
- Produced performance comparable to Logistic Regression.
- Demonstrated how tree-based models can solve classification problems without feature scaling.

---

# Learning Outcomes

After completing this task, I learned:

- The difference between Regression and Classification.
- How Logistic Regression works.
- How Decision Tree Classification works.
- How to split data into training and testing sets.
- How to evaluate a classification model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- How to interpret model performance and compare different classification algorithms.

---

# 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

# Conclusion

This task provided hands-on experience with Classification in Machine Learning. I implemented Logistic Regression and Decision Tree models, evaluated them using different performance metrics, and learned how to interpret the results using a Confusion Matrix.