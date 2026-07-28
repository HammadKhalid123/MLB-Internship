# Day 8 – Student Score Prediction System

## 📌 Project Overview

This project predicts a student's **Average Score** using **Linear Regression**.  
The dataset was preprocessed, split into training and testing sets, and used to train a machine learning model. A Streamlit application was also developed to allow users to make live predictions.

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 📂 Project Structure

```
Day_8/
│── app.py
│── preprocessing.py
│── linear_regression.py
│── requirements.txt
│── README.md
│
├── data/
│   └── cleaned_student_performance.csv
│
└── graphs/
    └── actual_vs_predicted.png
```

---

## 📊 Data Preprocessing

During preprocessing, the following steps were performed:

- Loaded the dataset using Pandas.
- Applied **Label Encoding** to the `Degree_Program` column.
- Applied **One-Hot Encoding** to the `Performance` column for analysis.
- Selected relevant features for training.
- Split the dataset into training and testing sets.
- Scaled numerical features using **StandardScaler**.

---

## 🎯 Why Train-Test Split is Important

Train-test splitting helps evaluate the model on unseen data.

- The **training set** is used to train the model.
- The **testing set** is used to evaluate its performance.
- It helps detect overfitting and measures how well the model generalizes to new data.

This project uses:

- **80% Training Data**
- **20% Testing Data**

---

## 🤖 Machine Learning Model

The project uses:

- **Linear Regression**

---

## 📈 Evaluation Metrics

The following metrics were used to evaluate the model:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📌 Model Performance & Observations

### Performance

- The Linear Regression model achieved a high **R² Score**, indicating a strong relationship between the selected features and the target variable.
- Error values (MAE, MSE, and RMSE) were relatively low, showing that the predictions were close to the actual scores.

### Observations

- Student subject scores had the strongest influence on the predicted average score.
- Feature scaling improved the consistency of the training process.
- The model performed well for this dataset and produced predictions close to the actual values.

---

## 🖥️ Streamlit Application

The Streamlit app provides:

- Dataset preview
- Model evaluation metrics
- Actual vs Predicted comparison
- Scatter plot visualization
- Feature coefficient visualization
- Live student score prediction

---

## ▶️ Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📚 Learning Outcomes

Through this project, I learned:

- Data preprocessing techniques.
- Label Encoding and One-Hot Encoding.
- Feature scaling using StandardScaler.
- The importance of train-test splitting.
- Training a Linear Regression model.
- Evaluating a regression model using MAE, MSE, RMSE, and R² Score.
- Building and deploying a machine learning application using Streamlit.