import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading Dataset
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# Exploring Dataset
# print("Features shape:", cancer.data.shape)
# print("Target shape:", cancer.target.shape)
# print("Feature names:\n", cancer.feature_names)
# print("Target names:", cancer.target_names)

# print("\nPehle 5 rows:")
# print(df.head())

# print("\nTarget distribution:")
# print(df["target"].value_counts())

# print("\nBasic statistics:")
# print(df.describe())

# print("\nTypes of data in each column:")
# print(df.dtypes)

# print("\nMissing values:")
# print(df.isnull().sum())

# Feature Selection
mi_scores = mutual_info_classif(df.drop("target", axis=1), df["target"])

feature_importance = pd.DataFrame({
    "feature": cancer.feature_names,
    "mi_score": mi_scores
}).sort_values("mi_score", ascending=False)

# print("\nFeatures sorted by Mutual Information:")
# print(feature_importance)

# Correlation Analysis
correlation_with_target = df.corr()["target"].sort_values(ascending=False)

print("\nCorrelation with target variable:")
# print(correlation_with_target)

# Selecting Features
selected_features = [
    "worst concave points",
    "worst perimeter",
    "mean concave points",
    "worst radius",
    "mean perimeter",
    "worst area",
    "mean radius",
    "mean area",
    "mean concavity",
    "worst concavity"
]

X = df[selected_features]
y = df["target"]

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Total samples in Training:", X_train.value_counts().sum())
print("Total samples in Testing:", X_test.value_counts().sum())
print("Total samples in Training (Target):", y_train.value_counts().sum())
print("Total samples in Testing (Target):", y_test.value_counts().sum())

# Training Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Making Predictions
predictions = model.predict(X_test)

# Evaluating Model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Visualizing Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()