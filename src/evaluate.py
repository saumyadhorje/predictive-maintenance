import json
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------
# Load saved model
# -----------------------------

model = joblib.load("models/model.pkl")

# -----------------------------
# Load test data
# -----------------------------

X_test = joblib.load(
    "notebooks/saved_model_original/X_test.pkl"
)

y_test = joblib.load(
    "notebooks/saved_model_original/y_test.pkl"
)

# -----------------------------
# Load threshold
# -----------------------------

with open("models/threshold.json", "r") as f:
    threshold_data = json.load(f)

threshold = float(threshold_data["threshold"])

print("Model loaded successfully!")
print("Test data shape:", X_test.shape)
print("Threshold:", threshold)

# -----------------------------
# Predict probabilities
# -----------------------------

probabilities = model.predict_proba(X_test)[:, 1]

# Apply our selected threshold
predictions = (probabilities >= threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nFinal Metrics:")
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall   :", recall_score(y_test, predictions))
print("F1-score :", f1_score(y_test, predictions))
print("ROC-AUC  :", roc_auc_score(y_test, probabilities))