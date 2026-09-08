import json
import joblib
import pandas as pd

# -----------------------------
# Load model and configuration
# -----------------------------

model = joblib.load("models/model.pkl")

with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

with open("models/threshold.json", "r") as f:
    threshold_data = json.load(f)

threshold = float(threshold_data["threshold"])

print("Model loaded successfully!")
print("Number of features:", len(feature_names))
print("Threshold:", threshold)

# -----------------------------
# Example input
# -----------------------------

sample = {
    "volt": 170,
    "rotate": 350,
    "pressure": 100,
    "vibration": 40,

    "hour": 12,
    "day": 20,
    "month": 10,
    "dayofweek": 1,

    "volt_mean_6h": 170,
    "volt_std_6h": 15,
    "rotate_mean_6h": 370,
    "rotate_std_6h": 50,
    "pressure_mean_6h": 100,
    "pressure_std_6h": 8,
    "vibration_mean_6h": 41,
    "vibration_std_6h": 5,

    "volt_change_1h": -5,
    "rotate_change_1h": -20,
    "pressure_change_1h": -3,
    "vibration_change_1h": 1,

    "volt_mean_24h": 170,
    "volt_std_24h": 14,
    "rotate_mean_24h": 370,
    "rotate_std_24h": 52,
    "pressure_mean_24h": 100,
    "pressure_std_24h": 8,
    "vibration_mean_24h": 41,
    "vibration_std_24h": 5,

    "volt_change_24h": -10,
    "rotate_change_24h": -40,
    "pressure_change_24h": -5,
    "vibration_change_24h": 0
}

# -----------------------------
# Prepare input
# -----------------------------

X = pd.DataFrame([sample])

# Ensure exact feature order
X = X[feature_names]

# -----------------------------
# Prediction
# -----------------------------

probability = model.predict_proba(X)[0][1]

prediction = int(probability >= threshold)

print("\nPrediction Results")
print("-----------------------------")
print("Failure probability:", round(probability, 4))
print("Threshold:", threshold)

if prediction == 1:
    print("Prediction: FAILURE")
else:
    print("Prediction: NO FAILURE")