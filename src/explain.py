import os
import json
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "shap_explainability")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load trained model
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

# Load feature names
with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)


# Load test data
X_test = pd.read_pickle(
    os.path.join(BASE_DIR, "notebooks", "saved_model_original", "X_test.pkl")
)

# Keep only the 32 model features
X_test = X_test[feature_names]

# Use a smaller sample for SHAP
X_sample = X_test.sample(
    n=min(2000, len(X_test)),
    random_state=42
)


print("Model loaded successfully!")
print("Features:", len(feature_names))
print("SHAP sample shape:", X_sample.shape)


# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_sample)


# Global feature importance
plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()

global_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)

plt.savefig(global_path, dpi=150, bbox_inches="tight")
plt.close()

print("Global SHAP plot saved:", global_path)


# SHAP summary / beeswarm plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.title("SHAP Feature Impact")
plt.tight_layout()

summary_path = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()

print("SHAP summary plot saved:", summary_path)


# Explain one individual prediction
sample_index = 0

plt.figure()

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value,
        data=X_sample.iloc[sample_index],
        feature_names=feature_names
    ),
    show=False
)

plt.tight_layout()

waterfall_path = os.path.join(
    OUTPUT_DIR,
    "shap_waterfall.png"
)

plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
plt.close()

print("SHAP waterfall plot saved:", waterfall_path)

print("\nSHAP analysis completed successfully!")