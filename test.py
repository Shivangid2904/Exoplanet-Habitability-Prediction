import shap
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

X_instance = pd.DataFrame([{
    "pl_rade": 1,
    "pl_bmasse": 1,
    "pl_orbper": 365,
    "pl_eqt": 288,
    "pl_insol": 1,
    "pl_orbeccen": 0,
    "st_teff": 5778,
    "st_rad": 1,
    "st_mass": 1,
    "st_met": 0,
    "sy_dist": 10
}])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_instance)

# ✅ Check format and unpack correctly
if isinstance(shap_values, list):
    # For binary classification: 1 = "habitable"
    shap_for_pred_class = shap_values[1][0]
else:
    # Single output — directly use
    shap_for_pred_class = shap_values[0]

# Display SHAP values
for feature, value, shap_val in zip(X_instance.columns, X_instance.iloc[0], shap_for_pred_class):
    print(f"{feature}\t{value}\t{shap_val}")
