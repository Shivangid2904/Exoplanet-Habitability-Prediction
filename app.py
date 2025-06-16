import streamlit as st
import joblib
from predict import predict
from explain import explain_prediction, export_html_explanation
from preprocess import load_and_clean_data  # Needed for global SHAP

# Load the model once at startup
model = joblib.load("models/model.pkl")

st.title("🪐 Exoplanet Habitability Predictor")

# Input section
inputs = {
    "pl_rade": st.number_input("Planet Radius (Earth radii)", 0.0, 10.0, 1.0),
    "pl_bmasse": st.number_input("Planet Mass (Earth masses)", 0.0, 1000.0, 1.0),
    "pl_orbper": st.number_input("Orbital Period (days)", 0.0, 10000.0, 365.0),
    "pl_eqt": st.number_input("Equilibrium Temperature (K)", 0.0, 1000.0, 288.0),
    "pl_insol": st.number_input("Insolation Flux (Earth flux)", 0.0, 10.0, 1.0),
    "pl_orbeccen": st.number_input("Eccentricity", 0.0, 1.0, 0.0),
    "st_teff": st.number_input("Stellar Temperature (K)", 0.0, 10000.0, 5778.0),
    "st_rad": st.number_input("Stellar Radius (solar radii)", 0.0, 10.0, 1.0),
    "st_mass": st.number_input("Stellar Mass (solar mass)", 0.0, 10.0, 1.0),
    "st_met": st.number_input("Stellar Metallicity (dex)", -2.0, 1.0, 0.0),
    "sy_dist": st.number_input("Distance (parsecs)", 0.0, 10000.0, 10.0)
}

if st.button("🔮 Predict Habitability"):
    result = predict(inputs)
    st.write("### 🧾 Prediction:", "✅ **Habitable 🌍**" if result == 1 else "❌ **Not Habitable**")

    st.write("### 🧠 Explanation")
    explain_prediction(inputs, model)

    # Add Downloadable HTML Explanation
    html_path = export_html_explanation(inputs, model)
    with open(html_path, 'rb') as f:
        st.download_button("📄 Download SHAP Explanation (HTML)", f, file_name="shap_explanation.html")

