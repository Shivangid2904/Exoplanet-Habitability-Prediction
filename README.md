
🌍 Exoplanet Habitability Predictor
A Streamlit-powered web app that predicts whether an exoplanet could be potentially habitable, based on planetary parameters and explainable AI techniques.

🚀 Features
🪐 Input planetary parameters like radius, equilibrium temperature, and stellar flux

🤖 Predict habitability using a trained Random Forest model

🧠 Visualize model reasoning with SHAP (SHapley Additive exPlanations)

💻 User-friendly Streamlit interface for real-time predictions

🛠️ Getting Started
✅ Installation
pip install -r requirements.txt
🧠 Train the Model
python model.py
(This will process data and save the trained model as models/model.pkl)

🚦 Run the App
streamlit run app.py

📁 Folder Structure

exoplanet-habitability-prediction/
├── app.py               # Streamlit UI
├── model.py             # Training script
├── predict.py           # Prediction logic
├── explain.py           # SHAP explanations
├── preprocess.py        # Data cleaning and feature selection
├── models/
│   └── model.pkl        # Trained model
├── data/
│   └── exoplanet_data.csv
├── requirements.txt
└── README.md
🧬 Dataset
Source: NASA Exoplanet Archive
⚠️ Note on Class Imbalance
The dataset is highly imbalanced, with significantly fewer known potentially habitable exoplanets compared to non-habitable ones. To address this:

We applied SMOTE (Synthetic Minority Over-sampling Technique) during model training.

This helps the model generalize better and reduces bias toward the majority class.
Preprocessed to start from header row 89

Includes features like:

pl_rade – Planet Radius

pl_bmasse – Planet Mass

pl_orbper – Orbital Period

pl_eqt – Equilibrium Temperature

pl_insol – Insolation Flux

pl_orbeccen – Orbital Eccentricity

st_teff – Stellar Temperature

st_rad – Stellar Radius

st_mass – Stellar Mass

st_met – Stellar Metallicity

sy_dist – Distance from Earth



🔗 Demo
🌐 Live App: https://exoplanet-habitability-prediction.streamlit.app/
📁 GitHub Repository: This repo

📜 License
This project is licensed under the MIT License. You’re free to use, modify, and distribute it with proper credit.

