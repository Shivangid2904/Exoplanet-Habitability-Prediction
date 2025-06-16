
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
Preprocessed to start from header row 89

Includes features like:

Planet Radius (pl_rade)
Equilibrium Temperature (pl_eqt)
Stellar Flux (pl_insol)

🔗 Demo
🌐 Live App: https://exoplanet-habitability-prediction.streamlit.app/
📁 GitHub Repository: This repo
