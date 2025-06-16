import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_and_clean_data
from sklearn.metrics import classification_report


def train_and_save_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = load_and_clean_data("data/exoplanetdata.csv")

    X = df[[
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]]
    y = df['is_habitable']

    # Show class distribution
    print("Class distribution in dataset:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model Accuracy on Test Set:", accuracy_score(y_test, preds))

    joblib.dump(model, "models/model.pkl")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    train_and_save_model()
