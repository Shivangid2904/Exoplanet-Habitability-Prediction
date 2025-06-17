import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_clean_data

def train_and_save_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load and prepare data
    df = load_and_clean_data("data/exoplanetdata.csv")

    X = df[[ 
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]]
    y = df['is_habitable']

    print("Original class distribution:\n", y.value_counts())

    # Train/test split (for final testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model with class_weight to handle imbalance
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Cross-validation scores (more reliable evaluation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print("Cross-validated F1 scores:", f1_scores)
    print("Mean F1 score:", f1_scores.mean())

    # Final model evaluation on the holdout test set
    preds = model.predict(X_test)
    print("\nFinal evaluation on test set:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(model, "models/model.pkl")
    print("\nModel saved to models/model.pkl")

if __name__ == "__main__":
    train_and_save_model()
