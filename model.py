import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from preprocess import load_and_clean_data
import seaborn as sns
import matplotlib.pyplot as plt

def train_with_smote_cv():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load and preprocess data
    df = load_and_clean_data("data/exoplanetdata.csv")

    # Feature and target split
    X = df[[ 
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]]
    y = df["is_habitable"]

    print("Original Class Distribution:\n", y.value_counts())

    # Set up SMOTE + Classifier pipeline
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')

    print("\nCross-Validated F1 Scores (Positive Class):", f1_scores)
    print("Mean F1 Score:", np.mean(f1_scores))

    # Train final model on full dataset and save
    pipeline.fit(X, y)
    joblib.dump(pipeline, "models/model.pkl")
    print("\nFinal model trained on full dataset and saved to models/model.pkl")

    # Evaluate on a holdout test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Accuracy score added here
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy on Holdout Test Set:", acc)

    print("\nClassification Report on Holdout Test Set:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    train_with_smote_cv()
