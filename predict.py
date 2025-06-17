import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(input_data):
    input_df = pd.DataFrame([input_data])
    probability = model.predict_proba(input_df)[0][1]  # probability for class 1
    prediction = int(probability >= 0.5)
    return prediction, probability
