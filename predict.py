import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction