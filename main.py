from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Uniblox Insurance Prediction API is Running!"}
model = joblib.load('model.pkl')

class PredictionInput(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    data_df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(data_df)
    return {"enrolled_prediction": int(prediction[0])}