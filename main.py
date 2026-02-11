from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Enrollment Prediction API", version="1.0.0")

try:
    model = joblib.load('model.pkl')
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class PredictionInput(BaseModel):
    age: int
    gender: str             
    marital_status: str     
    salary: float
    employment_type: str    
    region: str             
    has_dependents: str    
    tenure_years: float

@app.get("/")
def home():
    return {"message": "Insurance Prediction API is active. Go to /docs for testing."}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model pipeline not available.")
    
    try:
        data_df = pd.DataFrame([input_data.dict()])
        
        prediction = model.predict(data_df)
        
        return {
            "status": "success",
            "enrolled_prediction": int(prediction[0]),
            "note": "1 = Enrolled, 0 = Not Enrolled"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)