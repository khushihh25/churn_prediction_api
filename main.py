from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("churn_model.joblib")

# Define request body
class Customer(BaseModel):
    tenure: int
    monthly_charges: float

# Create API
app = FastAPI(title="Churn Prediction API")

@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API!"}

@app.post("/predict/")
def predict_churn(data: Customer):
    input_data = np.array([[data.tenure, data.monthly_charges]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "churn_prediction": "Yes" if prediction else "No",
        "churn_probability": round(float(probability), 3)
    }
