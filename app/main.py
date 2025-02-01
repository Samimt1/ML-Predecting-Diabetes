from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("app/models/diabetes_rf_model.pkl")
scaler = joblib.load("app/models/scaler.pkl")

# Define the input data schema using Pydantic
class PatientData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

# Initialize FastAPI app
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Route to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("app/static/index.html") as f:
        return f.read()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input data to a numpy array
        input_data = np.array([
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age
        ]).reshape(1, -1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)

        # Return the prediction and probability
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))