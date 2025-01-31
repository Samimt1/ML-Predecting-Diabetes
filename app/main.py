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
    with open("app/index.html") as f:
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

# Define the HTML GUI endpoint
@app.get("/", response_class=HTMLResponse)
async def get_gui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diabetes Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            input { margin: 5px; }
            button { margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Diabetes Prediction</h1>
        <form id="prediction-form">
            <label>Pregnancies: <input type="number" name="pregnancies" required></label><br>
            <label>Glucose: <input type="number" name="glucose" required></label><br>
            <label>Blood Pressure: <input type="number" name="blood_pressure" required></label><br>
            <label>Skin Thickness: <input type="number" name="skin_thickness" required></label><br>
            <label>Insulin: <input type="number" name="insulin" required></label><br>
            <label>BMI: <input type="number" name="bmi" required></label><br>
            <label>Diabetes Pedigree Function: <input type="number" step="0.01" name="diabetes_pedigree_function" required></label><br>
            <label>Age: <input type="number" name="age" required></label><br>
            <button type="submit">Predict</button>
        </form>
        <h2 id="result"></h2>
        <script>
            document.getElementById('prediction-form').onsubmit = async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const data = Object.fromEntries(formData);
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById('result').innerText = 'Prediction: ' + result.prediction + ', Probability: ' + result.probability;
            };
        </script>
    </body>
    </html>
    """
