from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('../../model.pkl')
app = FastAPI(title = "Simple ML Service")

class PredictRequest(BaseModel):
    feature: float

@app.post("/predict")
def predict(request: PredictRequest):
    x = np.array([[request.feature]])
    prediction = model.predict(x)
    return {"feature": request.feature, "prediction": prediction[0]}