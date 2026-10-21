from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
import os

# Logging configuration
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("model_service")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("logs/model_service.log", mode="a")
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define FastAPI app
app = FastAPI()

# Middleware for request logging


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} completed in {duration:.4f}s with status {response.status_code}"
    )
    return response

# Model and input schema


class InputData(BaseModel):
    x: float


model = joblib.load("model.pkl")

# Health check endpoint


@app.get("/health")
def health_check():
    return {"status": "ok"}

# Prediction endpoint


@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()  # Measure inference latency

    x_value = np.array([[data.x]])
    prediction = model.predict(x_value)
    pred_value = float(prediction[0])

    latency = time.time() - start_time

    # Log prediction details + model latency
    logger.info(
        f"x={data.x:.3f} -> prediction={pred_value:.4f} | latency={latency:.4f}s")

    return {"prediction": pred_value}
