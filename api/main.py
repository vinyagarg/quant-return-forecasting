from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

from src.model import FEATURES

app = FastAPI(title="Quant Return Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this if you ever deploy publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
dataset = pd.read_parquet("data/processed/dataset.parquet")

@app.get("/")
def root():
    return {"status": "ok", "message": "Quant Return Forecasting API is running"}

@app.get("/predictions/latest")
def latest_predictions():
    latest_date = dataset.index.max()
    latest = dataset[dataset.index == latest_date].copy()
    preds = model.predict(latest[FEATURES])
    latest["predicted_return"] = preds
    result = latest[["ticker", "predicted_return"]].sort_values(
        "predicted_return", ascending=False
    )
    return result.to_dict("records")