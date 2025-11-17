from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
from typing import Literal

# Load model at startup
print("Loading model.pkl... (first request may take 15-25 sec)")
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

app = FastAPI(
    title="Fake News Detector API",
    description="Trained on 40k+ articles • 96% accurate",
    version="1.0"
)

class NewsInput(BaseModel):
    title: str = Field("", example="Hillary Clinton has died")
    text: str = Field("", example="Sources say she collapsed at home...")

@app.get("/")
def home():
    return {
        "status": "API is LIVE",
        "model": "PassiveAggressive + TF-IDF (96% acc)",
        "endpoint": "POST /predict"
    }

@app.post("/predict")
def predict(news: NewsInput):
    if not news.title.strip() and not news.text.strip():
        raise HTTPException(status_code=400, detail="Please provide title or text")

    content = (news.title + " " + news.text).strip()
    
    prediction = model.predict([content])[0]
    probas = model.predict_proba([content])[0]
    
    # Handle class order safely
    classes = model.classes_
    fake_idx = list(classes).index('Fake') if 'Fake' in classes else -1
    real_idx = list(classes).index('Real') if 'Real' in classes else -1

    return {
        "prediction": prediction,
        "confidence": round(float(max(probas)), 4),
        "fake_probability": round(float(probas[fake_idx]), 4) if fake_idx != -1 else 0.0,
        "real_probability": round(float(probas[real_idx]), 4) if real_idx != -1 else 0.0,
        "input_words": len(content.split()),
        "model_info": "PassiveAggressive + trigrams"
    }
