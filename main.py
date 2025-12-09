from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # 💡 NEW: Import CORS middleware
from pydantic import BaseModel, Field
import joblib
import os
from typing import Literal

# Load model at startup
print("Loading model.pkl... (first request may take 15-25 sec)")
try:
    # Ensure this path is correct relative to where your app runs on Render
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    # Re-raise the error so the service fails to start if the model can't be loaded
    raise

app = FastAPI(
    title="Fake News Detector API",
    description="Trained on 40k+ articles • 96% accurate",
    version="1.0"
)

# --- 🛠️ CORS Configuration for Flutter/Web App Access ---
# This middleware allows requests from any origin ("*") to prevent the "405 Method Not Allowed" error
# during the CORS preflight (OPTIONS) check.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)
# -----------------------------------------------------------

class NewsInput(BaseModel):
    title: str = Field("", example="Hillary Clinton has died")
    text: str = Field("", example="Sources say she collapsed at home...")

@app.get("/")
def home():
    """
    Root endpoint to confirm the API is live and provide basic info.
    """
    return {
        "status": "API is LIVE",
        "model": "PassiveAggressive + TF-IDF (96% acc)",
        "endpoint": "POST /predict",
        "cors_enabled": True
    }

@app.post("/predict")
def predict(news: NewsInput):
    """
    Accepts news title and text, processes them, and returns a prediction 
    (Fake or Real) with confidence scores.
    """
    if not news.title.strip() and not news.text.strip():
        raise HTTPException(status_code=400, detail="Please provide title or text")
        
    # Combine title and text for prediction
    content = (news.title + " " + news.text).strip()
    
    # Perform prediction and probability calculation
    prediction = model.predict([content])[0]
    probas = model.predict_proba([content])[0]
    
    # Handle class order safely to ensure 'Fake' and 'Real' probabilities are correct
    classes = model.classes_
    fake_idx = list(classes).index('Fake') if 'Fake' in classes else -1
    real_idx = list(classes).index('Real') if 'Real' in classes else -1
    
    # Format and return the results
    return {
        "prediction": prediction,
        "confidence": round(float(max(probas)), 4),
        "fake_probability": round(float(probas[fake_idx]), 4) if fake_idx != -1 else 0.0,
        "real_probability": round(float(probas[real_idx]), 4) if real_idx != -1 else 0.0,
        "input_words": len(content.split()),
        "model_info": "PassiveAggressive + trigrams"
    }
