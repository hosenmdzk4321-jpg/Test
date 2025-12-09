from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import os

# --- 💡 Component Loading ---
# The model is now assumed to be split into two files:
# 1. vectorizer.pkl (TfidfVectorizer instance, fitted)
# 2. classifier.pkl (PassiveAggressiveClassifier instance, trained)
print("Loading model components... (first request may take 15-25 sec)")
try:
    # Load the vectorizer (needed to convert text input into features)
    vectorizer = joblib.load("vectorizer.pkl")
    # Load the classifier (the actual trained model)
    classifier = joblib.load("classifier.pkl")
    print("Model components loaded successfully!")
except Exception as e:
    print(f"ERROR loading components: {e}")
    # Raise the error if components can't be loaded, preventing the API from starting broken
    raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="Fake News Detector API",
    description="Trained on 40k+ articles • 96% accurate",
    version="1.0"
)

# --- CORS Configuration ---
# Allows requests from any origin, fixing the "405 Method Not Allowed" for OPTIONS preflight checks.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)
# ----------------------------

class NewsInput(BaseModel):
    title: str = Field("", example="Hillary Clinton has died")
    text: str = Field("", example="Sources say she collapsed at home...")

@app.get("/")
def home():
    """Root endpoint to confirm API status."""
    return {
        "status": "API is LIVE",
        "model": "PassiveAggressive + TF-IDF (96% acc)",
        "endpoint": "POST /predict",
        "cors_enabled": True
    }

@app.post("/predict")
def predict(news: NewsInput):
    """Processes text input, vectorizes it, and returns the prediction."""
    if not news.title.strip() and not news.text.strip():
        raise HTTPException(status_code=400, detail="Please provide title or text")
        
    content = (news.title + " " + news.text).strip()
    
    # --- 🎯 THE FIX: Vectorize input before predicting ---
    # 1. Transform the raw text content into a numerical feature vector
    feature_vector = vectorizer.transform([content])
    
    # 2. Predict using the numerical feature vector
    prediction = classifier.predict(feature_vector)[0]
    probas = classifier.predict_proba(feature_vector)[0]
    # ----------------------------------------------------
    
    # Handle class order safely
    classes = classifier.classes_
    fake_idx = list(classes).index('Fake') if 'Fake' in classes else -1
    real_idx = list(classes).index('Real') if 'Real' in classes else -1
    
    # Format and return the results
    return {
        "prediction": prediction,
        "confidence": round(float(max(probas)), 4),
        "fake_probability": round(float(probas[fake_idx]), 4) if fake_idx != -1 else 0.0,
        "real_probability": round(float(probas[real_idx]), 4) if real_idx != -1 else 0.0,
        "input_words": len(content.split()),
        "model_info": "PassiveAggressive + TF-IDF"
    }
