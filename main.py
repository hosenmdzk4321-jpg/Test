from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field
import joblib
import os

# --- Load the SINGLE Pipeline model at startup ---
print("Loading model.pkl (Pipeline)... (first request may take 15-25 sec)")
try:
    # Load the SINGLE file containing the full Pipeline (Vectorizer + Classifier)
    model = joblib.load("model.pkl") 
    print("Pipeline model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    # Raise the error if components can't be loaded
    raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="Fake News Detector API",
    description="Trained on 40k+ articles • 96% accurate",
    version="1.0"
)

# --- CORS Configuration (Fixes 405 OPTIONS error) ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------

class NewsInput(BaseModel):
    title: str = Field("", example="Hillary Clinton has died")
    text: str = Field("", example="Sources say she collapsed at home...")

@app.get("/")
def home():
    """Root endpoint to confirm API status."""
    return {
        "status": "API is LIVE",
        "model": "PassiveAggressive + TF-IDF (96% acc)",
        "endpoint": "POST /predict"
    }

@app.post("/predict")
def predict(news: NewsInput):
    """Processes text input using the Pipeline and returns the prediction."""
    if not news.title.strip() and not news.text.strip():
        raise HTTPException(status_code=400, detail="Please provide title or text")
        
    content = (news.title + " " + news.text).strip()
    
    # --- Prediction using the single Pipeline object ---
    # The 'model' (Pipeline) now handles both vectorization and prediction internally
    prediction = model.predict([content])[0]
    probas = model.predict_proba([content])[0]
    # ----------------------------------------------------
    
    # Handle class order safely
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
        "model_info": "PassiveAggressive + TF-IDF (via Pipeline)"
    }
