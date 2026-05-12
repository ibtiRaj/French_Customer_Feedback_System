from fastapi import FastAPI
import joblib
from src.camembert_model import CamembertSentimentModel



from src.preprocessing import clean_text

app = FastAPI(title="French Sentiment Analysis API")

# Load model
model = joblib.load("models/tfidf_model.joblib")
camembert_model = CamembertSentimentModel("models/camembert")

@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(text: str):
    text_clean = clean_text(text)
    pred = model.predict([text_clean])[0]

    return {
        "text": text,
        "sentiment": int(pred)
    }
@app.post("/predict_all")
def predict_all(text: str):
    tfidf_pred = model.predict([clean_text(text)])[0]
    cam_pred = camembert_model.predict(text)[0]

    return {
        "text": text,
        "tfidf_prediction": int(tfidf_pred),
        "camembert_prediction": int(cam_pred)
    }
@app.get("/keywords")
def keywords():
    return model.get_top_features(10)