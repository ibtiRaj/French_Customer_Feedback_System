import joblib
from src.preprocessing import clean_text
from src.camembert_model import CamembertSentimentModel


model = joblib.load("models/tfidf_model.joblib")
model_camembert = CamembertSentimentModel("models/camembert")
print(model_camembert)

def predict(text, model_name= "TFIDF"):
    text_clean = clean_text(text)
    if model_name== "TFIDF":
        return model.predict([text_clean])[0]
    if model_name == "camembert":
        return model_camembert.predict([text_clean])[0]

def debug_features(text, model_name= "TFIDF"):
    text_clean = clean_text(text)
    if model_name == "TFIDF":
        X = model.vectorizer.transform([text_clean])
        feature_names = model.vectorizer.get_feature_names_out()
    # if model_name == "camembert":
    #     X = model_camembert.tokenizer.transform([text_clean])
    #     feature_names = model_camembert.tokenizer.get_feature_names_out()
    indices = X.nonzero()[1]
    words = [feature_names[i] for i in indices]
    
    return words

if __name__ == "__main__":
    text = input("Enter a review: ")
    pred = predict(text)
    print("Prediction of TFIDF:", pred)
    print("Recognized words TFIDF model:", debug_features(text))

    pred_camembert = predict(text, "camembert")
    print("Prediction of camembert model:", pred_camembert)
    # print("Recognized words camembert model:", debug_features(text, "camembert"))