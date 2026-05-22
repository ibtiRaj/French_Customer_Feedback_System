from transformers import CamembertTokenizer, CamembertForSequenceClassification
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

model_path = "models/camembert"

tokenizer = CamembertTokenizer.from_pretrained(model_path)
model = CamembertForSequenceClassification.from_pretrained(model_path)

model.eval()

from transformers import CamembertTokenizer, CamembertForSequenceClassification
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

model_path = "models/camembert"

tokenizer = CamembertTokenizer.from_pretrained(model_path)
model = CamembertForSequenceClassification.from_pretrained(model_path)

model.eval()

dataset = load_dataset("allocine")

texts = dataset["test"]["review"]   # ou "test"
labels = dataset["test"]["label"]

preds = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    preds.append(pred)


accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print("Accuracy:", accuracy)
print("F1:", f1)
print(classification_report(labels, preds))

####TF-IDF

import joblib
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load("models/tfidf_model.joblib")

# suppose X_val, y_val déjà préparés
preds = model.predict(texts)

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print("TF-IDF Accuracy:", accuracy)
print("TF-IDF F1:", f1)