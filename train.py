from datasets import load_dataset
import joblib

from src.preprocessing import clean_text
from src.tfidf_model import TfidfSentimentModel

# Load dataset
dataset = load_dataset("allocine")

train_texts = dataset["train"]["review"]
train_labels = dataset["train"]["label"]

val_texts = dataset["validation"]["review"]
val_labels = dataset["validation"]["label"]

# Clean
X_train_clean = [clean_text(t) for t in train_texts]
X_val_clean = [clean_text(t) for t in val_texts]

# Train model
model = TfidfSentimentModel()
model.fit(X_train_clean, train_labels)

# Save model + vectorizer
joblib.dump(model, "models/tfidf_model.joblib")

print("✅ Model saved!")