from datasets import load_dataset
from src.camembert_model import CamembertSentimentModel

# -------------------------
# LOAD DATA
# -------------------------
dataset = load_dataset("allocine")

train_texts = dataset["train"]["review"]
train_labels = dataset["train"]["label"]


val_texts = dataset["validation"]["review"]
val_labels = dataset["validation"]["label"]

# -------------------------
# INIT MODEL
# -------------------------
model = CamembertSentimentModel()

# -------------------------
# TRAIN
# -------------------------
model.train(
    train_texts,
    train_labels,
    val_texts,
    val_labels
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("models/camembert")

print("✅ Camembert model trained and saved successfully.")