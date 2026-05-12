from datasets import load_dataset
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Load the allocine dataset: train and val
dataset = load_dataset("allocine")
train_texts = dataset["train"]["review"]
train_labels = dataset["train"]["label"]

val_texts = dataset["validation"]["review"]
val_labels = dataset["validation"]["label"]

#clean the data: remove extraspeces, lowercase the text
# Minimal preprocessing (lowercasing and whitespace normalization) was applied. 
# More aggressive cleaning was avoided to preserve linguistic information and ensure 
# fair comparison with transformer-based models.
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()

X_train_clean = [clean_text(t) for t in train_texts]
X_val_clean = [clean_text(t) for t in val_texts]

#vectorizing the text
vectorizer = TfidfVectorizer(max_features=10000)

X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_val_tfidf = vectorizer.transform(X_val_clean)

#Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, train_labels)

y_pred = model.predict(X_val_tfidf)

print(classification_report(val_labels, y_pred, digits=4))

#Baseline results
#TF-IDF + Logistic Regression: 92% accuracy

#Next section:

#Future improvement
#CamemBERT fine-tuning planned