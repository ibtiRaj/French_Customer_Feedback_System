# 🇫🇷 French Customer Feedback System

## 🇬🇧 English

An end-to-end Natural Language Processing (NLP) project for analyzing French customer feedback using both classical machine learning (TF-IDF) and modern deep learning (CamemBERT).

The project is designed as a complete system including:
- Data preprocessing
- Sentiment classification model
- Explainability layer
- REST API deployment using FastAPI
- Model comparison (TF-IDF vs CamemBERT)

---

## 📌 Project Objectives

- Build a sentiment analysis system for French text
- Compare classical NLP (TF-IDF + Logistic Regression) with transformer-based models
- Provide interpretability of predictions
- Deploy the model as a REST API
- Prepare a scalable architecture for production use

---

## 📊 Dataset

The model is trained on the **Allociné dataset**, a French movie review dataset:

- Positive / Negative sentiment labels
- Real-world natural language reviews

---

## ⚙️ Models Used

### 1. TF-IDF + Logistic Regression (Baseline)
- Simple and fast model
- High interpretability
- Strong performance on structured text

### 2. CamemBERT
- Transformer-based model for French language
- Better understanding of context and informal language

---

## 📈 Results (TF-IDF)

- Accuracy: ~92%
- Strong lexical sentiment detection
- Limitations with informal expressions and context

---

## 📊 Final Evaluation on Allociné Dataset

### 🤖 CamemBERT (Fine-tuned)

- Accuracy: **0.9749**
- F1-score: **0.9739**

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Negative (0) | 0.98 | 0.97 | 0.98 |
| Positive (1) | 0.97 | 0.98 | 0.97 |

---

### 📊 TF-IDF + Logistic Regression

- Accuracy: **0.9366**
- F1-score: **0.9341**

---

## ⚖️ Model Comparison

| Model | Accuracy | F1-score | Strength |
|------|----------|----------|---------|
| TF-IDF + Logistic Regression | 93.66% | 93.41% | Fast, interpretable |
| CamemBERT (fine-tuned) | 97.49% | 97.39% | Strong contextual understanding |

---

## 🧠 Key Insights

- CamemBERT significantly outperforms TF-IDF on French sentiment analysis.
- Transformer model handles negation and context much better.
- TF-IDF remains efficient for lightweight production systems.
- Trade-off: performance vs computational cost.

---

## 🚀 API Features (FastAPI)

The project includes a REST API with the following endpoints:

### 🔹 Predict sentiment

POST /predict


Input:

text: string


Output:
```json
{
  "text": "le film était moche",
  "sentiment": 0
}

🔹 Explain model predictions
GET /keywords

Returns most influential words:

{
  "positive": ["excellent", "superbe", "génial"],
  "negative": ["mauvais", "nul", "ennuyeux"]
}
```

Key Features
Text preprocessing (cleaning, normalization)
TF-IDF vectorization with n-grams
Logistic Regression classifier
Model persistence using joblib
REST API using FastAPI
Explainability through top features

Project Structure
project/
│
├── api/
│   └── main.py
├── src/
│   ├── preprocessing.py
│   ├── tfidf_model.py
│
├── models/
│   └── tfidf_model.joblib
│
├── train.py
├── predict.py
└── requirements.txt


▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train the model
python train.py
3. Start the API
uvicorn api.main:app --reload

## 🇫🇷 Version française

# 🇫🇷 Système d'analyse de sentiments pour les avis clients français

Projet de traitement automatique du langage naturel (NLP) permettant d’analyser des avis clients en français à l’aide de modèles de machine learning classiques (TF-IDF) et de modèles avancés basés sur les transformers (CamemBERT - extension future).

Le projet est conçu comme une application complète incluant :
- Prétraitement des données
- Classification de sentiments
- Interprétabilité des résultats
- Déploiement via API REST (FastAPI)
- Comparaison de modèles

---

## 📌 Objectifs du projet

- Analyser automatiquement le sentiment des avis clients en français
- Comparer une approche classique (TF-IDF) avec une approche deep learning
- Rendre les prédictions interprétables
- Déployer le modèle sous forme d’API
- Préparer une architecture évolutive vers la production

---

## 📊 Jeu de données

Le modèle est entraîné sur le dataset **Allociné**, composé d’avis de films en français :

- Avis positifs et négatifs
- Données textuelles réelles

---

## ⚙️ Modèles utilisés

### 1. TF-IDF + Régression logistique (baseline)
- Modèle simple et rapide
- Très interprétable
- Bonnes performances sur texte structuré

### 2. CamemBERT 
- Modèle transformer pour le français
- Meilleure compréhension du contexte et du langage informel

---

## 📈 Résultats (TF-IDF) 

- Accuracy : ~92%
- Bonne détection lexicale des sentiments
- Limites sur les expressions informelles et le contexte

---

## 🚀 API (FastAPI)

Le projet expose une API REST avec les endpoints suivants :

### 🔹 Prédiction de sentiment

POST /predict


Entrée :

text : string


Sortie :
```json
{
  "text": "le film était moche",
  "sentiment": 0
}
🔹 Explicabilité du modèle
GET /keywords

Retourne les mots les plus influents :

{
  "positive": ["excellent", "superbe", "génial"],
  "negative": ["mauvais", "nul", "ennuyeux"]
}
```
🧠 Fonctionnalités principales
- Nettoyage et prétraitement du texte
- Vectorisation TF-IDF avec n-grammes
- Classification avec régression logistique
- Sauvegarde du modèle (joblib)
- API REST avec FastAPI
- Interprétabilité via les mots les plus importants

🏗️ Structure du projet
project/
│
├── api/
│   └── main.py
├── src/
│   ├── preprocessing.py
│   ├── tfidf_model.py
│
├── models/
│   └── tfidf_model.joblib
│
├── train.py
├── predict.py
└── requirements.txt

▶️ Lancer le projet

1. Installer les dépendances
pip install -r requirements.txt
2. Entraîner le modèle
python train.py
3. Lancer l’API
uvicorn api.main:app --reload
