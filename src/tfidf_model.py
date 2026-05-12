from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TfidfSentimentModel:
    def __init__(self, max_features=10000):
        # self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def get_top_features(self, n=10):
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]

        top_pos = coefs.argsort()[-n:]
        top_neg = coefs.argsort()[:n]

        return {
            "positive": [feature_names[i] for i in reversed(top_pos)],
            "negative": [feature_names[i] for i in top_neg]
        }