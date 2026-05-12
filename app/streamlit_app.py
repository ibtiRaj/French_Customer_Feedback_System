import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="French Sentiment Analysis", layout="centered")

st.title("🇫🇷 French Customer Feedback Analyzer")
st.write("Enter a review and the model will predict its sentiment.")

# Input
text = st.text_area("Write your review here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Call FastAPI /predict
        response = requests.post(
            f"{API_URL}/predict_all",
            params={"text": text}
        )

        result = response.json()

        st.subheader("Result")

        sentiment_by_TFIDF = result["tfidf_prediction"]
        sentiment_by_camembert = result["camembert_prediction"]


        if sentiment_by_TFIDF == 1:
            if sentiment_by_camembert== 1:
                st.success("😊 Positive sentiment")
            else:
                st.error("Ups, our models desagree about the sentiment" )
        if sentiment_by_TFIDF == 0:
            if sentiment_by_camembert== 0:
                st.error("😠 Negative sentiment")
            else:
                st.error("Ups, our models desagree about the sentiment" )
        st.json(result)

        # Call /keywords
        st.subheader("Model Insights")

        kw_response = requests.get(f"{API_URL}/keywords")
        keywords = kw_response.json()

        st.write("### Positive words")
        st.write(", ".join(keywords["positive"]))

        st.write("### Negative words")
        st.write(", ".join(keywords["negative"]))