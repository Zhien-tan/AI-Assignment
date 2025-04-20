import streamlit as st
import gdown
import torch
import pandas as pd
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Set up event loop policy (important for Windows users)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis App")

# Download model and tokenizer from Google Drive (if needed)
@st.cache_resource
def download_model():
    model_path = "models/sentiment_model"
    gdown.download_folder(
        id="1pDgFlIuzYnoN1rBY74-KGyDBA5cFo1HD",  # Replace with your folder ID
        output=model_path,
        quiet=False,
        use_cookies=False
    )
    return model_path

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = download_model()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Sentiment labels
labels = ['negative', 'neutral', 'positive']

# Sentiment analysis function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    sentiment_scores = {label: float(prob) for label, prob in zip(labels, probs)}
    return sentiment_scores

# Streamlit UI
user_input = st.text_area("Enter text to analyze sentiment:", height=150)

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            scores = analyze_sentiment(user_input)
            st.subheader("Sentiment Scores:")
            st.write(scores)
    else:
        st.warning("Please enter some text before clicking Analyze.")
