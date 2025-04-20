import streamlit as st
import torch
import gdown
import os
import pickle
import asyncio
import sys

from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: only set event loop policy on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Paths and URLs
MODEL_PATH = "models/sentiment_model.pt"
TOKENIZER_NAME = "distilbert-base-uncased"
GDOWN_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"

# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        st.info("Downloading model...")
        gdown.download(GDOWN_URL, MODEL_PATH, quiet=False)

# Load model and tokenizer
@st.cache_resource
def load_model():
    download_model()
    model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return model, tokenizer

model, tokenizer = load_model()

# Title
st.title("🧠 Sentiment Analysis App")

# Input
user_input = st.text_area("Enter text to analyze:", height=150)

# Analyze
if st.button("Analyze") and user_input:
    with st.spinner("Analyzing..."):
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        scores = outputs.logits.detach().numpy()[0]
        probs = softmax(scores)

        labels = ["Negative", "Neutral", "Positive"]
        st.subheader("Sentiment Prediction:")
        for label, prob in zip(labels, probs):
            st.write(f"**{label}**: {prob:.2%}")
