import streamlit as st
import torch
import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# App config
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_resources():
    model_file = "sentiment_model.pt"
    
    # Download model if not exists
    if not os.path.exists(model_file):
        with st.spinner("Downloading model..."):
            try:
                gdown.download(
                    "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                    model_file,
                    quiet=False
                )
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None, None

    # Load custom or fallback model
    try:
        model = torch.load(model_file, map_location=torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.eval()
    except Exception as e:
        st.warning("Custom model failed to load. Falling back to SST-2 (binary only).")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model.eval()
        except Exception as e:
            st.error("All model loading attempts failed.")
            return None, None
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_resources()

# UI
if model and tokenizer:
    review = st.text_area("✍️ Enter your movie review:", height=150)
    
    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            
            # Handle 3-class or 2-class
            if probs.shape[0] == 3:
                labels = ["NEGATIVE 😞", "NEUTRAL 😐", "POSITIVE 😊"]
            else:
                labels = ["NEGATIVE 😞", "POSITIVE 😊"]
            
            pred_index = torch.argmax(probs).item()
            sentiment = labels[pred_index]
            confidence = probs[pred_index].item()

            st.markdown(f"### 🎯 Sentiment: **{sentiment}**")
            st.markdown(f"**Confidence:** `{confidence:.1%}`")
else:
    st.error("Model not available. Please check previous errors or setup instructions.")
