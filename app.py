import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model_and_tokenizer():
    # Download model
    model_file = "sentiment_model.pth"
    if not os.path.exists(model_file):
        gdown.download(
            "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
            model_file,
            quiet=False
        )
    
    # Force CPU loading
    device = torch.device('cpu')
    
    try:
        # Load model with explicit CPU mapping
        model = torch.load(model_file, map_location=device)
        
        # Handle DataParallel if used
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None, None

# Load resources
model, tokenizer = load_model_and_tokenizer()

# User input
review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment") and review:
    if model is None:
        st.error("Model failed to load. Please check the error message.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Tokenize input
                inputs = tokenizer(
                    review,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get probabilities
                probs = torch.softmax(outputs.logits, dim=1)[0]
                sentiment = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
                confidence = max(probs).item()
                
                # Display results
                emoji = "😊" if sentiment == "POSITIVE" else "😞"
                st.subheader(f"Result: {sentiment} {emoji}")
                st.progress(confidence)
                st.write(f"Confidence: {confidence:.1%}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
