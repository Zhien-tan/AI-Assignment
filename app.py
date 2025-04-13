import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model_and_tokenizer():
    # Download model file
    model_file = "sentiment_model.pkl"
    if not os.path.exists(model_file):
        with st.spinner("Downloading model..."):
            try:
                gdown.download(
                    "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                    model_file,
                    quiet=False
                )
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                return None, None
    
    # Attempt 1: Load with proper CPU mapping
    try:
        device = torch.device('cpu')
        model = torch.load(
            model_file,
            map_location=device,
            weights_only=False
        )
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.warning(f"Standard loading failed: {str(e)}")
    
    # Attempt 2: Try alternative loading method
    try:
        from pickle import Unpickler
        with open(model_file, 'rb') as f:
            unpickler = Unpickler(f)
            model = unpickler.load()
            
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model = model.to('cpu')
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Alternative loading failed: {str(e)}")
    
    # Attempt 3: Fallback to pretrained model
    try:
        st.warning("Using fallback pretrained model")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Fallback failed: {str(e)}")
        return None, None

# Load resources
model, tokenizer = load_model_and_tokenizer()

# User interface
review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment", type="primary") and review:
    if model is None or tokenizer is None:
        st.error("System not ready - failed to load model")
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
                
                # Process results
                probs = torch.softmax(outputs.logits, dim=1)[0]
                sentiment = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
                confidence = max(probs).item()
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", sentiment, 
                           delta="😊" if sentiment == "POSITIVE" else "😞")
                col2.metric("Confidence", f"{confidence:.1%}")
                
                # Confidence visualization
                st.progress(confidence)
                
                # Detailed scores
                with st.expander("Detailed Scores"):
                    pos, neg = st.columns(2)
                    pos.metric("Positive", f"{probs[1].item():.1%}")
                    neg.metric("Negative", f"{probs[0].item():.1%}")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Add troubleshooting section
with st.expander("Troubleshooting"):
    st.markdown("""
    **If model fails to load:**
    1. Check your `sentiment_model.pkl` file exists
    2. Verify file integrity (should be ~300-500MB)
    3. Try re-saving your model with:
       ```python
       torch.save(model.to('cpu').state_dict(), 'model_weights.pth')
       ```
    4. Ensure PyTorch versions match between saving/loading
    """)
    
    if os.path.exists("sentiment_model.pkl"):
        st.write(f"Model file size: {os.path.getsize('sentiment_model.pkl')/1e6:.2f} MB")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
