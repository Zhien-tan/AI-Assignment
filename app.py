import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer

# Setup
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model_and_tokenizer():
    # Download model file
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        with st.spinner("Downloading model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                model_file,
                quiet=False
            )
    
    try:
        # Load model with explicit CPU mapping and weights_only=False
        device = torch.device('cpu')
        model = torch.load(
            model_file,
            map_location=device,
            weights_only=False  # Required for PyTorch 2.6+
        )
        
        # Handle DataParallel if used
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Common solutions:
        1. Make sure the file is not corrupted
        2. Try re-saving your model with:
           torch.save(model.state_dict(), 'model_weights.pth')
        3. Check PyTorch version compatibility
        """)
        return None, None

# Load resources
model, tokenizer = load_model_and_tokenizer()

# User interface
review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment", type="primary") and review:
    if model is None:
        st.error("Model failed to load. Please check the error above.")
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
                    pos.metric("Positive Score", f"{probs[1].item():.1%}")
                    neg.metric("Negative Score", f"{probs[0].item():.1%}")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Add documentation
st.markdown("""
---
### ℹ️ About This App
- Model: DistilBERT fine-tuned on SST-2
- File: [model.pkl](https://drive.google.com/file/d/1pKFpU56YyLloC5IONDMxih5QMQSew54B/view)
- App URL: [https://ibtsamfft3vou2mqnd9zz5.streamlit.app/](https://ibtsamfft3vou2mqnd9zz5.streamlit.app/)
""")
