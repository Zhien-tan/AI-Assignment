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
        with st.spinner("Downloading model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                model_file,
                quiet=False
            )
    
    # Force CPU loading
    device = torch.device('cpu')
    
    try:
        # Load with weights_only=False for trusted models
        model = torch.load(
            model_file,
            map_location=device,
            weights_only=False  # Critical fix for PyTorch 2.6+
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
        ❌ Failed to load model: {str(e)}
        
        If this persists, try:
        1. Re-saving your model with: torch.save(model.state_dict(), 'model_weights.pth')
        2. Loading with model.load_state_dict() instead
        """)
        return None, None

# Load resources
model, tokenizer = load_model_and_tokenizer()

# User input
review = st.text_area("Enter your movie review:", height=150, placeholder="The acting was superb and the story was captivating...")

if st.button("Analyze Sentiment", type="primary") and review:
    if model is None:
        st.error("Please fix the model loading error above first.")
    else:
        with st.spinner("Analyzing sentiment..."):
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
                st.subheader("Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", f"{sentiment} {'😊' if sentiment == 'POSITIVE' else '😞'}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Confidence visualization
                st.progress(confidence)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Add safety disclaimer
st.markdown("""
---
⚠️ **Security Note**:  
This app uses `weights_only=False` to load the model because:
- You are the original creator of this model
- The model file comes from your trusted Google Drive
- For untrusted models, always use `weights_only=True`
""")
