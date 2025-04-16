import streamlit as st
import torch
import gdown
import os
import pickle
from transformers import AutoTokenizer

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment (Positive/Negative)")

# User input
user_input = st.text_area("Review Text:", height=150, placeholder="Type your movie review here...")

@st.cache_resource
def download_model():
    try:
        # Updated Google Drive link for .pkl file
        url = "https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
        output = "sentiment_model.pkl"
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (this may take a few minutes)..."):
                gdown.download(url, output, quiet=False)
        return output
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return None

@st.cache_resource
def load_sentiment_model():
    try:
        model_path = download_model()
        if not model_path:
            return None

        # Load the pickle file with CPU-only handling
        with open(model_path, 'rb') as f:
            if torch.__version__ >= "2.6.0":
                model = pickle.load(f)
            else:
                model = pickle.load(f)
        
        # Ensure model is on CPU
        if hasattr(model, 'to'):
            model.to('cpu')
        if hasattr(model, 'eval'):
            model.eval()
            
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Try these fixes:
        1. Delete the file 'sentiment_model.pkl' and refresh the app
        2. Check if you have at least 500MB free disk space
        3. Restart the application
        """)
        return None

@st.cache_resource
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        st.error(f"❌ Tokenizer failed to load: {str(e)}")
        return None

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            return None
            
        # Tokenize input
        inputs = tokenizer(text, 
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_label = "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE"
        confidence = probs.max().item()
        
        return {
            "label": pred_label,
            "score": confidence,
            "pos_score": probs[1].item(),
            "neg_score": probs[0].item()
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

if st.button("Analyze Sentiment", type="primary") and user_input:
    with st.spinner("🔍 Analyzing review..."):
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                # Display results
                st.subheader("Analysis Results")
                
                # Sentiment label with emoji
                emoji = "😊" if result['label'] == "POSITIVE" else "😞"
                st.markdown(f"### {emoji} {result['label']}")
                
                # Confidence meter
                st.progress(result['score'])
                st.caption(f"Confidence: {result['score']:.1%}")
                
                # Detailed scores
                with st.expander("Detailed Scores"):
                    col1, col2 = st.columns(2)
                    col1.metric("Positive", f"{result['pos_score']:.1%}")
                    col2.metric("Negative", f"{result['neg_score']:.1%}")

# Main execution
if st.button("Analyze Sentiment", type="primary"):
    if not user_input:
        st.warning("Please enter a review first")
    else:
        with st.spinner("Setting up analysis..."):
            model = load_sentiment_model()
            tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                st.subheader("Results")
                if result['label'] == "POSITIVE":
                    st.success(f"😊 Positive ({result['score']:.0%} confidence)")
                else:
                    st.error(f"😞 Negative ({result['score']:.0%} confidence)")
                
                st.progress(result['score'])
                
                with st.expander("Detailed scores"):
                    col1, col2 = st.columns(2)
                    col1.metric("Positive", f"{result['pos_score']:.1%}")
                    col2.metric("Negative", f"{result['neg_score']:.1%}")

# Troubleshooting section
with st.expander("⚠️ Troubleshooting"):
    st.markdown("""
    **Common issues and solutions:**
    
    1. **Model fails to load**:
       - Delete any existing `sentiment_model.pkl` file
       - Refresh the page to restart download
       - Ensure you have stable internet connection
    
    2. **Pickle errors**:
       - The app requires Python 3.8+
       - Make sure all dependencies are installed
    
    3. **Error messages**:
       - If you see "unsupported pickle", the file may be corrupted
       - Try re-downloading the model file
    """)

st.caption("Note: This app uses machine learning models. Initial loading may take time.")
