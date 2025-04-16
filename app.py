import streamlit as st
import torch
import gdown
import os
import pickle
import asyncio
from transformers import AutoTokenizer

# Fix for Streamlit event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment")

@st.cache_resource
def download_and_load_model():
    try:
        # Download model from Google Drive
        file_id = "19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "sentiment_model.pkl"
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (250MB, may take 2-3 minutes)..."):
                try:
                    gdown.download(url, output, quiet=False)
                except Exception as e:
                    st.error(f"Download failed: {str(e)}")
                    return None

        # Verify file integrity
        if os.path.getsize(output) < 1024:  # Check if file is too small
            os.remove(output)
            raise ValueError("Downloaded file is too small - likely corrupted")

        # Load the pickle file safely
        try:
            with open(output, 'rb') as f:
                model = pickle.load(f)
        except pickle.UnpicklingError:
            os.remove(output)
            raise ValueError("File is corrupted - deleting and please try again")

        # Ensure model is on CPU
        if hasattr(model, 'to'):
            model.to('cpu')
        if hasattr(model, 'eval'):
            model.eval()
            
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Try these steps:
        1. Delete 'sentiment_model.pkl' and refresh
        2. Check internet connection
        3. Restart the app
        4. Contact support if issue persists
        """)
        return None

@st.cache_resource
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        st.error(f"❌ Tokenizer failed: {str(e)}")
        return None

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            return None
            
        # Enhanced text preprocessing
        text = text.lower().strip()
        text = text.replace("no bad", "good").replace("not bad", "good")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        if hasattr(outputs, 'logits'):
            probs = torch.softmax(outputs.logits, dim=1)[0]
        else:
            probs = torch.softmax(outputs[0], dim=1)[0]
            
        return {
            "positive": probs[1].item(),
            "negative": probs[0].item()
        }
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Main app
user_input = st.text_area("Review Text:", height=150)

if st.button("Analyze", type="primary") and user_input:
    with st.spinner("Analyzing..."):
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            scores = predict_sentiment(model, tokenizer, user_input)
            
            if scores:
                # Determine sentiment
                if scores['positive'] > 0.65:
                    label, emoji, color = "POSITIVE", "😊", "green"
                elif scores['negative'] > 0.65:
                    label, emoji, color = "NEGATIVE", "😞", "red"
                else:
                    label, emoji, color = "NEUTRAL", "😐", "blue"
                
                # Display results
                st.markdown(f"### <span style='color:{color}'>{emoji} {label}</span>", unsafe_allow_html=True)
                
                # Show scores
                with st.expander("Details"):
                    st.metric("Positive", f"{scores['positive']:.1%}")
                    st.metric("Negative", f"{scores['negative']:.1%}")

# Troubleshooting
with st.expander("Need Help?"):
    st.markdown("""
    **Solutions for common issues:**
    
    1. **Model won't load**:
       - Delete the .pkl file and refresh
       - Check your internet connection
       - Ensure sufficient disk space
    
    2. **Strange predictions**:
       - Try clearer language
       - Avoid mixed sentiments
    
    3. **Other errors**:
       - Restart the app
       - Check console for details
    """)
