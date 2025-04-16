import streamlit as st
import torch
import gdown
import os
import pickle
from transformers import AutoTokenizer

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment (Positive/Neutral/Negative)")

@st.cache_resource
def download_and_load_model():
    try:
        # Download model from Google Drive
        file_id = "1pKFpU56YyLloC5IONDMxih5QMQSew54B"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "sentiment_model.pkl"  # Changed to .pkl
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (this may take a few minutes)..."):
                gdown.download(url, output, quiet=False)

        # Load the pickle file with CPU-only handling
        with open(output, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure model is on CPU and in eval mode
        if hasattr(model, 'to'):
            model.to('cpu')
        if hasattr(model, 'eval'):
            model.eval()
            
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Try these steps:
        1. Delete the file 'sentiment_model.pkl' and refresh
        2. Check you have at least 500MB free disk space
        3. Verify your internet connection
        4. Ensure the file is a valid pickle file
        
        If problems persist, contact support.
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
            
        # Enhanced negation handling
        text = text.lower()
        negation_map = {
            "no bad": "good",
            "not bad": "good",
            "wasn't bad": "was good",
            "isn't bad": "is good",
            "no good": "bad",
            "not good": "bad"
        }
        for phrase, replacement in negation_map.items():
            text = text.replace(phrase, replacement)
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to CPU explicitly
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Predict (handling different model types)
        with torch.no_grad():
            if hasattr(model, '__call__'):
                outputs = model(**inputs)
            elif hasattr(model, 'predict'):
                outputs = model.predict(inputs)
            else:
                raise ValueError("Model doesn't have callable predict method")
        
        # Process results (assuming 3-class output)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]
            
        probs = torch.softmax(logits, dim=1)[0]
        class_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        pred_idx = torch.argmax(probs).item()
        
        return {
            "label": class_names[pred_idx],
            "score": probs[pred_idx].item(),
            "scores": {
                "negative": probs[0].item(),
                "neutral": probs[1].item(),
                "positive": probs[2].item()
            }
        }
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# Main app
user_input = st.text_area("Review Text:", height=150)

if st.button("Analyze Sentiment", type="primary") and user_input:
    with st.spinner("🔍 Analyzing review..."):
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                # Display results
                st.subheader("Analysis Results")
                
                # Formatting based on sentiment
                sentiment_format = {
                    "POSITIVE": {"emoji": "😊", "color": "green"},
                    "NEUTRAL": {"emoji": "😐", "color": "blue"},
                    "NEGATIVE": {"emoji": "😞", "color": "red"}
                }
                fmt = sentiment_format.get(result['label'], {"emoji": "🤔", "color": "gray"})
                
                st.markdown(
                    f"### <span style='color:{fmt['color']}'>{fmt['emoji']} {result['label']}</span> "
                    f"({result['score']:.0%} confidence)",
                    unsafe_allow_html=True
                )
                
                # Confidence meter
                st.progress(result['score'])
                
                # Detailed scores
                with st.expander("📊 Detailed Scores"):
                    cols = st.columns(3)
                    cols[0].metric("Positive", f"{result['scores']['positive']:.1%}")
                    cols[1].metric("Neutral", f"{result['scores']['neutral']:.1%}")
                    cols[2].metric("Negative", f"{result['scores']['negative']:.1%}")

# Security note about pickle files
with st.expander("⚠️ Security Information"):
    st.warning("""
    This app loads a .pkl file which could potentially execute arbitrary code. 
    Only use this app with model files from trusted sources.
    """)

# Troubleshooting section
with st.expander("❓ Need Help?"):
    st.markdown("""
    **Common Solutions:**
    
    1. **Model won't load**:
       - Delete `sentiment_model.pkl` and refresh
       - Check disk space (need 500MB+ free)
       - Ensure stable internet connection
    
    2. **Wrong predictions**:
       - Try more explicit language
       - Avoid mixed sentiments
    
    3. **'no bad' shows negative**:
       - The app automatically converts these to positive
       - If still wrong, your model may need retraining
    """)
