import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        output = "sentiment_model.pkl"
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (this may take a few minutes)..."):
                gdown.download(url, output, quiet=False)

        # Initialize device
        device = torch.device('cpu')
        
        # Load model with explicit CPU mapping
        if torch.__version__ >= "2.6.0":
            model = torch.load(output, 
                             map_location=device,
                             weights_only=False)
        else:
            model = torch.load(output,
                             map_location=device)
        
        # Handle DataParallel if needed
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Try these solutions:
        1. Delete the file 'sentiment_model.pth' and refresh
        2. Ensure you have at least 500MB free space
        3. Check your internet connection
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
            "isn't bad": "is good"
        }
        for phrase, replacement in negation_map.items():
            text = text.replace(phrase, replacement)
        
        # Tokenize input
        inputs = tokenizer(text,
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512)
        
        # Move to CPU explicitly
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results (assuming 3-class output)
        probs = torch.softmax(outputs.logits, dim=1)[0]
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
    with st.spinner("🔍 Analyzing..."):
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                # Display results
                st.subheader("Results")
                
                # Color and emoji mapping
                format_map = {
                    "POSITIVE": {"emoji": "😊", "color": "green"},
                    "NEUTRAL": {"emoji": "😐", "color": "blue"},
                    "NEGATIVE": {"emoji": "😞", "color": "red"}
                }
                fmt = format_map.get(result['label'], {"emoji": "🤔", "color": "gray"})
                
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

# Troubleshooting section
with st.expander("⚠️ Troubleshooting"):
    st.markdown("""
    **If you see errors:**
    1. **Model loading fails**:
       - Delete `sentiment_model.pth` and refresh
       - Check disk space (need 500MB+ free)
    2. **Wrong predictions**:
       - Try more explicit language
       - Avoid mixed sentiments in one review
    """)
