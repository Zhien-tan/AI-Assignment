import streamlit as st
import torch
import gdown
import os
import pickle
import asyncio
from transformers import AutoTokenizer

# ======================
# STREAMLIT INITIALIZATION (With better error handling)
# ======================
try:
    asyncio.set_event_loop(asyncio.new_event_loop())

    st.set_page_config(
        page_title="Sentiment Analysis",
        layout="centered",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Initialization error: {e}")

# ======================
# MODEL DOWNLOAD & LOADING
# ======================
@st.cache_resource
def get_model():
    MODEL_URL = "https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
    MODEL_PATH = "sentiment_model.pkl"

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
        with st.spinner("Downloading model (250MB)..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                if os.path.getsize(MODEL_PATH) < 1024:
                    raise ValueError("Downloaded file is too small")
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                return None

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        if hasattr(model, 'to'):
            model.to('cpu')
        if hasattr(model, 'eval'):
            model.eval()

        return model

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

# ======================
# TOKENIZER LOADING
# ======================
@st.cache_resource
def get_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        st.error(f"Tokenizer failed: {str(e)}")
        return None

# ======================
# PREDICTION LOGIC
# ======================
def analyze_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            return None



        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to('cpu') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if hasattr(outputs, 'logits'):
            probs = torch.softmax(outputs.logits, dim=1)[0]
        else:
            probs = torch.softmax(outputs[0], dim=1)[0]

        return {
            "negative": probs[0].item(),
            "neutral": probs[1].item(),
            "positive": probs[2].item()
        }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# ======================
# MAIN APP INTERFACE
# ======================
st.title("Review Sentiment Analysis")
st.write("Enter your review below:")

user_input = st.text_area("Review Text:", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review first")
    else:
        with st.spinner("Processing..."):
            model = get_model()
            tokenizer = get_tokenizer()

            if model and tokenizer:
                results = analyze_sentiment(model, tokenizer, user_input)

                if results:
                    if results['positive'] > 0.65:
                        label, emoji, color = "POSITIVE", "😊", "green"
                    elif results['negative'] > 0.65:
                        label, emoji, color = "NEGATIVE", "😞", "red"
                    else:
                        label, emoji, color = "NEUTRAL", "😐", "blue"

                    st.markdown(
                        f"### <span style='color:{color}'>{emoji} {label}</span>",
                        unsafe_allow_html=True
                    )

                    confidence = max(results.values())
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.1%}")

                    with st.expander("Detailed Analysis"):
                        cols = st.columns(3)
                        cols[0].metric("Positive", f"{results['positive']:.1%}")
                        cols[1].metric("Neutral", f"{results['neutral']:.1%}")
                        cols[2].metric("Negative", f"{results['negative']:.1%}")

# ======================
# TROUBLESHOOTING SECTION
# ======================
with st.expander("⚠️ Troubleshooting Help"):
    st.markdown("""
    **Common Issues & Solutions:**

    1. **Model won't load**:
       - Delete `sentiment_model.pkl` and refresh
       - Check internet connection
       - Ensure you have 500MB+ free space

    2. **Strange predictions**:
       - Try more explicit language
       - Avoid mixed sentiments

    3. **App crashes**:
       - Restart the Streamlit server
       - Check console for errors
    """)

# ======================
# SECURITY WARNING
# ======================
st.sidebar.warning("""
⚠️ **Security Notice**  
This app loads pickle files which could execute arbitrary code.  
Only use models from trusted sources.
""") 
