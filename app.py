import streamlit as st
import torch
import gdown
import os
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================
# FIX 1: Event Loop Initialization
# =============================================
try:
    # Ensure event loop exists (critical for Streamlit)
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# =============================================
# Streamlit Setup
# =============================================
st.set_page_config(
    page_title="Sentiment Analysis",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("🎬 Movie Review Sentiment Analyzer")

# =============================================
# Model Loading with Robust Error Handling
# =============================================
@st.cache_resource
def load_model():
    MODEL_FILE = "sentiment_model.pth"
    MODEL_URL = "https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
    
    # 1. Download with validation
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1024:
        try:
            with st.spinner("Downloading model (250MB)..."):
                gdown.download(MODEL_URL, MODEL_FILE, quiet=True)
                if os.path.getsize(MODEL_FILE) < 1024:
                    raise ValueError("File too small - likely corrupted")
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # 2. Safe loading
    try:
        # Initialize fresh model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3
        )
        
        # Load state dict with explicit CPU mapping
        device = torch.device('cpu')
        if torch.__version__ >= "2.6.0":
            state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=False)
        else:
            state_dict = torch.load(MODEL_FILE, map_location=device)
        
        model.load_state_dict(state_dict)
        model.to(device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.error(f"""
        ❌ Model loading error: {str(e)}
        
        Likely causes:
        1. Model file is corrupted - delete '{MODEL_FILE}' and retry
        2. PyTorch version mismatch
        3. File saved incorrectly
        """)
        return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# =============================================
# Main Application
# =============================================
model, tokenizer, CLASS_NAMES = load_model()

if model is None:
    st.error("""
    ❌ Critical Error: Model unavailable
    
    Solutions:
    1. Delete any existing model files
    2. Check internet connection
    3. Verify the Google Drive file is accessible
    """)
    st.stop()

# Example reviews
EXAMPLE_REVIEWS = {
    "Positive": "The film was spectacular! Brilliant performances and a captivating plot...",
    "Neutral": "It was okay - some good moments but nothing memorable...",
    "Negative": "Terrible movie. Poor acting and a nonsensical story..."
}

# UI Layout
tab1, tab2 = st.tabs(["Custom Review", "Examples"])

with tab1:
    user_review = st.text_area("Your Review:", height=150)

with tab2:
    selected = st.selectbox("Sample Reviews:", list(EXAMPLE_REVIEWS.keys()))
    user_review = st.text_area("Example:", 
                             value=EXAMPLE_REVIEWS[selected], 
                             height=150)
    if st.button("Load Example"):
        st.rerun()

if st.button("Analyze", type="primary") and user_review:
    with st.spinner("Processing..."):
        try:
            # Tokenization
            inputs = tokenizer(
                user_review,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # Prediction
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Results processing
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            # Display
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                sentiment = CLASS_NAMES[pred_idx]
                emoji = "😊" if sentiment == "POSITIVE" else \
                       "😐" if sentiment == "NEUTRAL" else "😞"
                st.metric("Sentiment", f"{sentiment} {emoji}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)
            
            with st.expander("Detailed Scores"):
                cols = st.columns(3)
                for i, name in enumerate(CLASS_NAMES):
                    cols[i].metric(name, f"{probs[i].item():.1%}")
                    
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# =============================================
# Debugging Information
# =============================================
with st.expander("🛠️ Technical Info"):
    st.markdown(f"""
    **Environment:**
    - PyTorch: {torch.__version__}
    - Transformers: {AutoTokenizer.from_pretrained.__module__}
    - Streamlit: {st.__version__}
    
    **Model Info:**
    - Classes: {CLASS_NAMES}
    - Device: {'CPU' if next(model.parameters()).is_cpu else 'GPU'}
    """)

with st.expander("💾 Proper Model Saving"):
    st.code("""
    # Recommended method:
    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config
    }, 'model.pth')
    
    # Loading:
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        config=config
    )
    model.load_state_dict(torch.load('model.pth')['state_dict'])
    """)
