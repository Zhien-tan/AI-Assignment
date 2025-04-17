import streamlit as st
import torch
import gdown
import os
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================
# INITIALIZATION & SETUP
# =============================================

# Fix event loop issues (critical for Streamlit)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# App configuration
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("🎬 Professional Sentiment Analyzer")

# =============================================
# MODEL MANAGEMENT
# =============================================

def download_model():
    """Download model with validation and retries"""
    MODEL_URL = "https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
    MODEL_FILE = "sentiment_model_v2.pth"
    
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1024:
        with st.spinner("Downloading model (300MB)..."):
            try:
                gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
                
                # Validate download
                if os.path.getsize(MODEL_FILE) < 1024*1024:  # At least 1MB
                    os.remove(MODEL_FILE)
                    raise ValueError("Downloaded file too small - likely corrupted")
                    
                return MODEL_FILE
                
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                if os.path.exists(MODEL_FILE):
                    os.remove(MODEL_FILE)
                return None
    return MODEL_FILE

@st.cache_resource
def load_model_components():
    """Load model and tokenizer with comprehensive error handling"""
    try:
        # 1. Download or locate model file
        model_file = download_model()
        if not model_file:
            return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        # 2. Initialize fresh model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3
        )
        
        # 3. Load state dict with version-specific settings
        device = torch.device('cpu')
        load_args = {
            'map_location': device,
            'weights_only': False  # Required for custom models
        } if torch.__version__ >= "2.6.0" else {'map_location': device}
        
        state_dict = torch.load(model_file, **load_args)
        
        # Handle different save formats
        if 'state_dict' in state_dict:  # Properly saved model
            model.load_state_dict(state_dict['state_dict'])
        else:  # Direct state dict
            model.load_state_dict(state_dict)
            
        model.to(device).eval()
        
        # 4. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Required actions:
        1. Delete any existing model files
        2. Re-save your model using:
        ```python
        torch.save({
            'state_dict': model.state_dict(),
            'config': model.config
        }, 'sentiment_model_v2.pth')
        ```
        3. Re-upload to Google Drive
        """)
        return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# =============================================
# MAIN APPLICATION
# =============================================

# Load components
model, tokenizer, CLASS_NAMES = load_model_components()

if model is None:
    st.error("""
    ❌ System cannot start without model
    
    Immediate solutions:
    1. Check your internet connection
    2. Verify model file exists at:
       https://drive.google.com/file/d/19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH
    3. Re-save model using proper method
    """)
    st.stop()

# Example reviews
EXAMPLE_REVIEWS = {
    "🌟 Positive": "The film was an absolute masterpiece! Stellar performances combined with breathtaking cinematography made this an unforgettable experience.",
    "🔄 Neutral": "It was decent - some scenes were engaging while others dragged. The acting was fine but nothing extraordinary.",
    "💢 Negative": "A complete disappointment. The plot was incoherent, the acting wooden, and the pacing glacial. I wanted to walk out."
}

# UI Layout
tab1, tab2 = st.tabs(["Custom Review", "Sample Reviews"])

with tab1:
    user_review = st.text_area("Enter your review:", height=150, key="custom_input")

with tab2:
    selected = st.selectbox("Choose example:", list(EXAMPLE_REVIEWS.keys()))
    if st.button("Load Selected Example"):
        user_review = EXAMPLE_REVIEWS[selected]
        st.session_state.custom_input = user_review  # Update the other tab
    st.text_area("Example Content:", value=EXAMPLE_REVIEWS[selected], height=150, disabled=True)

if st.button("Analyze Sentiment", type="primary"):
    user_review = st.session_state.get("custom_input", "")
    if not user_review.strip():
        st.warning("Please enter a review or select an example")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Tokenization
                inputs = tokenizer(
                    user_review,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                # Prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process results
                probs = torch.softmax(outputs.logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                
                # Display results
                st.subheader("Analysis Results")
                
                # Sentiment indicator
                sentiment = CLASS_NAMES[pred_idx]
                color = "#4CAF50" if sentiment == "POSITIVE" else \
                       "#FFC107" if sentiment == "NEUTRAL" else "#F44336"
                
                st.markdown(
                    f"<h3 style='text-align: center; color: {color};'>"
                    f"{'😊' if sentiment == 'POSITIVE' else '😐' if sentiment == 'NEUTRAL' else '😞'} "
                    f"{sentiment} - {confidence:.1%} Confidence"
                    f"</h3>",
                    unsafe_allow_html=True
                )
                
                # Confidence visualization
                st.progress(confidence)
                
                # Detailed scores
                with st.expander("Detailed Breakdown"):
                    cols = st.columns(3)
                    for i, name in enumerate(CLASS_NAMES):
                        cols[i].metric(
                            name,
                            f"{probs[i].item():.1%}",
                            delta="↑ Strong" if probs[i] > 0.7 else 
                                 "→ Moderate" if probs[i] > 0.3 else "↓ Weak",
                            delta_color="normal"
                        )
                
                # Interpretation
                with st.expander("📊 Interpretation Guide"):
                    st.markdown("""
                    - **>70%**: Strong indication of this sentiment
                    - **30-70%**: Moderate indication
                    - **<30%**: Weak indication
                    """)
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# =============================================
# TECHNICAL INFORMATION
# =============================================

with st.expander("⚙️ System Configuration"):
    st.markdown(f"""
    **Environment Information:**
    - PyTorch Version: `{torch.__version__}`
    - Transformers Version: `{AutoTokenizer.from_pretrained.__module__.split('.')[0]}`
    - Device: `{'CPU' if next(model.parameters()).is_cpu else 'GPU'}`
    
    **Model Information:**
    - Architecture: DistilBERT
    - Classes: {CLASS_NAMES}
    - Input Length: 512 tokens
    """)

with st.expander("💾 Proper Model Saving Instructions"):
    st.markdown("""
    ```python
    # BEST PRACTICE: Save properly configured model
    torch.save({
        'state_dict': model.state_dict(),  # Model weights
        'config': model.config,           # Model configuration
        'class_names': ["NEG", "NEU", "POS"],  # classes
        'tokenizer_config': {             # Tokenizer info
            'name_or_path': 'distilbert-base-uncased'
        }
    }, 'sentiment_model_pro.pth')
    
    # Loading:
    checkpoint = torch.load('sentiment_model_pro.pth', map_location='cpu')
    model = AutoModelForSequenceClassification.from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_config']['name_or_path'])
    ```
    """)

# Security note
st.sidebar.warning("""
⚠️ **Security Notice**  
This app loads model files which could contain executable code.  
Only use models from trusted sources.
""")
