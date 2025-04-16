import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment (Positive/Negative)")

# User input
user_input = st.text_area("Review Text:", height=150, placeholder="Type your movie review here...")

@st.cache_resource
def download_model():
    try:
        file_id = "19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "sentiment_model.pkl"
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (250MB, may take a few minutes)..."):
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

        # Force CPU loading
        device = torch.device('cpu')
        
        # Load with explicit CPU mapping and safety settings
        if torch.__version__ >= "2.6.0":
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=False  # Required for custom models
            )
        else:
            state_dict = torch.load(
                model_path,
                map_location=device
            )
        
        # Initialize fresh model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        
        # Load state dict with strict=False for better compatibility
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Try these fixes:
        1. Delete the file 'sentiment_model.pth' and refresh the app
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
            st.warning("Please enter some text to analyze")
            return None
            
        # Tokenize with explicit CPU handling
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
        
        # Process results
        probs = torch.softmax(outputs.logits, dim=1)[0]
        return {
            "label": "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE",
            "score": probs.max().item(),
            "pos_score": probs[1].item(),
            "neg_score": probs[0].item()
        }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

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
       - Delete any existing `sentiment_model.pth` file
       - Refresh the page to restart download
       - Ensure you have stable internet connection
    
    2. **Slow performance**:
       - The app requires ~500MB RAM
       - First run will be slow as models download
    
    3. **Error messages**:
       - CUDA errors: This app runs on CPU only
       - Tokenizer errors: Try refreshing the page
    """)

st.caption("Note: This app uses a 250MB machine learning model. Initial loading may take time.")
