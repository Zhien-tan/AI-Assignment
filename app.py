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

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
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
            st.warning("Please enter some text to analyze")
            return None
            
        # Special handling for negation phrases
        text = text.replace("no bad", "good")  # Simple preprocessing
        text = text.replace("not bad", "good")
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True  # Better context handling
        )
        
        # Debug print to see processed text
        print(f"Processed text: {text}")
        
        inputs = {k: v.to('cpu') for k, v in inputs.items() if k != 'offset_mapping'}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.softmax(logits, dim=1)[0]
        
        # Debug probabilities
        print(f"Debug - Negative: {probs[0].item():.2%}, Positive: {probs[1].item():.2%}")
        
        # Adjusted classification with confidence threshold
        if probs[1] > 0.65:  # Higher threshold for positive
            return {
                "label": "POSITIVE",
                "score": probs[1].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        else:
            return {
                "label": "NEGATIVE",
                "score": probs[0].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None
# Main execution block
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
    
    2. **No negative results**:
       - Try clearly negative reviews like "This was the worst movie ever"
       - Check console for probability outputs
    """)

st.caption("Note: This app uses machine learning models. Initial loading may take time.")
