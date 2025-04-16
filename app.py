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
user_input = st.text_area("Review Text:", height=150)

@st.cache_resource
def download_and_load_model():
    # Download model from Google Drive
    file_id = "19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "sentiment_model.pth"
    
    if not os.path.exists(output):
        with st.spinner("Downloading model..."):
            gdown.download(url, output, quiet=False)
    
    try:
        # Force CPU loading regardless of where model was saved
        device = torch.device('cpu')
        
        # Load with proper device mapping and safety settings
        model = torch.load(output, 
                         map_location=device,
                         weights_only=False)
        
        # Handle DataParallel if needed
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

# Add some tips
st.markdown("""
**💡 Tips for better results:**
- Write at least 2-3 sentences
- Avoid neutral language
- Include emotional words
""")
