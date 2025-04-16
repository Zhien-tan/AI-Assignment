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
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            return None
            
        # Enhanced negation handling
        text = text.lower()
        negation_phrases = {
            "no bad": "good",
            "not bad": "good",
            "wasn't bad": "was good",
            "isn't bad": "is good",
            "no good": "bad",
            "not good": "bad"
        }
        
        for phrase, replacement in negation_phrases.items():
            text = text.replace(phrase, replacement)
        
        # Tokenize input
        inputs = tokenizer(text, 
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results for 3 classes (negative, neutral, positive)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        
        # Get predicted class
        pred_class = torch.argmax(probs).item()
        
        # Define labels based on your model's class mapping
        class_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        pred_label = class_labels[pred_class]
        confidence = probs[pred_class].item()
        
        return {
            "label": pred_label,
            "score": confidence,
            "scores": {
                "negative": probs[0].item(),
                "neutral": probs[1].item(),
                "positive": probs[2].item()
            }
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
                emoji_mapping = {
                    "POSITIVE": "😊",
                    "NEUTRAL": "😐",
                    "NEGATIVE": "😞"
                }
                emoji = emoji_mapping.get(result['label'], "🤔")
                
                # Color mapping
                color_mapping = {
                    "POSITIVE": "green",
                    "NEUTRAL": "blue",
                    "NEGATIVE": "red"
                }
                color = color_mapping.get(result['label'], "gray")
                
                st.markdown(f"### <span style='color:{color}'>{emoji} {result['label']}</span>", 
                           unsafe_allow_html=True)
                
                # Confidence meter
                st.progress(result['score'])
                st.caption(f"Confidence: {result['score']:.1%}")
                
                # Detailed scores
                with st.expander("Detailed Scores"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Positive", f"{result['scores']['positive']:.1%}")
                    col2.metric("Neutral", f"{result['scores']['neutral']:.1%}")
                    col3.metric("Negative", f"{result['scores']['negative']:.1%}")
                    
                    # Add explanation for negation handling if detected
                    if any(phrase in user_input.lower() for phrase in ["no bad", "not bad"]):
                        st.info("ℹ️ Note: Phrases like 'no bad' were interpreted as positive sentiment")

# Add tips
st.markdown("""
**💡 Tips for better results:**
- Write at least 2-3 sentences
- Clearly express your opinion
- For neutral reviews, use balanced language
- Include emotional words for stronger sentiment detection

**Examples:**
- Positive: "The acting was superb and the plot kept me engaged"
- Neutral: "The movie was okay, nothing special but not terrible"
- Negative: "I disliked the characters and found the story boring"
""")
