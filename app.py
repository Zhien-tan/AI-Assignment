import streamlit as st
import torch
import gdown
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    model_file = "sentiment_model.pth"  # Changed to .pth for clarity
    
    # 1. Download model if needed
    if not os.path.exists(model_file):
        try:
            with st.spinner("Downloading model..."):
                gdown.download(
                    "https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH",
                    model_file,
                    quiet=True
                )
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # 2. Load model with proper CPU mapping
    try:
        # Initialize model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3  # For 3-class classification
        )
        
        # Load state dict with CPU mapping
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Please try:
        1. Re-saving your model with:
           torch.save(model.to('cpu').state_dict(), 'sentiment_model.pth')
        2. Using matching PyTorch versions
        """)
        return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Load model
model, tokenizer, class_names = load_model()

if model is None:
    st.error("""
    ❌ System cannot start without a model.
    
    Immediate solutions:
    1. Re-save your model using:
    ```python
    torch.save(model.to('cpu').state_dict(), 'sentiment_model.pth')
    ```
    2. Upload the new file to Google Drive
    3. Update the file ID in this code
    """)
    st.stop()

# Example inputs
example_reviews = {
    "Positive": "This movie was absolutely incredible! The acting was superb, the storyline kept me engaged throughout, and the cinematography was breathtaking. One of the best films I've seen this year!",
    "Neutral": "The movie was okay. It had some good moments, but also some slow parts. The acting was decent but nothing special. Worth watching if you have nothing better to do.",
    "Negative": "I was really disappointed with this film. The plot was predictable, the characters were poorly developed, and the dialogue felt forced. The pacing dragged and I couldn't wait for it to end."
}

# Main app
tab1, tab2 = st.tabs(["Enter Review", "Try Examples"])

with tab1:
    review = st.text_area("Enter your movie review:", height=150)

with tab2:
    st.markdown("### Try these example reviews:")
    selected_example = st.selectbox("Choose an example:", list(example_reviews.keys()))
    review = st.text_area("Example review:", 
                         value=example_reviews[selected_example], 
                         height=150)
    if st.button("Use This Example"):
        st.session_state.example_used = True

if st.button("Analyze Sentiment", type="primary") and review:
    with st.spinner("Analyzing..."):
        try:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            
            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment = class_names[pred_class]
                emoji = "😊" if sentiment == "POSITIVE" else \
                       "😐" if sentiment == "NEUTRAL" else "😞"
                st.metric("Sentiment", f"{sentiment} {emoji}")
            
            with col2:
                confidence = probs[pred_class].item()
                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)
            
            # Detailed scores
            with st.expander("Detailed Scores"):
                for i, name in enumerate(class_names):
                    st.metric(name, f"{probs[i].item():.1%}")
                    
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Instructions for model saving
with st.expander("💻 How to Properly Save Your Model"):
    st.markdown("""
    ```python
    # BEST PRACTICE (save only weights)
    torch.save({
        'state_dict': model.to('cpu').state_dict(),
        'class_names': ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    }, 'sentiment_model.pth')
    
    # Alternative method (save complete model)
    torch.save(model.to('cpu'), 'sentiment_model.pth')
    ```
    """)

# Requirements info
with st.expander("📦 Required Packages"):
    st.code("""
    torch==2.0.1
    transformers==4.30.2
    gdown==4.7.1
    streamlit>=1.22.0
    """)
