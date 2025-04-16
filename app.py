import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Setup
st.set_page_config(page_title="3-Way Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Classifies reviews as **Positive** 😊, **Neutral** 😐, or **Negative** 😞")

@st.cache_resource
def load_model():
    model_file = "sentiment_model.pkl"
    
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
        # Force CPU loading with proper settings
        device = torch.device('cpu')
        model = torch.load(
            model_file,
            map_location=device,
            weights_only=False,
            pickle_module=torch.serialization.pickle
        )
        
        # Handle DataParallel if used
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Please try:
        1. Re-saving your model with:
           torch.save(model.to('cpu').state_dict(), 'model_weights.pth')
        2. Using PyTorch version 2.0.1
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

# Main app
review = st.text_area("Enter your movie review:", height=150)

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
    
    # Save with custom pickle protocol
    with open('sentiment_model.pkl', 'wb') as f:
        torch.save(model.to('cpu'), f, pickle_protocol=4)
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
