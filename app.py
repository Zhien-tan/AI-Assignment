import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer

# Setup
st.set_page_config(page_title="3-Way Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Classifies reviews as **Positive** 😊, **Neutral** 😐, or **Negative** 😞")

@st.cache_resource
def load_model():
    model_file = "sentiment_model.pth"  # Changed to .pth for clarity
    
    # 1. Download model if needed
    if not os.path.exists(model_file):
        try:
            with st.spinner("Downloading model..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                    model_file,
                    quiet=True
                )
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            return None, None, None

    # 2. Load model with proper CPU mapping
    try:
        # Initialize model architecture
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(
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
           torch.save(model.to('cpu').state_dict(), 'model_weights.pth')
        2. Using matching PyTorch versions
        """)
        return None, None, None

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

# Instructions for fixing
with st.expander("🔧 How to Fix Model Loading"):
    st.markdown("""
    1. **Re-save your model** using this code:
    ```python
    # Save just the weights (recommended)
    torch.save(model.to('cpu').state_dict(), 'sentiment_model.pth')
    
    # Or save the complete model (alternative)
    torch.save(model.to('cpu'), 'sentiment_model.pth')
    ```
    
    2. **Upload the new file** to Google Drive
    
    3. **Update the file ID** in this script
    
    4. **Verify PyTorch versions** match between saving and loading
    """)
