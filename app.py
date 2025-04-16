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
    model_file = "sentiment_model.pkl"
    
    # 1. Download model if needed
    if not os.path.exists(model_file):
        try:
            with st.spinner("Downloading your custom model..."):
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
        # Force CPU loading
        device = torch.device('cpu')
        
        # Load with both safety measures
        model = torch.load(
            model_file,
            map_location=device,
            weights_only=False  # Required for custom models
        )
        
        # Handle DataParallel if used
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        
        # Load tokenizer (use same one your model was trained with)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Please try:
        1. Re-saving your model with:
           torch.save(model.to('cpu').state_dict(), 'model_weights.pth')
        2. Using compatible PyTorch versions
        """)
        return None, None, None

# Load model
model, tokenizer, class_names = load_model()

if model is None:
    st.error("""
    ❌ System cannot start without a model.
    
    Possible solutions:
    1. Check your internet connection
    2. Verify the model file exists at:
       https://drive.google.com/file/d/1pKFpU56YyLloC5IONDMxih5QMQSew54B
    3. Contact support with the error message above
    """)
    st.stop()

# Main app
review = st.text_area("Enter your movie review:", height=150,
                     placeholder="The movie was surprisingly good...")

if st.button("Analyze Sentiment", type="primary") and review:
    with st.spinner("Analyzing..."):
        try:
            # Tokenize input
            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process results
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            
            # Display results
            st.subheader("Results")
            
            # Sentiment display
            col1, col2 = st.columns(2)
            with col1:
                sentiment = class_names[pred_class]
                emoji = "😊" if sentiment == "POSITIVE" else \
                       "😐" if sentiment == "NEUTRAL" else "😞"
                st.metric("Sentiment", f"{sentiment} {emoji}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Confidence visualization
            st.progress(confidence)
            
            # Detailed scores
            with st.expander("Detailed Scores"):
                for i, class_name in enumerate(class_names):
                    st.metric(f"{class_name} Score", f"{probs[i].item():.1%}")
                    
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Model information
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    **Model Type:** Custom 3-class Sentiment Analysis
    **Classes:** Negative, Neutral, Positive
    **Source:** Your Google Drive
    **File:** sentiment_model.pkl
    """)

# Sample inputs
with st.expander("💡 Try These Examples"):
    st.markdown("""
    **Positive:**  
    "This film was a masterpiece! The acting was superb and the story was captivating from start to finish."
    
    **Neutral:**  
    "It was okay - some parts were interesting but others dragged on too long."
    
    **Negative:**  
    "I hated this movie. The plot made no sense and the characters were annoying."
    """)
