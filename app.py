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
def load_resources():
    custom_model_path = "sentiment_model.pkl"
    try:
        # Download model if not present
        if not os.path.exists(custom_model_path):
            with st.spinner("Downloading custom model..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                    custom_model_path,
                    quiet=True
                )
        
        # Load model from file
        device = torch.device('cpu')
        model = torch.load(custom_model_path, map_location=device, weights_only=False)
        
        # Remove DataParallel if present
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        st.success("✅ Custom model loaded successfully!")
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None, None, None

# Load model and tokenizer
model, tokenizer, class_names = load_resources()

# Input box
review = st.text_area("Enter your movie review:", height=150, 
                     placeholder="The movie was okay... not great but not terrible either")

if st.button("Analyze Sentiment", type="primary") and review:
    if model is None:
        st.error("System not ready - failed to load model")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Tokenize and predict
                inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                
                # Display result
                st.subheader("Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    if class_names[pred_class] == "POSITIVE":
                        st.metric("Sentiment", "POSITIVE", delta="😊", delta_color="normal")
                    elif class_names[pred_class] == "NEUTRAL":
                        st.metric("Sentiment", "NEUTRAL", delta="😐", delta_color="off")
                    else:
                        st.metric("Sentiment", "NEGATIVE", delta="😞", delta_color="inverse")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                st.progress(confidence)
                
                # Detailed scores
                with st.expander("Detailed Scores"):
                    for i, class_name in enumerate(class_names):
                        st.metric(f"{class_name} Score", f"{probs[i].item():.1%}")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Info section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    **Model Info:**
    - 3-way sentiment classification (Positive / Neutral / Negative)
    - Uses a custom PyTorch model loaded from Google Drive

    **Deployment URL:**  
    [https://ibtsamfft3vou2mqnd9zz5.streamlit.app/](https://ibtsamfft3vou2mqnd9zz5.streamlit.app/)

    **Install Requirements:**
    ```bash
    pip install torch transformers streamlit gdown
    ```
    """)

# Sample inputs
with st.expander("💡 Try These Examples"):
    st.markdown("""
    **Positive:**  
    "This movie blew me away! The acting was phenomenal and the story kept me engaged from start to finish."
    
    **Neutral:**  
    "It was okay. Not as good as the original but had some entertaining moments."
    
    **Negative:**  
    "Terrible film. The plot made no sense and the acting was wooden. Would not recommend."
    """)
