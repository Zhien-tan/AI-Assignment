import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup
st.set_page_config(page_title="3-Way Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Classifies reviews as **Positive** 😊, **Neutral** 😐, or **Negative** 😞")

@st.cache_resource
def load_custom_model():
    model_file = "sentiment_model.pkl"
    try:
        # Download model from Google Drive if it doesn't exist locally
        if not os.path.exists(model_file):
            with st.spinner("Downloading your custom model..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",  # Update with your Google Drive link
                    model_file,
                    quiet=True
                )

        # Load model to CPU (map_location=torch.device('cpu') to avoid CUDA issues)
        device = torch.device('cpu')
        model = torch.load(model_file, map_location=device, weights_only=False)
        
        # If the model is wrapped in DataParallel, extract the model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()  # Set the model to evaluation mode
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Or any tokenizer that works with your model
        return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    except Exception as e:
        st.warning(f"Custom model loading failed: {str(e)}")
        return None, None, None

# Main app interface
model, tokenizer, class_names = load_custom_model()

if model is None:
    st.error("""
    ❌ No model available. Please:
    1. Check your internet connection
    2. Verify the model file exists on Google Drive
    3. Contact support if this persists
    """)
    st.stop()

review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment", type="primary") and review:
    with st.spinner("Analyzing..."):
        try:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            
            # Display results
            st.subheader("Analysis Results")
            
            # Sentiment indicator
            sentiment_emoji = {
                "POSITIVE": "😊", 
                "NEUTRAL": "😐", 
                "NEGATIVE": "😞"
            }[class_names[pred_class]]
            
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", f"{class_names[pred_class]} {sentiment_emoji}")
            col2.metric("Confidence", f"{confidence:.1%}")
            
            # Confidence visualization
            st.progress(confidence)
            
            # Detailed scores
            with st.expander("Detailed Scores"):
                for i, class_name in enumerate(class_names):
                    st.metric(f"{class_name} Score", f"{probs[i].item():.1%}")
                    
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Add model information
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    **Current Model:** {'Custom Model' if 'custom' in str(model) else 'Fallback Model'}
    
    **Class Labels:** {', '.join(class_names)}
    
    **Model Source:** {'Your Google Drive' if 'custom' in str(model) else 'Hugging Face'}
    """)

# Add sample inputs
with st.expander("💡 Try These Examples"):
    st.markdown(""" 
    **Positive:**  
    "The film was absolutely breathtaking! The cinematography and acting were award-worthy."
    
    **Neutral:**  
    "It was an average movie. Some parts were good, others were boring."
    
    **Negative:**  
    "I hated everything about this film. The plot was nonsensical and the acting was terrible."
    """)
