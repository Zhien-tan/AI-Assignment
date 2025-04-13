import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis App")
st.write("Analyze movie review sentiment (Positive/Negative)")

# User input
user_input = st.text_area("Enter your movie review here:")

@st.cache_resource
def download_and_load_model():
    # Download model from Google Drive
    file_id = "1pKFpU56YyLloC5IONDMxih5QMQSew54B"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "sentiment_model.pth"  # Changed extension to .pth for clarity
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    try:
        # Load with explicit CPU mapping and weights_only=False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(output, 
                         map_location=device,
                         weights_only=False)  # Important fix for PyTorch 2.6+
        
        # If model is wrapped in DataParallel, extract the actual model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    # Load tokenizer that matches your model
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
        
        # Move to same device as model
        inputs = {k:v.to(next(model.parameters()).device) for k,v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_label = "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE"
        confidence = probs.max().item()
        
        return {"label": pred_label, "score": confidence}
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

if st.button("Analyze Sentiment") and user_input:
    with st.spinner("Analyzing sentiment..."):
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            prediction = predict_sentiment(model, tokenizer, user_input)
            
            if prediction:
                st.subheader("Result:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", prediction['label'])
                with col2:
                    st.metric("Confidence", f"{prediction['score']:.2%}")
                
                if prediction['label'] == "POSITIVE":
                    st.success("😊 Positive review detected!")
                else:
                    st.warning("😞 Negative review detected")

st.markdown("""
**Note:** 
- The model is loaded with `weights_only=False` as this is a trusted source
- For production, consider saving your model with `torch.save(model.state_dict())` instead
""")
