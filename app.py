import streamlit as st
import torch
import gdown
import joblib
import os
from transformers import pipeline

# Set up Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis Model", page_icon=":guardsman:", layout="centered")

# Display title and description
st.title("Sentiment Analysis App")
st.write("""
    This app analyzes the sentiment of movie reviews.
    Enter a review below to get a positive/negative prediction.
""")

# User input
user_input = st.text_area("Enter your movie review here:")

# Function to download and load the model
@st.cache_resource
def download_and_load_model():
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1pKFpU56YyLloC5IONDMxih5QMQSew54B"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "sentiment_model.pkl"
    
    # Download if not exists
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    # Load the model with proper device handling
    try:
        # First try loading with default settings
        model = joblib.load(output)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function for prediction
def predict_sentiment(model, text):
    try:
        # Check if the model is a PyTorch model
        if isinstance(model, torch.nn.Module):
            # Preprocess text and make prediction
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).tolist()[0]
            
            # Return prediction
            if logits[0][0] > logits[0][1]:
                return {"label": "NEGATIVE", "score": probabilities[0]}
            else:
                return {"label": "POSITIVE", "score": probabilities[1]}
        else:
            # Fallback to transformers pipeline
            classifier = pipeline("sentiment-analysis")
            return classifier(text)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# When user submits text
if user_input:
    with st.spinner("Analyzing sentiment..."):
        model = download_and_load_model()
        
        if model is not None:
            prediction = predict_sentiment(model, user_input)
            
            if prediction:
                st.write(f"Prediction: {prediction['label']}")
                st.write(f"Confidence: {prediction['score']:.2f}")
                
                # Visual feedback
                if prediction['label'] == "POSITIVE":
                    st.success("😊 Positive sentiment detected!")
                else:
                    st.error("😞 Negative sentiment detected")

# Instructions
st.markdown("""
**How to use:**
1. Enter a movie review in the text box
2. Wait for the analysis to complete
3. View the results
""")
