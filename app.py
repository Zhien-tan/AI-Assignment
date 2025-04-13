import streamlit as st
import gdown
import joblib
from transformers import AutoTokenizer

# Download the model file from Google Drive using gdown
model_url = 'https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B'
gdown.download(model_url, 'sentiment_model.pkl', quiet=False)

# Load the model and tokenizer
model = joblib.load('sentiment_model.pkl')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Replace with your tokenizer if needed

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax(dim=1).item()
    return prediction

# Streamlit user interface
st.title('Sentiment Analysis with BERT')
user_input = st.text_area("Enter Review Text:")

if user_input:
    result = predict_sentiment(user_input)
    sentiment = ["Negative", "Neutral", "Positive"]
    st.write(f"Sentiment: {sentiment[result]}")
