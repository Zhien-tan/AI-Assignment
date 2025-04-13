import streamlit as st
import gdown
import zipfile
import os
from transformers import BertForSequenceClassification, BertTokenizer

# Download the zipped model from Google Drive
model_url = 'https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B'  # Replace with your file ID
gdown.download(model_url, 'sentiment_model.zip', quiet=False)

# Unzip the model
if not os.path.exists('sentiment_model'):
    os.makedirs('sentiment_model')
    
with zipfile.ZipFile('sentiment_model.pkl', 'r') as zip_ref:
    zip_ref.extractall('sentiment_model')

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('sentiment_model')
tokenizer = BertTokenizer.from_pretrained('sentiment_model')

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
