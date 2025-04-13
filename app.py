import streamlit as st
import joblib
import gdown
import torch  # PyTorch library
import pandas as pd

# Title for the web app
st.title("Sentiment Analysis Model")

# Add a text area for the user to input a review
user_review = st.text_area("Enter a review:")

# If the user inputs a review, perform the prediction
if user_review:
    # Download the model from Google Drive
    model_url = 'https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B'  # Update with your actual model file ID
    model_file = 'sentiment_model.pkl'
    
    # Use gdown to download the model
    gdown.download(model_url, model_file, quiet=False)

    # Load the pre-trained model with map_location to ensure it loads on CPU
    model = joblib.load(model_file, map_location=torch.device('cpu'))

    # Assuming you have a tokenizer saved or defined elsewhere, for example, using Hugging Face's tokenizer
    # If your model needs a tokenizer, make sure to load it similarly
    tokenizer = joblib.load('tokenizer.pkl')  # Load tokenizer if you have one saved

    # Preprocessing the user's input
    inputs = tokenizer(user_review, return_tensors='pt')

    # Perform prediction
    prediction = model.predict(inputs['input_ids'])

    # Display the result
    if prediction == 0:
        st.write("Sentiment: Negative")
    elif prediction == 1:
        st.write("Sentiment: Neutral")
    else:
        st.write("Sentiment: Positive")
