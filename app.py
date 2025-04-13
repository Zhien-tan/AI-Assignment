import streamlit as st
import torch
from transformers import pipeline
import gdown
import joblib
import os

# Step 1: Set up Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis Model", page_icon=":guardsman:", layout="centered")

# Step 2: Display a title and description
st.title("Sentiment Analysis App")
st.write("""
    This is a simple sentiment analysis model to analyze the sentiment of a movie review.
    You can enter a review below, and it will predict whether the sentiment is positive or negative.
""")

# Step 3: Provide the user an option to input text
user_input = st.text_area("Enter your movie review here:")

# Step 4: Define the function to download and load the model
def download_and_load_model():
    # Define the model URL
    model_url = "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B"
    model_file = "sentiment_model.pkl"
    
    # Check if the model is already downloaded
    if not os.path.exists(model_file):
        # Download the model from Google Drive
        gdown.download(model_url, model_file, quiet=False)
    
    # Load the model using joblib (for saved scikit-learn model)
    model = joblib.load(model_file)
    return model

# Step 5: Define function for sentiment prediction
def predict_sentiment(model, review_text):
    # Use transformers pipeline for sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(review_text)
    return result

# Step 6: When the user enters text, run sentiment analysis
if user_input:
    # Load the model
    model = download_and_load_model()
    
    # Get the sentiment prediction
    prediction = predict_sentiment(model, user_input)
    
    # Display the prediction result
    st.write(f"Prediction: {prediction[0]['label']} with a confidence score of {prediction[0]['score']:.2f}")

# Step 7: Display additional instructions
st.write("""
    **How to Use the App:**
    - Enter a review in the text box above.
    - The app will analyze the sentiment of your review and display the result as either Positive or Negative.
    - The confidence score is also displayed.
""")
