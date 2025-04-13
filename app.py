import streamlit as st
import joblib
import gdown
import os

# Title of the app
st.title("Sentiment Analysis Model")

# Define file and URL
file_id = "1pKFpU56YyLloC5IONDMxih5QMQSew54B"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_file = "sentiment_model.pkl"

# Download the model if it doesn't exist
if not os.path.exists(model_file):
    st.info("Downloading sentiment analysis model...")
    gdown.download(model_url, model_file, quiet=False)

# Load the model
try:
    model = joblib.load(model_file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
    st.stop()

# Text input
user_input = st.text_area("Enter a review:")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = model.predict([user_input])[0]
        st.write(f"**Predicted Sentiment:** {prediction}")
