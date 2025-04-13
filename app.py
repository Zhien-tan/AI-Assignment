import streamlit as st
import torch  # PyTorch library
import gdown
import joblib

# Title for the web app
st.title("Sentiment Analysis Model")

# Add a text area for the user to input a review
user_review = st.text_area("Enter a review:")

# If the user inputs a review, perform the prediction
if user_review:
    # Define the model URL from Google Drive (use the shared Google Drive link)
    model_url = 'https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B'  # Update with your actual model file ID
    model_file = 'sentiment_model.pkl'
    
    # Use gdown to download the model from Google Drive
    gdown.download(model_url, model_file, quiet=False)

    # Load the pre-trained model using torch.load, with map_location set to CPU
    model = torch.load(model_file, map_location=torch.device('cpu'))

    # You may also need to load a tokenizer if you have one. For example, if you used Hugging Face's tokenizer:
    tokenizer = joblib.load('tokenizer.pkl')  # Replace this if your tokenizer is stored elsewhere

    # Preprocessing the user's input, assuming your tokenizer requires the input to be tokenized
    inputs = tokenizer(user_review, return_tensors='pt')

    # Perform the sentiment prediction (ensure your model's forward method is correctly implemented)
    prediction = model(inputs['input_ids'])

    # Display the result
    if prediction == 0:
        st.write("Sentiment: Negative")
    elif prediction == 1:
        st.write("Sentiment: Neutral")
    else:
        st.write("Sentiment: Positive")
