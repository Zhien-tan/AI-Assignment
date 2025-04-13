import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Load the model and tokenizer (modified to load on CPU)
def load_model():
    model_file = 'path/to/your/sentiment_model.pth'  # Path to your model file
    model = torch.load(model_file, map_location=torch.device('cpu'))  # Load model to CPU
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # You can modify if you have a custom tokenizer
    return model, tokenizer

# Define a function to predict sentiment
def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()  # Get the class with the highest logit
    return predicted_class

# Streamlit app layout
def app():
    st.title("Sentiment Analysis")
    
    # Text input from the user
    user_input = st.text_area("Enter a review:")
    
    if st.button('Analyze'):
        if user_input:
            # Load model and tokenizer
            model, tokenizer = load_model()
            
            # Perform sentiment analysis
            predicted_class = predict_sentiment(model, tokenizer, user_input)
            
            # Map predicted class to sentiment label (modify as needed)
            if predicted_class == 1:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
            
            # Display the result
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter a review for analysis.")

if __name__ == "__main__":
    app()
