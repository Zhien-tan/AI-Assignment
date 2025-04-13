import streamlit as st
import torch  # PyTorch library
import joblib
import gdown

# Title for the web app
st.title("Sentiment Analysis Model")

# Add a text area for the user to input a review
user_review = st.text_area("Enter a review:")

# If the user inputs a review, perform the prediction
if user_review:
    # Define the model URL from Google Drive (use the shared Google Drive link)
    model_url = 'https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B'  # Update with your actual model file ID
    model_file = 'sentiment_model.pkl'  # Assuming your model is a .pkl file

    # Use gdown to download the model from Google Drive
    gdown.download(model_url, model_file, quiet=False)

    # Load the pre-trained model using torch.load() with map_location for CPU
    try:
        model = torch.load(model_file, map_location=torch.device('cpu'))
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise

    # Set the model to evaluation mode
    model.eval()

    # Preprocess the review input as needed for the model (ensure you tokenize/format the input)
    # Example: Assuming you're using the tokenizer to prepare the input for the model
    # inputs = tokenizer(user_review, return_tensors='pt')

    # Perform the sentiment prediction (ensure your model’s forward method is correctly implemented)
    # Example for prediction:
    # Assuming model outputs logits and you want to convert them to probabilities
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        inputs = torch.tensor([user_review])  # Convert review text to tensor (implement tokenization as needed)
        outputs = model(inputs)  # Get the raw prediction from the model
        predicted_class = torch.argmax(outputs, dim=1).item()  # Get the predicted class (0, 1, or 2)
    
    # Display the result based on the model's output
    if predicted_class == 0:
        st.write("Sentiment: Negative")
    elif predicted_class == 1:
        st.write("Sentiment: Neutral")
    else:
        st.write("Sentiment: Positive")
