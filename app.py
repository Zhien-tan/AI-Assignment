import streamlit as st
from transformers import pipeline

# Title of the app
st.title("Sentiment Analysis Model")

# Load a pretrained sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Text input from the user
user_input = st.text_area("Enter a review:")

# When the user clicks the analyze button
if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = model(user_input)[0]
            label = result['label']
            score = result['score']
            st.success(f"**Sentiment:** {label} ({score:.2f})")
    else:
        st.warning("Please enter some text to analyze.")
