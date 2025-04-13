import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model (adjust path if needed)
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Title
st.title("☕ Coffee Review Sentiment Analyzer")

# User input
user_input = st.text_area("Enter your coffee review:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        # You may need to preprocess input to match your model
        input_df = pd.DataFrame([user_input], columns=["review"])
        prediction = model.predict(input_df["review"])[0]

        # If your model returns a number (e.g., 0,1,2), map to label
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_map.get(prediction, "Unknown")

        st.success(f"Predicted Sentiment: **{sentiment}** 🎯")
