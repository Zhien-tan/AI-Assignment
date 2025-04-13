import streamlit as st
from predict import predict_sentiment

st.title("☕ Coffee Review Sentiment Classifier")
st.write("Enter a coffee review below to predict whether it's Positive, Neutral, or Negative.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence Score: {confidence:.2f}")
