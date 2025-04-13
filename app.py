import streamlit as st
from predict import predict_sentiment

st.title("☕ Coffee Review Sentiment Classifier")
st.write("Enter a coffee review to see if it's Positive, Neutral, or Negative.")

review = st.text_area("Write your review here:")

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review!")
    else:
        sentiment, confidence = predict_sentiment(review)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence Score: {confidence:.2f}")
