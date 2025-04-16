import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up Streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment (Positive/Negative)")

# User input
user_input = st.text_area("Review Text:", height=150, placeholder="Type your movie review here...")

@st.cache_resource
def download_and_load_model():
    try:
        # Download model from Google Drive
        file_id = "19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "sentiment_model.pth"
        
        if not os.path.exists(output):
            with st.spinner("📥 Downloading model (this may take a minute)..."):
                gdown.download(url, output, quiet=False)
        
        # Initialize device
        device = torch.device('cpu')
        
        # Load with weights_only=False since we trust the source
        with st.spinner("🔄 Loading model..."):
            state_dict = torch.load(
                output,
                map_location=device,
                weights_only=False  # Required for this model file
            )
            
            # Initialize model architecture
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            )
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            st.success("✅ Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        **Troubleshooting steps:**
        1. Check your internet connection
        2. Ensure you have sufficient disk space
        3. The model file might be corrupted - try deleting 'sentiment_model.pth' and restarting the app
        """)
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with st.spinner("🔤 Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            return tokenizer
    except Exception as e:
        st.error(f"❌ Tokenizer loading failed: {str(e)}")
        return None

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            st.warning("⚠️ Please enter some text to analyze")
            return None
            
        with st.spinner("🧠 Analyzing sentiment..."):
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to CPU
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process results
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_label = "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE"
            confidence = probs.max().item()
            
            return {
                "label": pred_label,
                "score": confidence,
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
            
    except Exception as e:
        st.error(f"""
        ❌ Prediction error: {str(e)}
        
        This might happen if:
        - The review is too short or contains only symbols
        - There's a problem with the model
        """)
        return None

# Main execution
if st.button("Analyze Sentiment", type="primary"):
    if not user_input:
        st.warning("Please enter a review to analyze")
    else:
        model = download_and_load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                # Display results in a nicer layout
                st.subheader("🎯 Analysis Results")
                
                # Sentiment label with color and emoji
                if result['label'] == "POSITIVE":
                    st.success(f"### 😊 Positive sentiment! ({(result['score']*100):.1f}% confidence)")
                else:
                    st.error(f"### 😞 Negative sentiment ({(result['score']*100):.1f}% confidence)")
                
                # Confidence meter
                st.progress(result['score'], 
                          text=f"Confidence level: {result['score']:.1%}")
                
                # Detailed scores in expandable section
                with st.expander("📊 Detailed Scores", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Positive Score", 
                        f"{result['pos_score']:.1%}",
                        delta=f"+{(result['pos_score']-0.5):.1%}" if result['pos_score'] > 0.5 else None
                    )
                    col2.metric(
                        "Negative Score", 
                        f"{result['neg_score']:.1%}",
                        delta=f"+{(result['neg_score']-0.5):.1%}" if result['neg_score'] > 0.5 else None
                    )

# Add sidebar with info
with st.sidebar:
    st.markdown("""
    ## ℹ️ About This App
    
    This app analyzes movie review sentiment using:
    - DistilBERT base model
    - PyTorch backend
    - Custom fine-tuned classifier
    
    **Model Info:**
    - Trained on IMDB reviews
    - ~94% accuracy
    - Max sequence length: 512 tokens
    
    [GitHub Repository](#) | [Report Issue](#)
    """)
    
    st.markdown("---")
    st.markdown("**💡 Pro Tip:**")
    st.markdown("For best results, write at least 2-3 complete sentences with clear emotional words.")

# Footer
st.markdown("---")
st.caption("""
Note: This app loads a ~250MB model file on first run. 
Analysis may take a few seconds on first use.
""")
