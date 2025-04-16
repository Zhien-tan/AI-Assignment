import streamlit as st
import torch
import gdown
import os
import pickle
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Setup
st.set_page_config(page_title="3-Way Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Classifies reviews as **Positive** 😊, **Neutral** 😐, or **Negative** 😞")

@st.cache_resource
def load_model():
    # Try multiple loading methods
    model_files = [
        ("sentiment_model.pth", self.load_weights),
        ("sentiment_model.pkl", self.load_pickle),
        ("https://drive.google.com/uc?id=19j0ACP1HblX7rYUMOmTAdqPAgofkgIdH", self.download_and_load)
    ]
    
    for file_info, loader in model_files:
        model, tokenizer = loader(file_info)
        if model is not None:
            return model, tokenizer, ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    return None, None, ["NEGATIVE", "NEUTRAL", "POSITIVE"]

def load_weights(self, file_path):
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3
        )
        model.load_state_dict(torch.load(file_path, map_location='cpu'))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except:
        return None, None

def load_pickle(self, file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model = model.to('cpu')
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except:
        return None, None

def download_and_load(self, url):
    try:
        model_file = "downloaded_model.pth"
        gdown.download(url, model_file, quiet=True)
        return self.load_weights(model_file)
    except:
        return None, None

# Load model
model, tokenizer, class_names = load_model()

if model is None:
    st.error("""
    ❌ Could not load any model. Please:
    
    1. Re-save your model using:
    ```python
    # Recommended method
    torch.save(model.to('cpu').state_dict(), 'sentiment_model.pth')
    
    # Alternative method
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model.to('cpu'), f, protocol=pickle.HIGHEST_PROTOCOL)
    ```
    
    2. Upload to Google Drive
    3. Update the file ID in this code
    """)
    st.stop()

# Main app
review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment", type="primary") and review:
    with st.spinner("Analyzing..."):
        try:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            
            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment = class_names[pred_class]
                emoji = "😊" if sentiment == "POSITIVE" else \
                       "😐" if sentiment == "NEUTRAL" else "😞"
                st.metric("Sentiment", f"{sentiment} {emoji}")
            
            with col2:
                confidence = probs[pred_class].item()
                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)
            
            # Detailed scores
            with st.expander("Detailed Scores"):
                for i, name in enumerate(class_names):
                    st.metric(name, f"{probs[i].item():.1%}")
                    
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Environment info
with st.expander("⚙️ System Information"):
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    if os.path.exists("sentiment_model.pth"):
        st.write(f"Model file size: {os.path.getsize('sentiment_model.pth')/1e6:.2f} MB")
    if os.path.exists("sentiment_model.pkl"):
        st.write(f"Pickle file size: {os.path.getsize('sentiment_model.pkl')/1e6:.2f} MB")
