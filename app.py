import streamlit as st
import torch
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_resources():
    # Try multiple loading methods
    methods = [
        try_load_pickle,
        try_load_state_dict,
        try_load_huggingface
    ]
    
    for method in methods:
        model, tokenizer = method()
        if model is not None:
            return model, tokenizer
    
    st.error("All loading methods failed. Please check your model file.")
    return None, None

def try_load_pickle():
    try:
        # Download model
        model_file = "model.pkl"
        if not os.path.exists(model_file):
            gdown.download(
                "https://drive.google.com/uc?id=1pKFpU56YyLloC5IONDMxih5QMQSew54B",
                model_file,
                quiet=False
            )
        
        # Load with CPU mapping
        device = torch.device('cpu')
        model = torch.load(model_file, map_location=device, weights_only=False)
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except:
        return None, None

def try_load_state_dict():
    try:
        # Download weights
        weights_file = "model_weights.pth"
        if not os.path.exists(weights_file):
            st.warning("Could not find state_dict file. Trying other methods...")
            return None, None
            
        # Initialize fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        
        # Load weights
        model.load_state_dict(torch.load(weights_file, map_location='cpu'))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except:
        return None, None

def try_load_huggingface():
    try:
        st.warning("Falling back to pretrained Hugging Face model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except:
        return None, None

# Load resources
model, tokenizer = load_resources()

# Rest of your app code...
# [Keep your existing UI and prediction code]
