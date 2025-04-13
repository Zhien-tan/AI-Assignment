import os
import gdown
import joblib
import numpy as np
from transformers import AutoTokenizer
from torch.nn.functional import softmax
import torch

# 🔽 File ID and model path
file_id = "1WLYPN-8N-rXBNbt1sMspTgk_SJ3yc_Vw"
output_path = "sentiment_model.pkl"

# 🔽 Download model if not exists
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# 🔽 Load tokenizer (assumes tokenizer folder exists in project directory)
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# 🔽 Load model
model = joblib.load(output_path)

# 🔽 Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[predicted_class], confidence
