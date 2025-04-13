import os
import gdown
import joblib
from transformers import AutoTokenizer
import torch
from torch.nn.functional import softmax

# 🔽 File ID and local filename
file_id = "1pKFpU56YyLloC5IONDMxih5QMQSew54B"  # Replace with the correct file ID
output_path = "sentiment_model.pkl"

# 🔽 Download the model if not already present
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# 🔽 Load tokenizer (you can replace with your custom tokenizer path if needed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 🔽 Load the model
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(output_path)

model.eval()

# 🔽 Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    predicted_class = probs.argmax(dim=1).item()
    confidence = probs[0, predicted_class].item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[predicted_class], confidence
