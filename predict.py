import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer (assuming you downloaded them from GitHub or have them locally)
model = joblib.load("sentiment_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return labels[prediction], probs[0][prediction].item()
