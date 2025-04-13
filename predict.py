from transformers import BertTokenizer, BertForSequenceClassification
import torch

output_path = "model"  # folder with model files

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(output_path)
model = BertForSequenceClassification.from_pretrained(output_path)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class
