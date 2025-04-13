import os
import gdown
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax

# 🔽 File ID and model path
file_id = "1WLYPN-8N-rXBNbt1sMspTgk_SJ3yc_Vw"
output_path = "sentiment_model.pth"  # Changed to .pth, which is the typical PyTorch model extension

# 🔽 Download model if not exists
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# 🔽 Load tokenizer (you can replace with your custom tokenizer path if needed)
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# 🔽 Load model (torch.load is the correct method for PyTorch models)
model = torch.load(output_path, map_location=torch.device("cpu"))  # Use torch.load to load the model
model.eval()  # Put the model in evaluation mode

# 🔽 Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits  # Forward pass
    probs = softmax(logits, dim=1)  # Convert logits to probabilities
    predicted_class = torch.argmax(probs, dim=1).item()  # Get the predicted class index
    confidence = probs[0, predicted_class].item()  # Get the confidence score
    labels = ["Negative", "Neutral", "Positive"]  # Assuming 3 classes
    return labels[predicted_class], confidence
