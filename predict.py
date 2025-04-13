import os
import gdown
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax

# ⬇Download the model from Google Drive
file_id = "1WLYPN-8N-rXBNbt1sMspTgk_SJ3yc_Vw"
output_path = "sentiment_model.pkl"

if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/file/d/1WLYPN-8N-rXBNbt1sMspTgk_SJ3yc_Vw/view?usp=sharing", output_path, quiet=False)

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Load model
model = torch.load(output_path, map_location=torch.device("cpu"))
model.eval()

# ✅ Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[predicted_class], confidence
