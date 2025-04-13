import gdown
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# ⬇️ Download the model from Google Drive
gdown.download("https://drive.google.com/file/d/1WLYPN-8N-rXBNbt1sMspTgk_SJ3yc_Vw/view?usp=sharing", "sentiment_model", quiet=False)

# ⬇️ Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # or your custom tokenizer path
model = torch.load("sentiment_model", map_location=torch.device("cpu"))
model.eval()

# ⬇️ Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class
