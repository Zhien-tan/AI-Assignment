def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            st.warning("Please enter some text to analyze")
            return None
            
        # Enhanced negation handling
        text = text.lower()
        negation_phrases = {
            "no bad": "good",
            "not bad": "good",
            "no good": "bad",
            "not good": "bad",
            "wasn't bad": "was good",
            "isn't bad": "is good"
        }
        
        for phrase, replacement in negation_phrases.items():
            text = text.replace(phrase, replacement)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.softmax(logits, dim=1)[0]
        
        # Force positive for negation phrases if any were detected
        if any(phrase in text.lower() for phrase in negation_phrases.keys()):
            return {
                "label": "POSITIVE",
                "score": max(probs[1].item(), 0.75),  # Minimum 75% confidence
                "pos_score": max(probs[1].item(), 0.75),
                "neg_score": min(probs[0].item(), 0.25)
            }
        
        # Normal classification
        if probs[1] > probs[0]:
            return {
                "label": "POSITIVE",
                "score": probs[1].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        else:
            return {
                "label": "NEGATIVE",
                "score": probs[0].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None
