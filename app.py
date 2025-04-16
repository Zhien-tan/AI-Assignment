def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            st.warning("Please enter some text to analyze")
            return None
            
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and process
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.softmax(logits, dim=1)[0]
        
        # Debug output
        print(f"Raw probabilities: {probs.tolist()}")  # For debugging
        
        # Determine class mapping (critical fix)
        if probs[0] > probs[1]:  # Index 0 is negative
            return {
                "label": "NEGATIVE",
                "score": probs[0].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        else:
            return {
                "label": "POSITIVE",
                "score": probs[1].item(),
                "pos_score": probs[1].item(),
                "neg_score": probs[0].item()
            }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        st.error(f"❌ Tokenizer failed to load: {str(e)}")
        return None

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            st.warning("Please enter some text to analyze")
            return None
            
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Predict (handling different model types)
        with torch.no_grad():
            if hasattr(model, '__call__'):
                outputs = model(**inputs)
            elif hasattr(model, 'predict'):
                outputs = model.predict(inputs)
            else:
                raise ValueError("Model doesn't have callable predict method")
        
        # Process results
        if isinstance(outputs, dict):
            logits = outputs['logits']
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]
            
        probs = torch.softmax(logits, dim=1)[0]
        return {
            "label": "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE",
            "score": probs.max().item(),
            "pos_score": probs[1].item(),
            "neg_score": probs[0].item()
        }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

# Main execution
if st.button("Analyze Sentiment", type="primary"):
    if not user_input:
        st.warning("Please enter a review first")
    else:
        with st.spinner("Setting up analysis..."):
            model = load_sentiment_model()
            tokenizer = load_tokenizer()
        
        if model and tokenizer:
            result = predict_sentiment(model, tokenizer, user_input)
            
            if result:
                st.subheader("Results")
                if result['label'] == "POSITIVE":
                    st.success(f"😊 Positive ({result['score']:.0%} confidence)")
                else:
                    st.error(f"😞 Negative ({result['score']:.0%} confidence)")
                
                st.progress(result['score'])
                
                with st.expander("Detailed scores"):
                    col1, col2 = st.columns(2)
                    col1.metric("Positive", f"{result['pos_score']:.1%}")
                    col2.metric("Negative", f"{result['neg_score']:.1%}")

# Troubleshooting section
with st.expander("⚠️ Troubleshooting"):
    st.markdown("""
    **Common issues and solutions:**
    
    1. **Model fails to load**:
       - Delete any existing `sentiment_model.pkl` file
       - Refresh the page to restart download
       - Ensure you have stable internet connection
    
    2. **Pickle errors**:
       - The app requires Python 3.8+
       - Make sure all dependencies are installed
    
    3. **Error messages**:
       - If you see "unsupported pickle", the file may be corrupted
       - Try re-downloading the model file
    """)

st.caption("Note: This app uses machine learning models. Initial loading may take time.")
