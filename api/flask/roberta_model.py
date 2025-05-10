"""
RoBERTa model handler - Uses CPU-only mode based on notebook code
"""
import os
import numpy as np

def check_availability():
    """
    Check if PyTorch and transformers are available
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Check if model directory exists
        model_path = './saved_models/roberta_full_model'
        if not os.path.exists(model_path):
            return False, f"Model not found at {model_path}"
        
        return True, None
    except Exception as e:
        return False, str(e)

def load_model(model_path=None):
    """
    Load RoBERTa model using the same code that works in the notebook
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Use the exact path that works in the notebook
        full_model_path = './saved_models/roberta_full_model'
        
        print("Loading RoBERTa model...")
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        
        print("RoBERTa model loaded successfully!")
        return True, (model, tokenizer), None
    except Exception as e:
        error_msg = f"Error loading RoBERTa model: {str(e)}"
        print(error_msg)
        return False, None, error_msg

def predict(texts, model_data):
    """
    Predict sentiment using the same function that works in the notebook
    """
    model, tokenizer = model_data
    results = []
    predictions_array = []
    
    try:
        import torch
        
        for text in texts:
            # Using the exact prediction function from the notebook
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get prediction
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            # Map prediction to sentiment
            sentiment = "Positive" if prediction == 1 else "Negative"
            
            results.append({
                'prediction': prediction,
                'label': sentiment,
                'score': confidence,
                'all_scores': probabilities[0].tolist()
            })
            
            predictions_array.append(prediction)
        
        return np.array(predictions_array), results, None
    except Exception as e:
        error_msg = f"Error making predictions with RoBERTa: {str(e)}"
        print(error_msg)
        return None, None, error_msg
