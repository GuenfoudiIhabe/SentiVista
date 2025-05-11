"""
SentiVista Sentiment Analysis Application
"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.sparse import hstack

# Don't import torch or transformers here to avoid PyTorch CUDA issues

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)  

# Define model paths
ROBERTA_MODEL_PATH = './saved_models/roberta_full_model'  # Updated path to match notebook
TRADITIONAL_MODELS = {
    'lr': 'sentiment_model_lr.pkl',
    'nb': 'sentiment_model_nb.pkl'
}

# Set up for traditional models
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Cache for loaded models to avoid reloading
model_cache = {}

# Check if RoBERTa is available without loading the model
try:
    import roberta_model
    roberta_available, roberta_error = roberta_model.check_availability()
    if not roberta_available:
        print(f"RoBERTa model will not be available: {roberta_error}")
except ImportError:
    roberta_available = False
    print("RoBERTa module not found. RoBERTa model will not be available.")

def clean_tweet(tweet):
    """
    Clean tweets using the same preprocessing steps as in training
    """
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove special characters, numbers, and punctuations
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\d', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    # Simple tokenization by splitting on whitespace
    tokens = tweet.split()
    # Remove stopwords and apply stemming/lemmatization
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens if word.lower() not in stop_words]
    # Rejoin tokens into a single string
    tweet = ' '.join(tokens)
    return tweet

def batch_predict_roberta(texts):
    """Prediction function for the RoBERTa model"""
    # Only import and use roberta_model when actually needed
    try:
        import roberta_model
        
        # Load model if not already in cache
        if 'roberta' not in model_cache:
            success, model_data, error = roberta_model.load_model(ROBERTA_MODEL_PATH)
            if not success:
                print("failed loading model", flush=True)
                raise Exception(error)
            model_cache['roberta'] = model_data
        else:
            model_data = model_cache['roberta']
        
        # Make predictions
        predictions, results, error = roberta_model.predict(texts, model_data)
        if error:
            raise Exception(error)
        
        return predictions, results
    except Exception as e:
        raise Exception(f"Failed to use RoBERTa model: {str(e)}")

def batch_predict_traditional(texts, model_type='lr'):
    """Prediction function with traditional ML models"""
    model_path = TRADITIONAL_MODELS.get(model_type, TRADITIONAL_MODELS['lr'])
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    # Load model and vectorizer from cache or disk
    cache_key = f"trad_{model_type}"
    if cache_key not in model_cache:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        model_cache[cache_key] = (model, vectorizer)
    else:
        model, vectorizer = model_cache[cache_key]
    
    # Clean texts using the same function used during training
    cleaned_texts = [clean_tweet(text) for text in texts]
    
    # Transform texts to numerical features using TF-IDF
    X_tfidf = vectorizer.transform(cleaned_texts)
    
    # Extract additional features just like in training
    tweet_lengths = [len(text) for text in texts]
    hashtag_counts = [len([word for word in text.split() if word.startswith('#')]) for text in texts]
    mention_counts = [len([word for word in text.split() if word.startswith('@')]) for text in texts]
    
    # Combine all features into the same format used during training
    X_additional_features = np.array([tweet_lengths, hashtag_counts, mention_counts]).T
    X_combined = hstack([X_tfidf, X_additional_features])
    
    # Predict sentiment
    predictions = model.predict(X_combined)
    
    # Get probabilities if the model supports it
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_combined)
    
    return predictions, probabilities

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('texts', [])
    model_name = data.get('model', 'lr')  # Default to logistic regression model
    
    try:
        detailed_results = []
        
        # Use appropriate model based on selection
        if model_name == 'roberta':
            if not roberta_available:
                return jsonify({'error': 'RoBERTa model is not available due to missing PyTorch/Transformers dependencies.'}), 400
            predictions, detailed = batch_predict_roberta(texts)
            detailed_results = detailed
        else:  # Traditional ML models (lr or nb)
            predictions, probabilities = batch_predict_traditional(texts, model_name)
            
            # Format detailed results if probabilities are available
            if probabilities is not None:
                for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
                    detailed_results.append({
                        'prediction': int(pred),
                        'confidence': float(max(probs)),
                        'probabilities': probs.tolist()
                    })
        
        sentiment_labels = ["Negative" if pred == 0 else "Positive" for pred in predictions]
        
        return jsonify({
            'predictions': predictions.tolist(),
            'sentiment_labels': sentiment_labels,
            'detailed_results': detailed_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Print status message about available models
    print("\n" + "="*50)
    print("SentiVista Sentiment Analysis API")
    print("="*50)
    print("Available models:")
    print("- Logistic Regression (lr)")
    print("- Naive Bayes (nb)")
    if roberta_available:
        print("- RoBERTa Deep Learning (roberta)")
    else:
        print("- RoBERTa Deep Learning: NOT AVAILABLE (PyTorch/CUDA dependency issues)")
    print("\nStarting server...\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)