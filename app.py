from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.sparse import hstack

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_tweet(tweet):
    """
    Clean tweets using the same preprocessing steps as in training
    """
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove mentions
    tweet = re.sub(r"@\w+", "", tweet)
    # Remove special characters, numbers, and punctuations
    tweet = re.sub(r"\W", " ", tweet)
    tweet = re.sub(r"\d", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    # Simple tokenization by splitting on whitespace
    tokens = tweet.split()
    # Remove stopwords and apply stemming/lemmatization
    tokens = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in tokens
        if word.lower() not in stop_words
    ]
    # Rejoin tokens into a single string
    tweet = " ".join(tokens)
    return tweet


def batch_predict(
    texts, model_path="sentiment_model_lr.pkl", vectorizer_path="tfidf_vectorizer.pkl"
):
    """
    Prediction function with additional features to match the training process
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Clean texts using the same function used during training
    cleaned_texts = [clean_tweet(text) for text in texts]

    # Transform texts to numerical features using TF-IDF
    X_tfidf = vectorizer.transform(cleaned_texts)

    # Extract additional features just like in training
    tweet_lengths = [len(text) for text in texts]
    hashtag_counts = [
        len([word for word in text.split() if word.startswith("#")]) for text in texts
    ]
    mention_counts = [
        len([word for word in text.split() if word.startswith("@")]) for text in texts
    ]

    # Combine all features into the same format used during training
    X_additional_features = np.array([tweet_lengths, hashtag_counts, mention_counts]).T
    X_combined = hstack([X_tfidf, X_additional_features])

    # Predict sentiment
    predictions = model.predict(X_combined)
    return predictions


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])
    model_name = data.get("model", "sentiment_model_lr.pkl")

    try:
        predictions = batch_predict(texts, model_path=model_name)
        sentiment_labels = [
            "Negative" if pred == 0 else "Positive" for pred in predictions
        ]

        return jsonify(
            {"predictions": predictions.tolist(), "sentiment_labels": sentiment_labels}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
