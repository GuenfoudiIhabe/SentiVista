from flask import Flask, request, jsonify
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
    return """
    <html>
        <head>
            <title>SentiVista Sentiment Analysis</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 0 auto;
                    padding: 20px; }
                textarea { width: 100%; height: 100px; margin-bottom: 10px; }
                button { padding: 10px; background-color: #4CAF50;
                    color: white; border: none; cursor: pointer; }
                .results { margin-top: 20px; }
                .positive { color: green; font-weight: bold; }
                .negative { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>SentiVista Sentiment Analysis</h1>
            <textarea id="textInput" placeholder="Enter text to analyze">
            I love this app, it's amazing!</textarea>
            <button onclick="analyzeSentiment()">Analyze Sentiment</button>
            <div class="results" id="results"></div>
            <script>
                async function analyzeSentiment() {
                    const textInput = document.getElementById('textInput').value;
                    const texts = textInput.split('\\n').filter(
                    text => text.trim() !== '');
                    document.getElementById('results').innerHTML =
                    '<p>Analyzing...</p>';
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ texts }),
                        });
                        const data = await response.json();
                        if (data.error) {
                            document.getElementById('results').innerHTML =
                            `<p>Error: ${data.error}</p>`;
                        } else {
                            let resultsHtml = '<h2>Results:</h2>';
                            texts.forEach((text, index) => {
                                const sentiment = data.sentiment_labels[index];
                                const sentimentClass = sentiment === 'Positive' ?
                                'positive' : 'negative';
                                resultsHtml += `<p><strong>Text:</strong>
                                ${text}<br><strong>Sentiment:</strong>
                                <span class="${sentimentClass}">${sentiment}
                                </span></p>`;
                            });
                            document.getElementById('results').innerHTML = resultsHtml;
                        }
                    } catch (error) {
                        document.getElementById('results').innerHTML = `<p>Error:
                            ${error.message}</p>`;
                    }
                }
            </script>
        </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
