<p align="center">
  <img src="logo.jpg" alt="My Image" width="400"/>
</p>

# SentiVista: Social Media Sentiment Analysis

## 📊 Overview

SentiVista is an advanced sentiment analysis system that analyzes text content to determine emotional tone. Using machine learning models trained on the Sentiment140 dataset (1.6 million tweets), SentiVista can accurately classify text as expressing positive or negative sentiment with approximately 77% accuracy.

## 🔍 Key Features

- **Dual Model Architecture:** Uses both Naive Bayes and Logistic Regression models
- **Text Preprocessing Pipeline:** Removes noise and normalizes text data
- **Feature Engineering:** Combines TF-IDF vectorization with metadata features
- **REST API:** Simple HTTP interface for real-time sentiment prediction
- **Interactive Web Interface:** Test predictions through a user-friendly UI
- **Comprehensive Test Suite:** Evaluates model performance across various text types

## 🛠️ Technical Implementation

### Preprocessing Pipeline

### Feature Engineering
- **TF-IDF Vectorization:** 10,000 text features capturing word importance
- **Metadata Features:** Tweet length, hashtag count, mention count
- **Combined Features:** 10,003 total features per text sample

### Models

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 76% | 76% | 76% | 76% |
| Logistic Regression | 77% | 77% | 77% | 77% |

## 📋 Installation & Setup (local)


### **Clone the repository:**
   ```
   git clone https://github.com/yourusername/sentivista.git
   cd sentivista
   ```

### **Run in container**

  - start Docker Desktop
  - in the same directory as the root of the project
  ```
    docker build -t app .
    docker run -p 5001:5001 app
  ```


## 🚀 Usage Examples

### Web Interface (local)
Access the web UI at http://localhost:5001 after starting the server.

Example workflow:
1. Enter your text in the input field
2. Click "Analyze Sentiment"
3. View the sentiment analysis results with confidence score

### ☁️ Web Interface (cloud)

The app is currently running on the cloud at https://app-24294949938.europe-west1.run.app .
It is not garantied to be up at any moment but at least during the review.

### API Endpoint

SentiVista provides a RESTful API for sentiment analysis:

**Endpoint:** `/predict`

**Method:** POST

**Request Body:**
```json
{
  "texts": ["Your text to analyze"]
}
```

**Response:**
```json
{
  "predictions": [
    4
  ],
  "sentiment_labels": [
    "Positive"
  ]
}
```

**Example using curl (all platforms, local run):**
```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"texts\":[\"I love this product, it works great!\"]}"
```

**Example using PowerShell (Windows, local run):**
```powershell
Invoke-WebRequest -Uri "http://localhost:5001/predict" -Method Post -ContentType "application/json" -Body '{"texts": ["I love this product, it works great!"]}'
```

**Example with the Cloud-hosted application:**
*Idem but replace the ```localhost``` url by the "real" one.*

### Testing Script

For batch processing and testing, use the included `test_api.py` script:

```bash
# Windows
python test_api.py

# macOS/Linux
python3 test_api.py
```

The `test_api.py` file contains a ready-to-use script that tests multiple examples and outputs the sentiment analysis results. You can easily modify this file to test your own text samples.

## 🔬 Project Goals

- **Real-time Sentiment Monitoring:** Track public opinion on current events
- **Trend Analysis:** Identify emerging topics and sentiment shifts
- **Market Intelligence:** Gauge customer reactions to products and services
- **Public Opinion Research:** Analyze feedback at scale

## 🧠 How It Works

1. **Text Processing:** Cleans and normalizes input text
2. **Feature Extraction:** Converts text to 10,003-dimensional vectors
3. **Model Prediction:** Applies ML models to determine sentiment polarity
4. **Result Classification:** Returns sentiment as "Positive" or "Negative"

## 📁 Project Data

We use the [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140) available on Kaggle. This dataset consists of tweets published around 2009 and includes:

- **Tweet ID:** Unique identifier for each tweet
- **Tweet Text:** The content of the tweet
- **Date Published:** Timestamp of when the tweet was published
- **Username:** Twitter handle of the tweet's author
- **Query:** The query used to retrieve the tweet (if available)
- **Sentiment Score:** A manually annotated score indicating the sentiment (0 for negative, 4 for positive)

## 📁 Repository Structure

```
SentiVista/
├── app.py                  # Flask API and web interface
├── Dockerfile              # Build container
├── requirements.txt        # Required packages to run the app
├── model.ipynb             # Model training and evaluation
├── EDA.ipynb               # Dataset exploration
├── test_api.py             # API testing script
├── tfidf_vectorizer.pkl    # Saved vectorizer (preprocess data)
├── sentiment_model_lr.pkl  # Logistic Regression model
├── sentiment_model_nb.pkl  # Naive Bayes model
└── README.md               # Documentation
```

## 👥 SentiVista Team

- **DI RENZO Julien** - [julien.direnzo@student.uliege.be](mailto:julien.direnzo@student.uliege.be)
- **FLOREA Robert** - [robert.florea@student.uliege.be](mailto:robert.florea@student.uliege.be)
- **GUENFOUDI Ihabe** - [ihabe.guenfoudi@student.uliege.be](mailto:ihabe.guenfoudi@student.uliege.be)
- **LEFEVRE Thibaut** - [thibaut.lefevre@student.uliege.be](mailto:thibaut.lefevre@student.uliege.be)

## 📞 Contact

- [t.vrancken@uliege.be](mailto:t.vrancken@uliege.be)
- [Matthias.Pirlet@uliege.be](mailto:Matthias.Pirlet@uliege.be)
