<p align="center">
  <img src="logo.jpg" alt="My Image" width="400"/>
</p>

# SentiVista: Social Media Sentiment Analysis

[![SentiVista CI/CD](https://github.com/GuenfoudiIhabe/SentiVista/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/GuenfoudiIhabe/SentiVista/actions/workflows/ci-cd.yml)

## **Milestone 3 Review**

What's new?
- RoBERTa fine-tuned model with 80% accuracy
- CI/CD pipeline with GitHub Actions
- Dual deployment architecture (Flask API and Streamlit UI)
- Code quality improvements and automated testing
- Updated requirements to fix security vulnerabilities

## 📊 Overview

SentiVista is an advanced sentiment analysis system that analyzes text content to determine emotional tone. Using machine learning models trained on the Sentiment140 dataset (1.6 million tweets), SentiVista can accurately classify text as expressing positive or negative sentiment with up to 80% accuracy using our fine-tuned RoBERTa model.

## 🔍 Key Features

- **Triple Model Architecture:** Uses Naive Bayes, Logistic Regression, and fine-tuned RoBERTa models
- **Text Preprocessing Pipeline:** Removes noise and normalizes text data
- **Feature Engineering:** Combines TF-IDF vectorization with metadata features
- **REST API:** Simple HTTP interface for real-time sentiment prediction
- **Interactive Web Interface:** Test predictions through a user-friendly UI
- **Comprehensive Test Suite:** Evaluates model performance across various text types
- **CI/CD Pipeline:** Automated testing and deployment with GitHub Actions

## 🛠️ Technical Implementation

### Preprocessing Pipeline
Our preprocessing pipeline removes URLs, mentions, special characters, and stopwords from text, then applies stemming and lemmatization to normalize the content.

### Feature Engineering
- **TF-IDF Vectorization:** 10,000 text features capturing word importance
- **Metadata Features:** Tweet length, hashtag count, mention count
- **Combined Features:** 10,003 total features per text sample
- **Transformer Embeddings:** RoBERTa model utilizes contextualized word embeddings

### Models

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 76% | 76% | 76% | 76% |
| Logistic Regression | 77% | 77% | 77% | 77% |
| RoBERTa Fine-tuned | 80% | 81% | 79% | 80% |

## 📋 Installation & Setup (local)

### **Clone the repository:**
   ```
   git clone https://github.com/yourusername/sentivista.git
   cd sentivista
   ```

### **Run Flask API in container**

  - Start Docker Desktop
  - In the same directory as the root of the project
  ```
    docker build -t sentiment-api -f api/flask/Dockerfile api/flask
    docker run -p 5000:5000 sentiment-api
  ```

### **Run Streamlit UI in container**

  ```
    docker build -t sentiment-ui -f api/streamlit/Dockerfile api/streamlit
    docker run -p 8501:8501 sentiment-ui
  ```

## 🚀 Usage Examples

### Web Interface (local)
Access the web UI at http://localhost:8501 after starting the Streamlit server.

Example workflow:
1. Enter your text in the input field
2. Choose your preferred model (Logistic Regression, Naive Bayes, or RoBERTa)
3. Click "Analyze Sentiment"
4. View the sentiment analysis results with confidence score

### ☁️ Web Interface (cloud)

The app is currently running on the cloud at https://app-24294949938.europe-west1.run.app.
It is not guaranteed to be up at any moment but at least during the review.

### API Endpoint

SentiVista provides a RESTful API for sentiment analysis:

**Endpoint:** `/predict`

**Method:** POST

**Request Body:**
```json
{
  "texts": ["Your text to analyze"],
  "model": "roberta"  // Options: "lr", "nb", "roberta"
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
  ],
  "detailed_results": [
    {
      "prediction": 4,
      "confidence": 0.92,
      "probabilities": [0.08, 0.92]
    }
  ]
}
```

**Example using curl (all platforms, local run):**
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"texts\":[\"I love this product, it works great!\"], \"model\":\"roberta\"}"
```

**Example using PowerShell (Windows, local run):**
```powershell
Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method Post -ContentType "application/json" -Body '{"texts": ["I love this product, it works great!"], "model": "roberta"}'
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

## 🧪 Continuous Integration/Continuous Deployment

SentiVista uses GitHub Actions for CI/CD to ensure code quality and testing:

- **Automated Testing:** All code changes are automatically tested with pytest
- **Code Quality Checks:** Pre-commit hooks ensure consistent code formatting
- **Mock Deployment:** Workflow includes a simulation of deployment steps when code is merged to main
- **Dependency Management:** Requirements are specified for consistent environments

The CI/CD pipeline triggers automatically on:
- Pushes to the `main` and `ci-cd-setup` branches
- Pull requests to the `main` branch
- Changes to key files including README.md, API code, and tests

To view the pipeline status and run history, check the [GitHub Actions tab](https://github.com/GuenfoudiIhabe/SentiVista/actions) in our repository.

## 🔬 Project Goals

- **Real-time Sentiment Monitoring:** Track public opinion on current events
- **Trend Analysis:** Identify emerging topics and sentiment shifts
- **Market Intelligence:** Gauge customer reactions to products and services
- **Public Opinion Research:** Analyze feedback at scale

## 🧠 How It Works

1. **Text Processing:** Cleans and normalizes input text
2. **Feature Extraction:** 
   - Traditional models: Convert text to TF-IDF vectors with metadata features
   - RoBERTa model: Generate contextual embeddings from the pre-trained transformer
3. **Model Prediction:** Apply selected ML model to determine sentiment polarity
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
├── api/
│   ├── flask/
│   │   ├── app.py                  # Flask API backend
│   │   ├── Dockerfile              # Docker configuration for API
│   │   ├── requirements.txt        # API dependencies
│   │   ├── roberta_model.py        # RoBERTa model implementation
│   │   └── saved_models/           # Trained model files
│   │       ├── roberta_full_model/  # RoBERTa model files
│   │       ├── sentiment_model_lr.pkl  # Logistic Regression model
│   │       └── sentiment_model_nb.pkl  # Naive Bayes model
│   └── streamlit/
│       ├── streamlit_app.py        # Streamlit UI application
│       ├── Dockerfile              # Docker configuration for UI
│       └── requirements.txt        # UI dependencies
├── tests/
│   ├── __init__.py                 # Package marker
│   └── test_app.py                 # API tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # CI/CD configuration
├── TweetNormalizer.py              # Text preprocessing utilities
├── test_api.py                     # API testing script
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── model.ipynb                     # Model training notebook
├── EDA.ipynb                       # Data exploration notebook
└── README.md                       # Documentation
```

## 👥 SentiVista Team

- **DI RENZO Julien** - [julien.direnzo@student.uliege.be](mailto:julien.direnzo@student.uliege.be)
- **FLOREA Robert** - [robert.florea@student.uliege.be](mailto:robert.florea@student.uliege.be)
- **GUENFOUDI Ihabe** - [ihabe.guenfoudi@student.uliege.be](mailto:ihabe.guenfoudi@student.uliege.be)
- **LEFEVRE Thibaut** - [thibaut.lefevre@student.uliege.be](mailto:thibaut.lefevre@student.uliege.be)

## 📞 Contact

- [t.vrancken@uliege.be](mailto:t.vrancken@uliege.be)
- [Matthias.Pirlet@uliege.be](mailto:Matthias.Pirlet@uliege.be)