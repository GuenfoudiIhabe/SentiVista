FROM python:3.9-slim

# Set environment variables to force CPU-only PyTorch
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"

WORKDIR /app

# Copy application files
COPY app.py roberta_model.py /app/
COPY requirements.txt /app/

# Copy model files
COPY sentiment_model_lr.pkl sentiment_model_nb.pkl tfidf_vectorizer.pkl /app/

# Create directory for RoBERTa model and copy all files
RUN mkdir -p /app/saved_models/roberta_full_model
COPY saved_models/roberta_full_model/* /app/saved_models/roberta_full_model/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Install dependencies - install werkzeug first to ensure correct version
RUN pip install --no-cache-dir werkzeug==2.0.3 && \
    pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0

# Run the application with python directly instead of using flask command
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]