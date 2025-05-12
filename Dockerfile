FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY app.py /app
COPY sentiment_model_lr.pkl /app
COPY tfidf_vectorizer.pkl /app
COPY requirements.txt /app

ENV PORT 5001

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app