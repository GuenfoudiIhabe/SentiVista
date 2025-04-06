FROM python:3.9

WORKDIR /app

COPY app.py /app
COPY sentiment_model_lr.pkl /app
COPY tfidf_vectorizer.pkl /app
COPY requirements.txt /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV FLASK_APP=app.py

# Run hello.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]