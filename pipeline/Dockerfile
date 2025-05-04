FROM mirror.gcr.io/library/python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY run_pipeline.py .

ENTRYPOINT ["bash"]
