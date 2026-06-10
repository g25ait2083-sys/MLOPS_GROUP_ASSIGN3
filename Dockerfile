# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Accept model name as build argument
ARG HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script
COPY src/inference.py .

# Run inference
CMD ["python", "inference.py"]
