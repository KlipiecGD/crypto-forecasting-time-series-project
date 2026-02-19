FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code 
COPY inference.py .

# Environment variables
ENV MODEL_PATH /opt/ml/model

# Expose port 8080 (Good practice, though SageMaker works without it)
EXPOSE 8080

# Define the entrypoint. 
# SageMaker runs: docker run <image> serve
# This becomes: python inference.py serve
ENTRYPOINT ["python", "inference.py"]