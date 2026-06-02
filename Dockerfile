FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY *.py /app/

# Create necessary directories
RUN mkdir -p /app/src/data/graphs \
    && mkdir -p /app/src/data/graphs_hetero \
    && mkdir -p /app/src/artifacts \
    && mkdir -p /app/src/artifacts_hetero \
    && mkdir -p /app/artifacts

# Set Python path
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

# Expose API and dashboard ports
EXPOSE 8000 8501

# Default command — override per service in docker-compose
CMD ["python", "src/model_api.py"]
