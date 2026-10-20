# Use official Python 3.12 slim image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Explicitly copy .env into the container as requested
COPY .env .env

# Ensure entrypoint script has execution permissions
RUN chmod +x /app/entrypoint.sh

# Run entrypoint script which first generates the vector DB and keeps container active
ENTRYPOINT ["/app/entrypoint.sh"]
