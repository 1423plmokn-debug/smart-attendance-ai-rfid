# Base image: Python 3.11 slim
FROM python:3.11-slim

# System dependencies for OpenCV, InsightFace, and compiler tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files and directories
COPY database.py main.py pkg_resources.py rfid_cards.json ./
COPY docs/ ./docs/

# Create runtime directories and ensure non-root permission compatibility (Hugging Face Spaces UID 1000)
RUN mkdir -p /app/known_faces /app/reports /app/models \
    && chmod -R 777 /app

# Environment variables for cloud deployment
ENV CLOUD_DEMO=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI server
CMD ["python", "main.py"]
