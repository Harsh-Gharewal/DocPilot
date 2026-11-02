# DocPilot Production Container
FROM python:3.12-slim

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt matplotlib

# Copy application source code
COPY . .

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Default command launches FastAPI server
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

