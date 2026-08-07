FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for PyMuPDF, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy the backend code and the rag code
COPY backend/ /app/backend/
COPY rag/ /app/rag/

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
