# Use lightweight Python image
FROM python:3.10-slim


# Prevent Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Set working directory
WORKDIR /app

# Install system dependencies (needed by sklearn, pandas)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first (for Docker cache)
COPY requirements-api.txt .


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt


# Copy application code
COPY src/ src/



# Expose FastAPI port
EXPOSE 8000


# Run FastAPI using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

