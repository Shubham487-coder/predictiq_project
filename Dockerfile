# Base image — Python 3.11 slim (small, no extras)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Dockerfile caches this layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY api/ ./api/
COPY models/ ./models/

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]