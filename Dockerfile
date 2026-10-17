# Use official slim Python runtime
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Default entrypoint runs unified pipeline
ENTRYPOINT ["python", "main_unified_pipeline.py"]
CMD ["--dataset", "all"]
