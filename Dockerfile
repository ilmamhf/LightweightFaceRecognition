# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies including cmake and build-essential for compiling dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your face-recognition API
CMD ["python", "face-rec-api.py"]
