FROM python:3.10-slim

# Install system dependencies required by dlib and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    liblapack3 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dlib via pre-built wheel
RUN pip install --no-cache-dir \
    https://github.com/RPi-Distro/python-dlib/releases/download/v19.24.0/dlib-19.24.0-cp310-cp310-manylinux_2_24_x86_64.whl

# Install face_recognition via pip (akan menggunakan dlib yang sudah ada)
RUN pip install --no-cache-dir face_recognition==1.3.0

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasinya
CMD ["python", "face-rec-api.py"]
