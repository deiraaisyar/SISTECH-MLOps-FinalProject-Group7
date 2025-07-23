# Gunakan Python 3.13 slim
FROM python:3.10-slim

# Install dependencies untuk build Rust & wheel
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust untuk build tokenizers jika tidak ada wheel
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt

# Copy seluruh source code ke container
COPY . /app

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI dengan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
