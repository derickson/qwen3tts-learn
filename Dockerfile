FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 via deadsnakes PPA, plus audio libs
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 python3.12-venv python3.12-dev \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

# Install Python dependencies
RUN pip install qwen-tts soundfile
RUN pip install fastapi 'uvicorn[standard]' python-multipart
RUN pip install flash-attn --no-build-isolation
RUN pip install nvidia-ml-py
RUN pip install anthropic

# Copy application code and default voice sample
WORKDIR /app
COPY app/ ./app/
COPY DaveSample.m4a .

EXPOSE 8335

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8335"]
