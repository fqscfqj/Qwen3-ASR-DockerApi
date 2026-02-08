FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ID=Qwen/Qwen3-ASR-1.7B \
    MODEL_NAME=qwen3-asr-1.7b \
    MODEL_CACHE_DIR=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models

COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-server.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "qwen3_asr_toolkit.server:app", "--host", "0.0.0.0", "--port", "8000"]
