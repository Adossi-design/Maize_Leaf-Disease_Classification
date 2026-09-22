FROM python:3.11-slim

# Hugging Face Spaces run the container as user 1000, so keep caches writable.
ENV HOME=/tmp \
    KERAS_HOME=/tmp/.keras \
    MPLCONFIGDIR=/tmp/.matplotlib \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
