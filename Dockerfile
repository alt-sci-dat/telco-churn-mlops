# A Dockerfile is a recipe for building a "container image" — a self-contained
# box that holds your code + Python + every dependency at exact versions. Anyone
# (your laptop, a CI runner, Render, Hugging Face) can run that identical box, so
# "works on my machine" stops being a problem. The image is built once and run
# anywhere.

# Start from a slim official Python image (small = faster pulls, less to attack).
FROM python:3.12-slim

# Don't write .pyc files; flush logs immediately so they show up in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies FIRST, in their own layer. Docker caches layers, so as
# long as requirements.txt doesn't change, rebuilds skip re-installing — fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the trained artifacts.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY pyproject.toml README.md ./

# Install our package so `import churn` works inside the container.
RUN pip install --no-cache-dir -e . --no-deps

# Run as a non-root user — a basic but important security hardening step.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Render and Hugging Face inject the port to listen on via $PORT. Default 8000.
ENV PORT=8000
EXPOSE 8000

# Start the API. `--host 0.0.0.0` makes it reachable from outside the container.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
