FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080 AAVAIL_WARMUP=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the per-country models at build time so the container can answer /predict immediately,
# then run the full unit-test suite; a failing test fails the build.
RUN python -c "import model; model.model_train()" && python run-tests.py

EXPOSE 8080
# One worker: the model cache and the training lock live in process memory.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:${PORT} app:app"]
