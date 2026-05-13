FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XAI_HOST=0.0.0.0 \
    XAI_PORT=5000

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 5000

WORKDIR /app/src
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
