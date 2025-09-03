# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CACHE_DIR=/app/.cache
ENV KB_QCACHE_TTL=86400
ENV URL_TXT_TTL=2592000
ENV KB_DOCS_DIR=/app/kb/raw

RUN apt-get update && apt-get install -y --no-install-recommends

#fonts-dejavu-core
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

#Copia codice
COPY . .

#crea la cache e assegna permessi all'utente non-root
RUN mkdir -p "$CACHE_DIR" && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

#Avvio (FastAPI con uvicorn)
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]