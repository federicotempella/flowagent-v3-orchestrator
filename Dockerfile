# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dipendenze minime di sistema (aggiungi qui se ti servono altre lib di OS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copia solo requirements per sfruttare la cache
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) Copia il resto del codice
COPY . .

# 3) Utente non-root (best practice)
RUN useradd -m appuser
USER appuser

# 4) Avvio: Render passa $PORT; in locale default 8080
#    Uso sh -c per espandere ${PORT:-8080}
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
