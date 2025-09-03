# Usa una base Python leggera
FROM python:3.11-slim

# Evita bytecode e forza stdout/err unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Setta la working directory
WORKDIR /app

# Installa dipendenze di sistema utili (es: build tools, libxml)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia i requirements e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice dentro il container
COPY . .

# (Opzionale) esponi la porta 8080 solo a scopo documentativo
EXPOSE 8080

# Comando di avvio: Uvicorn su host 0.0.0.0 e porta fornita da Render
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

