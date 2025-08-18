# Flowagent V3 Orchestrator

Questo repo contiene l'AI Orchestrator (FastAPI) + OpenAPI + ingest RAG + prompt.

## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # compila OPENAI_API_KEY e BEARER_TOKEN
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## 2) Ingest KB (RAG)

Metti i tuoi file in `kb/raw/` (docx/pdf/md/txt) poi:

```bash
python scripts/ingest_kb.py --path ./kb/raw --out ./kb/index --industry retail --role CIO --lang it
```

## 3) Test veloci

Usa `scripts/smoke_tests.http` (con l'estensione REST Client) oppure curl:

```bash
curl -s -X POST http://localhost:8080/run/generate_assets   -H "Authorization: Bearer your_orchestrator_secret"   -H "Content-Type: application/json"   -d '{"mode":"AE","level":"Advanced","language":"it","persona_id":"fashion_cio"}'
```

## 4) Docker

```bash
docker build -t flowagent-v3 .
docker run -p 8080:8080 --env-file .env flowagent-v3
```

## 5) OpenAPI

Apri `openapi.yaml` in Swagger UI / Postman per testare tutti gli endpoint.
