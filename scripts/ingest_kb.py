import os, json, argparse, base64, re, hashlib
from pathlib import Path
import numpy as np

try:
    import faiss
except Exception as e:
    raise SystemExit("⚠️  Install faiss-cpu (pip install faiss-cpu)")

try:
    import tiktoken
except Exception as e:
    raise SystemExit("⚠️  Install tiktoken (pip install tiktoken)")

from typing import List, Dict, Tuple

# Optional parsers
def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_docx(p: Path) -> str:
    try:
        import docx
    except:
        raise SystemExit("⚠️  Install python-docx to read .docx")
    d = docx.Document(str(p))
    return "\n".join([para.text for para in d.paragraphs])

def read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except:
        raise SystemExit("⚠️  Install pypdf to read .pdf")
    reader = PdfReader(str(p))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt": return read_txt(path)
    if ext == ".md": return read_md(path)
    if ext == ".docx": return read_docx(path)
    if ext == ".pdf": return read_pdf(path)
    raise SystemExit(f"Formato non supportato: {ext} ({path})")

def chunk_text(text: str, max_tokens: int = 900, overlap: int = 120) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        chunk = toks[i:i+max_tokens]
        chunks.append(enc.decode(chunk))
        i += max_tokens - overlap
    return [c.strip() for c in chunks if c.strip()]

# Embeddings (OpenAI)
def embed_texts(texts: List[str]) -> np.ndarray:
    import os
    import httpx
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("⚠️  Set OPENAI_API_KEY in env")
    url = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-small"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    # Batch in groups to avoid payload too large
    vecs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        payload = {"input": batch, "model": model}
        r = httpx.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        for item in data["data"]:
            vecs.append(np.array(item["embedding"], dtype="float32"))
    return np.vstack(vecs)

def build_index(chunks: List[str], meta: List[Dict], out_dir: Path):
    vecs = embed_texts(chunks)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine
    faiss.normalize_L2(vecs)
    index.add(vecs)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "meta.json","w",encoding="utf-8") as f:
        json.dump({"chunks": chunks, "meta": meta}, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved index to {out_dir} (vectors={len(chunks)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Folder with raw docs")
    ap.add_argument("--out", required=True, help="Output index folder")
    ap.add_argument("--industry", default="", help="Default industry tag")
    ap.add_argument("--role", default="", help="Default role tag")
    ap.add_argument("--lang", default="it", help="Language tag")
    args = ap.parse_args()

    raw_dir = Path(args.path)
    out_dir = Path(args.out)

    files = [p for p in raw_dir.glob("**/*") if p.is_file() and p.suffix.lower() in [".txt",".md",".docx",".pdf"]]
    if not files:
        print("Nessun file trovato in", raw_dir)
        return

    all_chunks, all_meta = [], []
    for p in files:
        print("• Parsing", p.name)
        text = load_text(p)
        chs = chunk_text(text)
        meta = [{
            "source_file": p.name,
            "industry": args.industry,
            "role": args.role,
            "lang": args.lang,
            "tags": []
        } for _ in chs]
        all_chunks.extend(chs)
        all_meta.extend(meta)

    build_index(all_chunks, all_meta, out_dir)

if __name__ == "__main__":
    main()
