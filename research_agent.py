# research_agent.py
import os
import re, hashlib, time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlparse

import requests

def _maybe_guard(provider: str):
    # tenta di usare i guard del main app, se disponibili
    try:
        from app import _guard_budget  # nel tuo corretta-app.py
        _guard_budget(provider)
    except Exception:
        pass

# === CACHE & PATHS (env-aware, nessun hardcode .cache) ===
CACHE_DIR     = Path(os.getenv("CACHE_DIR", "/tmp/flowagent-cache"))
KB_DOCS_DIR   = Path(os.getenv("KB_DOCS_DIR", "kb/raw"))

# TTL configurabili via ENV (fallback sicuri)
KB_QCACHE_TTL = int(os.getenv("KB_QCACHE_TTL", "86400"))     # 24h
URL_TXT_TTL   = int(os.getenv("URL_TXT_TTL", "2592000"))     # 30 giorni

# directory di cache
KB_QCACHE_DIR = CACHE_DIR / "kb_query"
TEXT_CACHE    = CACHE_DIR / "url_text"

# creazione cartelle
KB_QCACHE_DIR.mkdir(parents=True, exist_ok=True)
TEXT_CACHE.mkdir(parents=True, exist_ok=True)

# [research_agent.py] --- costanti/cache (in alto, vicino alle altre) ---

def _kb_qcache_path(q: str) -> Path:
    h = hashlib.sha1(q.encode("utf-8")).hexdigest()
    return KB_QCACHE_DIR / f"{h}.json"

def _kb_qcache_load(q: str) -> Optional[dict]:
    p = _kb_qcache_path(q)
    if not p.exists(): return None
    if time.time() - p.stat().st_mtime > KB_QCACHE_TTL:
        try: p.unlink()
        except: pass
        return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return None

def _kb_qcache_save(q: str, obj: dict) -> None:
    try: _kb_qcache_path(q).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except: pass

# [research_agent.py] --- TF-IDF KB search live ---
def kb_search_tfidf(query: str, top_k: int = 5) -> List[dict]:
    """
    Ricerca nei file di KB (docx/txt/md/pdf se sidecar .txt presente) usando TF-IDF.
    Ritorna: [{title, snippet, url(path), score}]
    """
    if not query or not KB_DOCS_DIR.exists():
        return []

    # cache query
    qnorm = query.strip().lower()
    cached = _kb_qcache_load(qnorm)
    if cached:
        return cached.get("results", [])

    docs, titles, paths = [], [], []
    for p in KB_DOCS_DIR.glob("**/*"):
        if not p.is_file(): continue
        if p.suffix.lower() in (".docx",".md",".txt",".pdf"):
            # usa sidecar .txt se esiste, altrimenti prova a leggere testo grezzo
            side = p.with_suffix(".txt")
            try:
                txt = side.read_text(encoding="utf-8") if side.exists() else p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = ""
            if txt.strip():
                docs.append(txt)
                titles.append(p.stem)
                paths.append(str(p))

    if not docs:
        return []

    # TF-IDF + cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vect.fit_transform(docs)
    qv = vect.transform([qnorm])
    sims = linear_kernel(qv, X).ravel()

    # top-k
    idx = sims.argsort()[::-1][:max(1, int(top_k or 5))]
    out = []
    for i in idx:
        score = float(sims[i])
        blob = docs[i]
        snip = (blob[:220] + "…") if len(blob) > 220 else blob
        out.append({"title": titles[i], "snippet": snip, "url": paths[i], "score": round(score, 4)})

    _kb_qcache_save(qnorm, {"results": out, "ts": time.time()})
    return out


@dataclass
class ResearchOptions:
    mode: str = "kb_only"      # "off" | "kb_only" | "web"
    max_sources: int = 8
    seeds: list[str] = field(default_factory=list)
    language: str = "it"

class ResearchAgent:
    """
    Ricerca web minimale con fallback chain.
    Ad oggi implementa SEARXNG se presente SEARXNG_URL.
    Altri provider sono placeholder (ritornano []) per non rompere i flussi.
    """
    def __init__(
        self,
        provider: str = "router",
        budgets: Optional[Dict[str, int]] = None,
        fallback_chain: Optional[List[str]] = None,
        daily_cap: int = 0,
        degrade_to: str = "kb_only",
    ):
        self.provider = (provider or "router").lower()
        self.budgets = budgets or {}
        self.fallback_chain = [p.strip().lower() for p in (fallback_chain or ["searxng"])]
        self.daily_cap = int(daily_cap or 0)
        self.degrade_to = degrade_to

        # configurazioni provider
        self.searxng_url = os.getenv("SEARXNG_URL", "").rstrip("/")


    # ------------------ provider implementations ------------------

    def _search_searxng(self, q: str, lang: str = "it", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Chiama /search di SearxNG in formato JSON.
        Richiede SEARXNG_URL nell'env. Se non presente → [].
        """
        if not self.searxng_url:
            return []
        params = {
            "q": q,
            "language": lang or "it",
            "format": "json",
            "safesearch": 1,
            "categories": "general",
            "time_range": "",
        }
        url = f"{self.searxng_url}/search?{urlencode(params)}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results = []
        for r in (data.get("results") or []):
            item = {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content") or r.get("snippet"),
                "source": "searxng",
            }
            if item["title"] or item["url"]:
                results.append(item)
            if len(results) >= limit:
                break
        return results

    # ------------------ router ------------------

    def _search_with(self, provider: str, q: str, lang: str, limit: int) -> List[Dict[str, Any]]:
        _maybe_guard(provider)
        if provider == "searxng":
            return self._search_searxng(q, lang=lang, limit=limit)

        # placeholder per altri provider (tavily/serper/serpapi/brave) se vorrai implementarli
        return []

    def run(self, queries: List[str], mode: str = "web", max_sources: int = 5, lang: str = "it") -> List[Dict[str, Any]]:
        """
        Esegue la ricerca su tutte le query, usando:
        - provider singolo (se self.provider != "router"), oppure
        - fallback_chain se provider == "router".
        Deduplica per URL/titolo e limita a max_sources.
        """
        if not queries:
            return []

        queries = [q for q in queries if q]
        results: List[Dict[str, Any]] = []

        providers: List[str]
        if self.provider == "router":
            providers = self.fallback_chain or ["searxng"]
        else:
            providers = [self.provider]

        for q in queries:
            for p in providers:
                _maybe_guard(p)
                hits = self._search_with(p, q, lang=lang, limit=max_sources)
                for h in hits:
                    h["via"] = p
                results.extend(hits)
                if len(results) >= max_sources:
                    break
            if len(results) >= max_sources:
                break

        # dedup e cut
        uniq, seen = [], set()
        for r in results:
            key = r.get("url") or r.get("title")
            if key and key not in seen:
                uniq.append(r)
                seen.add(key)
            if len(uniq) >= max_sources:
                break
        return uniq

URL_RE = re.compile(r"https?://[^\s)>\]\"'}]+", re.I)

def _extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(URL_RE, text)
    # pulizia minimale (niente ) o , finali)
    clean = []
    for u in urls:
        u = u.rstrip(").,;\"'›」」》]")
        clean.append(u)
    # dedup mantenendo l'ordine
    seen = set()
    out = []
    for u in clean:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# retrocompatibilità se altrove importi singolare
_extract_url = _extract_urls

DEFAULT_TIMEOUT = 12


def _fetch_text_from_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "FlowAgent/1.0"})
        r.raise_for_status()
        # best-effort: HTML -> testo
        txt = r.text
        # strip html grezzo
        txt = re.sub(r"(?is)<script.*?</script>", " ", txt)
        txt = re.sub(r"(?is)<style.*?</style>", " ", txt)
        txt = re.sub(r"(?is)<[^>]+>", " ", txt)
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""

# --- ADD HERE: research_enrich_contacts (research_agent.py) ---
def _guess_company_homepage(company: str) -> str:
    # euristica minimale + opzione Clearbit-like futura
    company = (company or "").strip().lower()
    if not company:
        return ""
    for tld in (".com", ".eu", ".it"):
        candidate = f"https://www.{re.sub(r'[^a-z0-9]+', '', company)}{tld}"
        try:
            import requests
            r = requests.get(candidate, timeout=8, allow_redirects=True)
            if r.status_code < 400:
                return r.url
        except Exception:
            pass
    return ""


_FEATURE_SYNONYMS = {
  "edi": [r"\bedi\b", r"electronic\s+data\s+interchange", r"\bansi[- ]?x12\b", r"\bedifact\b"],
  "api": [r"\bapi\b", r"\brest(\b|ful)", r"\bgraphql\b", r"\bswagger\b", r"\bopenapi\b"],
  "sap": [r"\bsap\b", r"\bs4[- ]?hana\b", r"\bnetweaver\b", r"\bariba\b"],
  "oracle": [r"\boracle\b", r"\boci\b", r"\boracle\s+fusion\b", r"\boracle\s+erp\b"],
  "integration": [r"\bintegration\b", r"\besb\b", r"\betl\b", r"\bintegration\s+platform\b"],
  "message_queue": [r"\bkafka\b", r"\brabbitmq\b", r"\bevent[- ]?hub\b", r"\bpub[- ]?sub\b"],
  "modern_data": [r"\bdatabricks\b", r"\bsnowflake\b", r"\bdelta\s+lake\b"],
  "security": [r"\bsso\b", r"\boauth2\b", r"\bsaml\b", r"\biam\b"],
  "cloud": [r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bkubernetes\b", r"\bk8s\b"],
  "hiring_signal": [r"\bhiring\b", r"\bwe're\s+Hiring\b", r"\bposizioni\s+aperte\b", r"\bjob\s+opening\b"],
  "erp_migration": [r"\bmigration\b", r"\bmigrazion[ei]\b", r"\bupgrade\b", r"\brollout\b"],
}

def research_enrich_contacts(contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich: homepage, possibile LinkedIn, estrazione keyword/trigger base."""
    out = []
    for c in contacts or []:
        name = c.get("name") or f"{c.get('first_name','')} {c.get('last_name','')}".strip()
        company = c.get("company") or ""
        li = c.get("linkedin_url") or ""
        homepage = c.get("company_website") or _guess_company_homepage(company)
        text_blob = ""

        # fetch homepage best-effort
        if homepage:
            txt = _fetch_text_from_url(homepage, timeout=8)
            text_blob += " " + txt

        # pattern basic per trigger
        triggers = []
        for k, pats in _FEATURE_SYNONYMS.items():
            for p in pats:
                if re.search(p, text_blob, flags=re.I):
                    triggers.append(k)
                    break

        out.append({
            **c,
            "display_name": name,            # <— aggiungi
            "linkedin_url": li or c.get("linkedin_url"),
            "company_website": homepage,
            "enrich": {"text_len": len(text_blob), "triggers": sorted(set(triggers))}
        })

    return out

def research_extract(raw_text: str) -> Dict[str, Any]:
    """Estrae URL e (opzionale) scarica il contenuto per arricchire la KB."""
    urls = _extract_urls(raw_text or "")
    fetched = []
    for u in urls:
        h = hashlib.sha1(u.encode()).hexdigest()
        fp = TEXT_CACHE / f"{h}.txt"
        if fp.exists():
            age = time.time() - fp.stat().st_mtime
            if age <= URL_TXT_TTL:
                fetched.append({"url": u, "path": str(fp)})
                continue
            # scaduto → rimuovi e riscarica
            try: fp.unlink()
            except: pass

        body = _fetch_text_from_url(u)
# ============================ PUBLIC API ============================
# --- API pubblica (usata da app.py) ---
# ============================ PUBLIC API ============================
def _effective_research_mode(opts: Optional[ResearchOptions]) -> ResearchOptions:
    if not opts:
        return ResearchOptions()
    if opts.mode not in {"off","kb_only","web"}:
        opts.mode = "kb_only"
    opts.max_sources = max(1, min(int(opts.max_sources or 8), 50))
    return opts


def _perform_research(req) -> List[dict]:
    """
    Ricerca KB + Web (tramite ResearchAgent) in base a req.research.mode, con cache.
    - mode: "off" | "kb_only" | "web"
    - seeds: lista di query dal client
    - max_sources: limitiamo i risultati finali
    - lang: usa req.language se presente
    """
    # --- opzioni/lingua ---
    lang = getattr(req, "language", None) or "it"
    ropt = getattr(req, "research", None)
    if isinstance(ropt, dict):
        ropt = ResearchOptions(**{**ResearchOptions().__dict__, **ropt})
    ropt = _effective_research_mode(ropt)
    mode = ropt.mode
    seeds = list(ropt.seeds or [])
    max_sources = int(ropt.max_sources or 5)

    # --- query seed + base ---
    base_queries: List[str] = []
    if getattr(req, "persona_id", None):
        base_queries.append(f"buyer persona {req.persona_id}")
    if hasattr(req, "sequence_type"):
        base_queries.append(f"{req.sequence_type} cold outreach best practices")
    queries = [q for q in (seeds + base_queries) if q] or ["EDI API onboarding retail"]

    # --- cache (usa funzioni esposte da app.py se presenti) ---
    try:
        from app import research_cache_get, research_cache_set  # type: ignore
    except Exception:
        research_cache_get = lambda key: None
        research_cache_set = lambda key, value: None

    cache_key = {
        "mode": mode,
        "seeds": tuple(seeds),
        "max_sources": max_sources,
        "lang": lang,
        "queries": tuple(queries),
    }
    cached = research_cache_get(cache_key)
    if cached is not None:
        return cached

    # --- KB: TF-IDF (preferito) con fallback alla vecchia _do_kb_search ---
    # --- KB: TF-IDF (preferito) con fallback
    kb_hits: List[Dict[str, Any]] = []
    if mode in {"kb_only", "web"} and queries:
        q_join = " ".join(queries)
        try:
            kb_hits = kb_search_tfidf(q_join, top_k=max_sources)   # usa la funzione locale del file
        except Exception:
            # fallback alla vecchia _do_kb_search dell'app, se esiste
            try:
                from app import _do_kb_search  # opzionale
                for q in queries:
                    try:
                        kb_hits.extend(_do_kb_search(q, top_k=3))
                    except Exception:
                        pass
            except Exception:
                kb_hits = []

    # --- WEB: solo se richiesto ---
    web_hits: List[Dict[str, Any]] = []
    if mode == "web":
        agent = ResearchAgent(
            provider=os.getenv("RESEARCH_PROVIDER", "router"),
            budgets={
                "tavily": int(os.getenv("RESEARCH_BUDGET_TAVILY", "0") or 0),
                "serper": int(os.getenv("RESEARCH_BUDGET_SERPER", "0") or 0),
                "serpapi": int(os.getenv("RESEARCH_BUDGET_SERPAPI", "0") or 0),
                "brave": int(os.getenv("RESEARCH_BUDGET_BRAVE", "0") or 0),
                "searxng": 10**9,
            },
            fallback_chain=os.getenv("RESEARCH_FALLBACKS", "searxng").split(","),
            daily_cap=int(os.getenv("RESEARCH_BUDGET_DAY", "0") or 0),
            degrade_to=os.getenv("RESEARCH_DEGRADE_TO", "kb_only"),
        )
        web_hits = agent.run(queries=queries, mode="web", max_sources=max_sources, lang=lang)

    # --- fusione + dedup + cut ---
    hits = (kb_hits + web_hits) if mode == "web" else kb_hits
    uniq, seen = [], set()
    for r in hits:
        key = r.get("url") or r.get("title")
        if key and key not in seen:
            uniq.append(r); seen.add(key)
        if len(uniq) >= max_sources:
            break

    try:
        research_cache_set(cache_key, uniq)
    except Exception:
        pass

    return uniq
