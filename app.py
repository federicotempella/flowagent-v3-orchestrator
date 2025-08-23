import os
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date, timedelta, timezone
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
load_dotenv()


APP_ENV = os.getenv("APP_ENV", "test").lower()
APPROVAL_HEADER = "X-Connector-Approved"
IDEMPOTENCY_HEADER = "Idempotency-Key"
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "")

TRUTHY = {"true", "yes", "1", "on"}

docs_on = os.getenv("DOCS_ENABLED", "true").lower() in ("1","true","yes","on")

app = FastAPI(
    title="Flowagent V3 Orchestrator",
    version="1.1.0",
    docs_url="/docs" if docs_on else None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

@app.get("/", include_in_schema=False)
def root_banner():
    app_name = "Flowagent V3 Orchestrator"
    app_ver  = os.getenv("APP_VERSION", "1.1.0")
    build_sha = os.getenv("BUILD_SHA", "render")
    app_env  = APP_ENV
    docs_url = "/docs" if docs_on else "/openapi.json"

    html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>{app_name}</title>
    <style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b1020;color:#e6e8ef}}
    .wrap{{max-width:820px;margin:6rem auto;padding:2rem;text-align:center}}
    .card{{background:#121936;border:1px solid #222a51;border-radius:14px;padding:2rem}}
    h1{{margin:0 0 1rem;font-size:1.8rem}}
    .meta{{opacity:.8;margin:.5rem 0 1.2rem}}
    .btns a{{display:inline-block;margin:.25rem;padding:.7rem 1rem;border-radius:10px;text-decoration:none;border:1px solid #3b4799}}
    .btns a.primary{{background:#2b3bbb;color:white;border-color:#2b3bbb}}
    .btns a:hover{{filter:brightness(1.08)}}</style></head>
    <body><div class="wrap"><div class="card">
    <h1>{app_name}</h1>
    <div class="meta">Env: <code>{app_env}</code> · Versione: <code>{app_ver}</code> · Build: <code>{build_sha[:7]}</code></div>
    <div class="btns"><a class="primary" href="{docs_url}">📚 API Docs</a>
    <a href="/openapi.json">🧩 OpenAPI</a> <a href="/health">✅ Health</a></div>
    </div></div></body></html>
    """
    return HTMLResponse(html)

@app.get("/ping", include_in_schema=False)
def ping():
    return {"service":"flowagent-v3","status":"ok","docs":"/docs"}

# PRIMA di app.add_middleware
# .env: ALLOW_ORIGINS=https://chatgpt.com,https://chat.openai.com,https://flowagent-v3-orchestrator.onrender.com
raw_origins = os.getenv(
    "ALLOW_ORIGINS",
    "https://chatgpt.com,https://chat.openai.com,https://flowagent-v3-orchestrator.onrender.com"
)

ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,            # richiede che NON ci sia "*"
    allow_methods=["*"],
    allow_headers=["*"],               # include Authorization, X-Connector-Approved
    expose_headers=["X-Connector-Approved", "Idempotency-Key"],
    max_age=3600,
)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()+ "Z"}

# ---------- Types ----------
Mode = Literal["AE", "SDR"]
Level = Literal["Beginner", "Intermediate", "Advanced"]
SequenceType = Literal["with_inmail", "without_inmail"]
Framework = Literal["Auto", "TIPPS", "TIPPS+COI", "Poke", "HarrisNEAT", "NEAT_structure", "ShowMeYouKnowMe", "BoldInsight", "Challenger"]
Channel = Literal["email", "linkedin_dm", "inmail", "voice_note", "video_dm"]

class Trigger(BaseModel):
    manual_priority: bool = False
    personal: Optional[str] = None
    competitor: Optional[List[str]] = None
    erp: Optional[List[str]] = None
    company_signal: Optional[str] = None
    linkedin_signal: Optional[str] = None
    read_inferred: Optional[str] = None

class Contact(BaseModel):
    name: str
    role: str
    company: str
    lang: str = "it"
    email: Optional[EmailStr] = None
    company_id: Optional[str] = None

class CalendarRules(BaseModel):
    no_weekend: bool = True
    holiday_calendar: Optional[str] = None
    working_hours: Optional[dict] = None

class Message(BaseModel):
    channel: Channel
    step: str
    variant: Literal["A", "B", "C"] = "A"
    subject: Optional[str] = None
    text: str
    tips: Optional[List[str]] = None

class COI(BaseModel):
    status: Literal["none", "estimated", "computed"] = "none"
    note: Optional[str] = None
    assumptions: Optional[List[str]] = None

class SequenceAction(BaseModel):
    day: int
    action: str

class CalendarEvent(BaseModel):
    date: date
    action: str
    no_weekend_respected: bool = True

class Citation(BaseModel):
    url: str
    title: str
    published_at: Optional[datetime] = None
    source: Optional[str] = None

class EnrichedFact(BaseModel):
    fact: str
    confidence: float = 0.8
    citations: Optional[List[Citation]] = None

class ResearchResult(BaseModel):
    triggers: Optional[dict] = None
    facts: Optional[List[EnrichedFact]] = None
    citations: Optional[List[Citation]] = None

class WhatIUsed(BaseModel):
    personas: Optional[List[str]] = None
    files: Optional[List[str]] = None
    triggers: Optional[List[str]] = None

class StandardOutput(BaseModel):
    messages: Optional[List[Message]] = None
    coi: Optional[COI] = None
    sequence_next: Optional[List[SequenceAction]] = None
    calendar: Optional[List[CalendarEvent]] = None
    labels: Optional[List[str]] = None
    what_i_used: Optional[WhatIUsed] = None
    rationale: Optional[str] = None
    logging: Optional[dict] = None
    research: Optional[ResearchResult] = None

class ComposeStrategy(BaseModel):
    order: list[str] = Field(default_factory=list)
    apply_cleanup: bool = True

class ResearchParams(BaseModel):
    enabled: bool = False
    approval_mode: Literal["auto","manual"] = "manual"
    time_window_days: int = 90
    language: str = "it"
    geo: Optional[str] = None
    types: Optional[List[Literal["news","press_release","careers","techstack","finance","blog"]]] = None
    domains_whitelist: Optional[List[str]] = None
    max_results: int = 12

class GenerateAssetsRequest(BaseModel):
    mode: Mode
    level: Level
    language: str = "it"
    tone: str = "formale"
    persona_id: str
    pain_id: Optional[str] = None
    symptom_id: Optional[str] = None
    kpi_id: Optional[str] = None
    framework_override: Optional[Framework] = None
    frameworks: Optional[List[Framework]] = None
    compose_strategy: Optional[ComposeStrategy] = None
    triggers: Optional[Trigger] = None
    kb_snapshot: Optional[str] = None
    use_assets: Optional[List[Literal["subject","hook","value_prop","proof","cta","objection_handler","ps"]]] = None
    research: Optional[ResearchParams] = None

class GenerateAssetsResponse(BaseModel):
    assets: List[dict]
    what_i_used: Optional[WhatIUsed] = None
    rationale: Optional[str] = None
    logging: Optional[dict] = None

class GenerateSequenceRequest(BaseModel):
    sequence_type: SequenceType
    language: str = "it"
    tone: str = "formale"
    mode: Optional[Mode] = None
    level: Optional[Level] = None
    framework_override: Optional[Framework] = None
    frameworks: Optional[List[Framework]] = None
    compose_strategy: Optional[ComposeStrategy] = None
    use_assets: Optional[List[str]] = None
    ab_test: Optional[dict] = None
    calendar_rules: Optional[CalendarRules] = Field(default_factory=CalendarRules)
    contacts: Optional[List[Contact]] = None
    triggers: Optional[Trigger] = None
    buyer_persona_ids: Optional[List[str]] = None
    research: Optional[ResearchParams] = None



class RankItem(BaseModel):
    id: str
    channel: Channel
    subject: Optional[str] = None
    text: str

class RankRequest(BaseModel):
    objective: Literal["reply_rate","open_rate","clarity"] = "reply_rate"
    items: List[RankItem]

class RankResponse(BaseModel):
    ranked: List[dict]

class ComplianceRequest(BaseModel):
    text: str
    rules: Dict[str, Any] = Field(default_factory=dict)

class ComplianceResponse(BaseModel):
    pass_: bool = Field(alias="pass")
    violations: List[dict] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True

class CalendarBuildRequest(BaseModel):
    start_date: date | None = None
    rules: CalendarRules = Field(default_factory=CalendarRules)          # sempre presente
    base_sequence: list[SequenceAction] = Field(default_factory=list)    # [] se assente
    signals: list[Literal["open","soft_reply","profile_view","site_visit"]] = Field(default_factory=list)

class CalendarBuildResponse(BaseModel):
    calendar: List[CalendarEvent]
    ics: Optional[str] = None

# --- KB typed responses ---
class KBSearchMatch(BaseModel):
    doc_id: str
    score: float
    chunk: str
    metadata: Optional[dict] = None

class KBSearchResponse(BaseModel):
    matches: List[KBSearchMatch]

class KBListItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    lang: Optional[str] = None
    metadata: Optional[dict] = None

class KBListResponse(BaseModel):
    docs: List[KBListItem]

class KBIngestRequest(BaseModel):
    content_type: Literal["docx","pdf","html","md","txt","url"]
    url: Optional[str] = None
    content_base64: Optional[str] = None
    metadata: Optional[dict] = None

class KBIngestResponse(BaseModel):
    doc_id: str
    chunks: int

class KBDeleteResponse(BaseModel):
    ok: bool
    deleted_doc_id: str

class SendEmailRequest(BaseModel):
    provider: Literal["sendgrid","mailgun","smtp","gmail","outlook"]
    to: EmailStr
    from_: EmailStr = Field(alias="from")
    subject: str
    text: str
    html: Optional[str] = None
    variant_id: Optional[str] = None
    campaign_id: Optional[str] = None

    class Config:
     allow_population_by_field_name = True

class SendEmailResponse(BaseModel):
    message_id: str
    provider: str
    queued_at: datetime

class CompanyEvidence(BaseModel):
    company_id: str
    erp: Optional[List[str]] = None
    competitor: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    source: Optional[str] = None

class ManualSignal(BaseModel):
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    type: Literal["profile_view","site_visit","manual_note"]
    meta: Optional[dict] = None

class COIEstimateRequest(BaseModel):
    volume: Optional[int] = None
    error_rate: Optional[float] = None
    avg_order_value: Optional[float] = None
    process_costs: Optional[float] = None
    assumptions: Optional[dict] = None

class COIEstimateResponse(BaseModel):
    status: Literal["estimated","computed"]
    note: str
    assumptions: list[str] = Field(default_factory=list)

class UpsertResponse(BaseModel):
    ok: bool
    stored: dict

class ExportSequenceResponse(BaseModel):
    ok: bool
    format: Literal["csv","json"]
    sequence_id: str

# ---------- Helpers ----------
def ensure_auth(authorization: Optional[str]):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token or token != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- Approval gate (bypassabile in TEST, richiede conferma in PROD) ---

def approval_gate(consequential: bool, approved: str | None):
    """
     Se l'azione è 'consequenziale' e l'ambiente è prod/production,
    richiede conferma esplicita via header X-Connector-Approved.
    In altri ambienti (test/dev/staging) non blocca.
    """
    if not consequential:
        return

    is_prod = APP_ENV in ("prod", "production")
    if is_prod:
        val = (approved or "").strip().lower()
        if val not in ("true", "yes", "1", "on"):
            # 428 = Precondition Required
            raise HTTPException(
                status_code=428,
                detail={
                    "message": "The requested action requires approval",
                    "consequential": True,
                    "env": APP_ENV,
                    "hint": f"Send header {APPROVAL_HEADER}: true to proceed."
                }
            )

def _inline_research(company: Optional[str], req: Optional[ResearchParams]) -> Optional[ResearchResult]:
    if not req or not req.enabled:
        return None
    cites = [Citation(url="https://newsroom.example.com/pr-edi-upgrade", title="PR: upgrade EDI Q4", source="newsroom")]
    facts = [EnrichedFact(fact="Upgrade EDI pianificato nel Q4", confidence=0.82, citations=cites)]
    triggers = {"erp":["SAP"], "company_signal":"Hiring EDI Specialist"}
    return ResearchResult(triggers=triggers, facts=facts, citations=cites)

# ---------- Endpoints ----------

@app.post("/run/generate_assets", response_model=GenerateAssetsResponse)
def generate_assets(
    payload: GenerateAssetsRequest,
    authorization: Optional[str] = Header(None),
    idem_hdr: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    idem_q: Optional[str] = Query(default=None, alias="idempotency_key"),

):
    ensure_auth(authorization)
    idempotency_key = idem_hdr or idem_q
    # TODO: se vuoi, usa effective_idemp per dedup/log

    rr = _inline_research(company=None, req=payload.research)
    assets = []
    for t in (payload.use_assets or ["subject","hook","value_prop","proof","cta"]):
        if t == "subject":
            text = "EDI/API senza carichi extra per l’IT — idea veloce?"
        elif t == "hook":
            text = "Ho visto che state spingendo sull’upgrade EDI; spesso il collo di bottiglia è l’onboarding partner."
        elif t == "value_prop":
            text = "Riduci il tempo medio di integrazione partner da settimane a giorni con connettori pre-validati."
        elif t == "proof":
            text = "Caso retail: -35% ticket EDI e lead time onboarding -50% in 90 giorni."
        elif t == "cta":
            text = "Ha senso uno scambio di 12’ questa settimana per capire fit e tempi?"
        else:
            text = f"Elemento {t} non standardizzato"
        assets.append({"type": t, "text": text, "lang": payload.language, "source_refs": ["Buyer persona - fashion retail.docx","TIPPS framework + The 1-Sentence Email_framework example.docx"]})
    used_triggers = []
    if payload.triggers:
        for k,v in payload.triggers.dict().items():
            if v: used_triggers.append(f"{k}:{v}")
    if rr and rr.triggers and payload.research and payload.research.approval_mode == "auto":
        used_triggers += [f"auto:{k}" for k in rr.triggers.keys()]
    return GenerateAssetsResponse(
        assets=assets,
        what_i_used=WhatIUsed(personas=[payload.persona_id], files=["TIPPS framework + The 1-Sentence Email_framework example.docx"] + ([c.title for c in (rr.citations if rr else [])] or []), triggers=used_triggers or None),
        rationale="Generazione TIPPS con pre-hook personalizzato; ricerca inline applicata" if rr else "Generazione TIPPS con pre-hook personalizzato; nessuna ricerca",
        logging={"prompt_version":"v3.1.0","kb_snapshot": payload.kb_snapshot or "auto"}
    )

@app.post("/run/generate_sequence", response_model=StandardOutput)
def generate_sequence(
    payload: GenerateSequenceRequest,
    authorization: Optional[str] = Header(None),
    idem_hdr: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    idem_q: Optional[str] = Query(default=None, alias="idempotency_key"),
):
    ensure_auth(authorization)
    idempotency_key = idem_hdr or idem_q
    # TODO: opzionale: dedup/lock in base a idempotency_key

    company = payload.contacts[0].company if payload.contacts else None
    rr = _inline_research(company=company, req=payload.research)

    # Build messages (email + linkedin)
    email_text = (
    	"Oggetto: Onboarding EDI senza carichi extra\n\n"
    	"Buongiorno {Nome},\n"
    	"ho notato segnali di upgrade EDI; spesso l’impatto è su ticket e tempi di integrazione.\n"
    	"In un caso retail simile abbiamo ridotto i ticket -35% e il lead time -50% in 90 giorni.\n"
    	"Se utile, posso condividere la check-list in 12 minuti – le va bene mercoledì?\n"
    )
    dm_text = "Spunto rapido su EDI/API – se utile scambiamo 10’ questa settimana per capire fit e tempi."

    msgs = [
    Message(
        channel="email",
        step="1",
        variant="A",
        subject="EDI/API: evitare carichi extra per l'IT",
        text=email_text,
    ),
    Message(
        channel="linkedin_dm",
        step="2",
        variant="A",
        text=dm_text,
    ),
    Message(
        channel="inmail",
        step="3",
        variant="A",
        subject="Richiesta rapida",
        text="Ha senso valutare un check rapido sul flusso EDI?",
    ),
]

    cal = [CalendarEvent(date=date.today(), action="Email Step 1", no_weekend_respected=True)]
    labels = ["Poke-AB"]
    if payload.triggers and payload.triggers.erp:
        labels += [f"CompanyEvidence: {e}" for e in payload.triggers.erp]
    if rr and rr.triggers and payload.research and payload.research.approval_mode == "auto":
        for e in (rr.triggers.get("erp") or []):
            labels.append(f"CompanyEvidence: {e}")

    coi = COI(status="estimated", note="Ritardi EDI 3–5% → rischio €25–40k/anno", assumptions=["Volumi simili a case retail","Ticket medio €20-30"])
    return StandardOutput(
        messages=msgs,
        coi=coi,
        sequence_next=[SequenceAction(day=5, action="InMail bump"), SequenceAction(day=7, action="Follow-up Poke")],
        calendar=cal,
        labels=list(sorted(set(labels))),
        what_i_used=WhatIUsed(personas=payload.buyer_persona_ids, files=["Case studies CPG - Retail.docx"] + ([c.title for c in (rr.citations if rr else [])] or []), triggers=["ERP:"+",".join(payload.triggers.erp)] if payload.triggers and payload.triggers.erp else None),
        rationale="Frameworks compositi; ricerca inline applicata" if rr else "Frameworks compositi; nessuna ricerca",
        logging={"prompt_version":"v3.1.0","kb_snapshot":"auto"},
        research=rr
    )

@app.post("/run/rank", response_model=RankResponse)
def rank(payload: RankRequest, authorization: Optional[str] = Header(None)):
    ensure_auth(authorization)
    ranked = [{"id": it.id, "score": round(0.9 - i*0.07, 3), "reason": f"Coerenza {payload.objective}"} for i,it in enumerate(payload.items)]
    return RankResponse(ranked=ranked)

@app.post("/validate/compliance", response_model=ComplianceResponse)
def compliance_validate(
    payload: ComplianceRequest,
    authorization: Optional[str] = Header(None),
):
    ensure_auth(authorization)

    # Normalizzazioni safe
    text = payload.text or ""
    text_low = text.lower()
    rules = payload.rules or {}
    if not isinstance(rules, dict):
        rules = {}

    # Estrai e pulisci banned_terms
    raw_banned = rules.get("banned_terms", [])
    if not isinstance(raw_banned, list):
        raw_banned = [raw_banned]
    banned_terms = []
    for x in raw_banned:
        try:
            banned_terms.append(str(x))
        except Exception:
            continue

    require_cta = bool(rules.get("require_cta", True))
    anti_jargon = bool(rules.get("anti_jargon", True))
    max_words   = rules.get("max_words", None)

    violations = []

    # 1) CTA presente?
    if require_cta:
        tokens = ("?", "prenot", "disponibile", "chi")
        if not any(tok in text_low for tok in tokens):
            violations.append({"code": "CTA_MISSING", "detail": "Manca una call-to-action chiara"})

    # 2) Anti-jargon
    if anti_jargon and banned_terms:
        for bad in banned_terms:
            if bad.lower() in text_low:
                violations.append({"code": "JARGON", "detail": f"Termine vietato: {bad}"})
                break

    # 3) Lunghezza massima parole
    if max_words is not None:
        try:
            mw = int(max_words)
            words = len(text.split())
            if words > mw:
                violations.append({"code": "TOO_LONG", "detail": f"Testo {words} parole > max {mw}"})
        except Exception:
            # se max_words non è castabile, ignora senza crash
            pass

    return ComplianceResponse(pass_=(len(violations) == 0), violations=violations)

def next_workday(d: date) -> date:
    """Rimanda a lunedì se la data cade nel weekend, senza saltare giorni feriali."""
    while d.weekday() >= 5:  # 5=Sab, 6=Dom
        d += timedelta(days=1)
    return d

@app.post("/calendar/build", response_model=CalendarBuildResponse)
def calendar_build(
    payload: CalendarBuildRequest,
    authorization: Optional[str] = Header(None),
):
    ensure_auth(authorization)

    events = []
    start = payload.start_date if payload.start_date else date.today()
    rules = payload.rules or CalendarRules()

    base = payload.base_sequence or [SequenceAction(day=0, action="Step 1")]
    for sa in base:
        d = start + timedelta(days=sa.day)
        if rules.no_weekend:
            d = next_workday(d)
        events.append(
            CalendarEvent(date=d.isoformat(), action=sa.action, no_weekend_respected=True)
        )

    # (eventuale logica su payload.signals...)

    ics = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//FlowagentV3//EN\nEND:VCALENDAR"
    return CalendarBuildResponse(calendar=events, ics=ics)

@app.post("/kb/ingest", response_model=KBIngestResponse)
def kb_ingest(
    payload: KBIngestRequest,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    approved_q: Optional[str] = Query(default=None, alias="approved"),   # <-- NEW
):
    ensure_auth(authorization)
    approval_gate(True, approved_hdr or approved_q)


    # TODO: integrazione reale parser/indice
    return KBIngestResponse(doc_id="doc_123", chunks=42)

@app.get("/kb/search", response_model=KBSearchResponse)
def kb_search(
    q: str,
    industry: Optional[str] = None,
    role: Optional[str] = None,
    lang: Optional[str] = None,
    top_k: int = 5,
    authorization: Optional[str] = Header(None),
):
    ensure_auth(authorization)
    return KBSearchResponse(matches=[
        KBSearchMatch(
            doc_id="doc_123",
            score=0.82,
            chunk="...TIPPS framework...",
            metadata={"industry": industry, "role": role, "lang": lang}
        )
    ])

@app.get("/kb/list", response_model=KBListResponse)
def kb_list(
    industry: Optional[str] = None,
    role: Optional[str] = None,
    lang: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    ensure_auth(authorization)
    return KBListResponse(docs=[
        KBListItem(
            doc_id="doc_123",
            title="Buyer persona - fashion retail.docx",
            lang="it",
            metadata={"industry":"retail","role":"CIO","tags":["persona"]}
        )
    ])
@app.delete("/kb/doc/{doc_id}", response_model=KBDeleteResponse)
def kb_delete(
    doc_id: str,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    approved_q:  Optional[str] = Query(default=None, alias="approved"),
):
    ensure_auth(authorization)
    # usa SEMPRE l'OR tra header e query:
    approval_gate(True, approved_hdr or approved_q)

    return KBDeleteResponse(ok=True, deleted_doc_id=doc_id)


@app.post("/company/evidence/upsert", response_model=UpsertResponse)
def company_evidence_upsert(
    payload: CompanyEvidence,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    approved_q: Optional[str] = Query(default=None, alias="approved"),
):
    ensure_auth(authorization)
    approval_gate(True, approved_hdr or approved_q)

    labels = []
    for e in (payload.erp or []):
        labels.append(f"CompanyEvidence: {e}")

    stored = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return UpsertResponse(ok=True, stored=stored)

@app.get("/company/evidence/{company_id}", response_model=CompanyEvidence)
def company_evidence_get(
    company_id: str,
    authorization: Optional[str] = Header(None),
):
    ensure_auth(authorization)
    return CompanyEvidence(company_id=company_id, erp=["SAP"], competitor=["X"], tools=["Y"])

@app.post("/signals/record", response_model=UpsertResponse)
def record_signal(
    payload: ManualSignal,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    approved_q: Optional[str] = Query(default=None, alias="approved"),
):
    ensure_auth(authorization)
    approval_gate(True, approved_hdr or approved_q)      # <- azione mutante => richiede conferma in prod

    stored = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return UpsertResponse(ok=True, stored=stored)

@app.post("/coi/estimate", response_model=COIEstimateResponse)
def coi_estimate(payload: COIEstimateRequest, authorization: Optional[str] = Header(None)):
    ensure_auth(authorization)
    return COIEstimateResponse(
        status="estimated",
        note="COI ~ €25–40k/anno",
        assumptions=["5% errori","AOV €120","Process cost €2/ordine"]
    )

class ABPromoteRequest(BaseModel):
    sequence_id: str
    variant_id: str

class ABPromoteResponse(BaseModel):
    ok: bool
    sequence_id: str
    variant_id: str

@app.post("/ab/promote", response_model=ABPromoteResponse)
def ab_promote(
    payload: ABPromoteRequest,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    approved_q: Optional[str] = Query(default=None, alias="approved"),
):
    ensure_auth(authorization)
    approval_gate(True, approved_hdr or approved_q)
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return ABPromoteResponse(ok=True, sequence_id=data["sequence_id"], variant_id=data["variant_id"])

@app.get("/export/sequence/{sequence_id}", response_model=ExportSequenceResponse)
def export_sequence(sequence_id: str, format: str = "json", authorization: Optional[str] = Header(None)):
    ensure_auth(authorization)
    if format == "csv":
        return ExportSequenceResponse(ok=True, format="csv", sequence_id=sequence_id)
    return ExportSequenceResponse(ok=True, format="json", sequence_id=sequence_id)

@app.post("/send/email", response_model=SendEmailResponse)
def send_email(
    payload: SendEmailRequest,
    authorization: Optional[str] = Header(None),
    approved_hdr: Optional[str] = Header(default=None, alias=APPROVAL_HEADER),
    # Query: ?approved=true|yes|1|on
    approved_q: Optional[str] = Query(default=None, alias="approved"),
):
    ensure_auth(authorization)
    approval_gate(True, approved_hdr or approved_q)

    msg_id = f"msg_{int(datetime.now(timezone.utc).timestamp())}"
    queued = datetime.now(timezone.utc)
    return SendEmailResponse(message_id=msg_id, provider=payload.provider, queued_at=queued)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

@app.post("/webhooks/events", include_in_schema=False)
def webhooks (
    payload: dict,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    if not WEBHOOK_SECRET or x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # TODO: gestisci evento
    return {"ok": True}

