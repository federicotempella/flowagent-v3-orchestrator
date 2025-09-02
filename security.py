# security.py
import ipaddress
from fastapi import Request, HTTPException

ALLOWED_CIDRS = [
    "0.0.0.0/0",  # <-- restringi in produzione
]

def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host

def _guard_ip_from_request(request: Request):
    client_ip = _client_ip(request)
    ip = ipaddress.ip_address(client_ip)
    for cidr in ALLOWED_CIDRS:
        if ip in ipaddress.ip_network(cidr):
            return
    raise HTTPException(status_code=403, detail="Forbidden IP")

# --- CENTRALIZZATI ---
APPROVAL_HEADER = "X-Connector-Approved"

def ensure_auth(authorization: str | None) -> None:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization")

def approval_gate(consequential: bool, approved: str | None) -> None:
    if consequential and (approved or "").lower() not in {"1","true","yes"}:
        raise HTTPException(status_code=428, detail="Approval required")

