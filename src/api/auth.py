import os

from fastapi import Header, HTTPException


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    """Guards write/expensive operations (re-ingestion) that only the owner should trigger.
    Fails closed: if ADMIN_API_KEY isn't configured, the endpoint is disabled entirely,
    not silently open."""
    expected = os.environ.get("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="This operation is disabled (no admin key configured).")
    if x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key header.")
