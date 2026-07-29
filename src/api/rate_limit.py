import time
from collections import defaultdict

from fastapi import HTTPException, Request

_request_log: dict[str, list[float]] = defaultdict(list)


def rate_limiter(max_calls: int, window_seconds: int):
    """Per-IP, per-endpoint sliding window limit. In-memory, so it resets on restart
    and only limits a single process — fine for a single-instance deployment."""

    def dependency(request: Request) -> None:
        key = f"{request.url.path}:{request.client.host if request.client else 'unknown'}"
        now = time.monotonic()
        window_start = now - window_seconds

        recent_calls = [t for t in _request_log[key] if t > window_start]
        if len(recent_calls) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {max_calls} requests per {window_seconds}s on this endpoint.",
            )
        recent_calls.append(now)
        _request_log[key] = recent_calls

    return dependency
