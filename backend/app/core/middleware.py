"""Custom ASGI middleware for the AI Trust Forensics Platform.

Provides:
- RequestIDMiddleware  — generates X-Request-ID for every request and injects it
                         into the logging context and response headers.
- SecurityHeadersMiddleware — adds strict HTTP security headers to every response,
                               including Content-Security-Policy (required by
                               docs/decisions.md for the localStorage JWT strategy).
"""

from __future__ import annotations

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate a unique request ID and make it available in logs and responses.

    The ID is injected into:
      - The `request_id_var` ContextVar (picked up by _RequestIDFilter in logging)
      - The request state as `request.state.request_id`
      - The `X-Request-ID` response header
    """

    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Honour an incoming ID if provided (e.g. from an upstream proxy)
        request_id = request.headers.get(self.HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.HEADER] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add strict HTTP security headers to every response.

    CSP is configured for the analyst-facing SPA (same-origin, no inline scripts
    beyond what React/Vite requires, no external resources except Google Fonts).
    Adjust `_CSP` when upgrading the frontend or adding CDN assets.
    """

    # Content-Security-Policy for the internal analyst deployment.
    # - default-src 'self'      : only same-origin by default
    # - script-src 'self'       : no inline scripts (XSS mitigation for JWT)
    # - style-src 'self' fonts  : Tailwind + Google Fonts
    # - font-src 'self' fonts   : Google Fonts woff2
    # - img-src 'self' data:    : inline SVG data URIs (Recharts)
    # - connect-src 'self' ws:  : WebSocket to same origin
    # - frame-ancestors 'none'  : clickjacking prevention
    # - object-src 'none'       : no plugins
    _CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:; "
        "frame-ancestors 'none'; "
        "object-src 'none';"
    )

    _HEADERS: dict[str, str] = {
        "Content-Security-Policy": _CSP,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        for header, value in self._HEADERS.items():
            response.headers[header] = value
        return response
