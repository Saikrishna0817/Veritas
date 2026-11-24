"""Error bodies: ``{"detail": ..., "code": "<snake_case>"}``. ``detail`` is unchanged from before, ``code`` is stable."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)

_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
          409: "conflict", 410: "gone", 413: "payload_too_large", 415: "unsupported_media_type",
          422: "validation_error", 429: "rate_limited", 500: "internal_error", 503: "unavailable"}


class ApiError(HTTPException):
    """HTTPException with a specific ``code`` (use where the status alone is ambiguous, e.g. 429 quota vs rate limit)."""

    def __init__(self, status_code: int, detail, code: str, headers: dict | None = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code


def code_for(status: int) -> str:
    return _CODES.get(status, "client_error" if status < 500 else "server_error")


def install(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        body = {"detail": exc.detail, "code": getattr(exc, "code", None) or code_for(exc.status_code)}
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": jsonable_encoder(exc.errors()), "code": "validation_error"}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal Server Error", "code": "internal_error"}, status_code=500)
