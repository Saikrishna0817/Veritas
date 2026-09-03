"""API router composition.

We keep REST endpoints under `/api/v1/*` and WebSocket endpoints at `/ws/*`.
"""

from fastapi import APIRouter
from fastapi import Depends

from app.core.security import require_user

from app.api.routes.auth import router as auth_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.models import router as models_router
from app.api.routes.reports import router as reports_router
from app.api.routes.upload import router as upload_router
from app.api.routes.websocket import router as websocket_router


api_router = APIRouter()
ws_router = APIRouter()

# REST routes
api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(upload_router, tags=["upload"], dependencies=[Depends(require_user)])
api_router.include_router(models_router, tags=["models"], dependencies=[Depends(require_user)])
api_router.include_router(datasets_router, tags=["datasets"], dependencies=[Depends(require_user)])
api_router.include_router(reports_router, tags=["reports"], dependencies=[Depends(require_user)])
api_router.include_router(jobs_router, tags=["jobs"], dependencies=[Depends(require_user)])

# WebSocket routes (no `/api/v1` prefix)
ws_router.include_router(websocket_router, tags=["websocket"])

# Backwards-compat export
router = api_router
