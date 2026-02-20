"""Auth routes (lightweight stubs for this hackathon build)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/auth/me")
async def me():
    return {"user": {"id": "demo", "name": "Demo User", "role": "analyst"}}

