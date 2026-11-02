"""API Key authentication dependency for FastAPI routes."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate incoming request header against configured API key."""
    # If no key configured, bypass verification
    if not settings.api_key:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API key",
        )

    return api_key

