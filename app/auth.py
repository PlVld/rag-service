from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from app.core.config import settings
import secrets

# --- Bearer токен для REST API и MCP ---
scheme = HTTPBearer(auto_error=False)

# Валидный API-ключ из настроек
VALID_API_KEY = settings.rag_service_api_key

async def verify_api_key(bearer_token: str = Security(scheme)):
    """
    Dependency-функция для проверки Bearer токена (Authorization: Bearer <api_key>).
    Используется для авторизации REST API и MCP endpoints.
    """
    if not bearer_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide Bearer token in Authorization header.",
        )

    # Проверяем Bearer токен
    if not secrets.compare_digest(bearer_token.credentials, VALID_API_KEY):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return True
