import os
from pathlib import Path
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, TypedDict
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.v1.routes import router as api_router

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))


class UserContext(TypedDict, total=False):
    """User context propagated from auth layer (Keycloak/JWT)."""

    user_id: str
    roles: list[str]
    scope: list[str]
    username: str


class Context(TypedDict):
    """LangGraph runtime context."""

    thread_id: str
    user: UserContext


def extract_user_context(req: Request) -> Optional[Dict[str, Any]]:
    """Extract user context from request headers.

    Extracts the following headers:
    - X-User-Id (required for authentication)
    - X-Username
    - X-User-Roles (comma-separated)
    - X-User-Scope (space-separated)
    - Authorization token (Bearer token)

    Returns:
        Optional[Dict[str, Any]]: User context dict or None if no X-User-Id
    """
    user_id = req.headers.get("X-User-Id")
    if user_id:
        roles_raw = req.headers.get("X-User-Roles", "")
        scope_raw = req.headers.get("X-User-Scope", "")
        token = req.headers.get("Authorization", "").replace("Bearer ", "")
        return {
            "user_id": user_id,
            "username": req.headers.get("X-Username"),
            "roles": roles_raw.split(",") if roles_raw else [],
            "scope": scope_raw.split(" ") if scope_raw else [],
            "token": token,
        }
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown tasks."""
    # Startup
    print("Starting Agent Backend Service...")
    yield
    # Shutdown
    print("Shutting down Agent Backend Service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agent Framework Service",
        description="Generalized LangGraph Agent with Graph Registry Integration",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1/agent")

    return app


app = create_app()


@app.get("/api/auth/me")
async def get_current_user(req: Request):
    """Get current authenticated user.

    Returns user context if authenticated, otherwise returns
    {"authenticated": False}.
    """
    user = extract_user_context(req)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, **user}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("APP_ENV") == "development",
    )
