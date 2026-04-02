from pathlib import Path
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, TypedDict
from config.config import config

sys.path.append(str(Path(__file__).parent.parent))
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.v1.routes import router as api_router
from db.database import db
from aop_logging import AOPLoggingMiddleware, RequestTimingMiddleware
from aop_logging import get_aop_logger


class UserContext(TypedDict, total=False):
    user_id: str
    roles: list[str]
    scope: list[str]
    username: str


class Context(TypedDict):
    thread_id: str
    user: UserContext


def extract_user_context(req: Request) -> Optional[Dict[str, Any]]:
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
    logger = get_aop_logger().logger
    logger.info("Starting Agent Backend Service...")
    await db.init_db(run_migrations=True)
    yield
    logger.info("Shutting down Agent Backend Service...")
    await db.close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agent Framework Service",
        description="Generalized LangGraph Agent with Graph Registry Integration",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(AOPLoggingMiddleware)
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1/agent")
    return app


app = create_app()


@app.get("/api/auth/me")
async def get_current_user(req: Request):
    user = extract_user_context(req)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, **user}


if __name__ == "__main__":
    import uvicorn

    host = config.server.host
    port = config.server.port
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=config.server.app_env == "development",
    )
