import asyncio
import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
from dotenv import load_dotenv

load_dotenv()
engine: Optional[AsyncEngine] = None
async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


async def run_alembic_migrations() -> None:
    from alembic.config import Config
    from alembic import command

    def _run_migrations():
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        alembic_ini_path = os.path.join(project_root, "alembic.ini")
        if alembic_ini_path:
            alembic_cfg = Config(alembic_ini_path)
            alembic_cfg.set_main_option(
                "script_location", os.path.join(project_root, "alembic")
            )
            command.upgrade(alembic_cfg, "head")
        else:
            raise RuntimeError(f"Alembic config not found: {alembic_ini_path}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_migrations)


async def init_db(
    pool_size: int = 5,
    max_overflow: int = 10,
    run_migrations: bool = True,
) -> AsyncEngine:
    global engine, async_session_factory
    if engine is not None:
        return engine
    db_url = get_database_url()
    engine = create_async_engine(
        db_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        poolclass=AsyncAdaptedQueuePool,
        echo=False,
    )
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    if run_migrations:
        try:
            await run_alembic_migrations()
        except Exception:
            pass
    return engine


async def get_engine() -> Optional[AsyncEngine]:
    return engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        await init_db()
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session_context() -> AsyncSession:
    if async_session_factory is None:
        await init_db()
    session = async_session_factory()
    return session


async def close_db() -> None:
    global engine, async_session_factory
    if engine:
        await engine.dispose()
        engine = None
        async_session_factory = None
