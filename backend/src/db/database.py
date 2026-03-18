import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

from config.config import config


class Database:
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._lock = asyncio.Lock()

    def get_database_url(self) -> str:
        return config.database.url

    async def init_db(
        self,
        pool_size: int = 5,
        max_overflow: int = 10,
        run_migrations: bool = True,
    ) -> None:
        async with self._lock:
            if self.engine is not None:
                return

            self.engine = create_async_engine(
                config.database.url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                poolclass=AsyncAdaptedQueuePool,
                echo=False,
            )
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            if run_migrations:
                await self._run_migrations()

    async def _run_migrations(self) -> None:
        from alembic import command
        from alembic.config import Config

        def _run():
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            alembic_ini_path = os.path.join(project_root, "alembic.ini")

            if not os.path.exists(alembic_ini_path):
                raise RuntimeError(f"Alembic config not found: {alembic_ini_path}")

            alembic_cfg = Config(alembic_ini_path)
            alembic_cfg.set_main_option(
                "script_location", os.path.join(project_root, "alembic")
            )
            command.upgrade(alembic_cfg, "head")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialised. Call db.init() first.")

        async with self._session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    raise

    async def close_db(self) -> None:
        async with self._lock:
            if self.engine is not None:
                await self.engine.dispose()
                self.engine = None
                self._session_factory = None


db = Database()
