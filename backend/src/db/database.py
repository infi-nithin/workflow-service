# import asyncio
# import os
# from typing import AsyncGenerator, Optional
# from sqlalchemy.ext.asyncio import (
#     AsyncEngine,
#     AsyncSession,
#     async_sessionmaker,
#     create_async_engine,
# )
# from sqlalchemy.pool import AsyncAdaptedQueuePool
# from config.config import config

# engine: Optional[AsyncEngine] = None
# async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


# def get_database_url() -> str:
#     return config.database.url


# async def run_alembic_migrations() -> None:
#     from alembic.config import Config
#     from alembic import command

#     def _run_migrations():
#         project_root = os.path.dirname(
#             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         )
#         alembic_ini_path = os.path.join(project_root, "alembic.ini")
#         if alembic_ini_path:
#             alembic_cfg = Config(alembic_ini_path)
#             alembic_cfg.set_main_option(
#                 "script_location", os.path.join(project_root, "alembic")
#             )
#             command.upgrade(alembic_cfg, "head")
#         else:
#             raise RuntimeError(f"Alembic config not found: {alembic_ini_path}")

#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, _run_migrations)


# async def init_db(
#     pool_size: int = 5,
#     max_overflow: int = 10,
#     run_migrations: bool = True,
# ) -> AsyncEngine:
#     global engine, async_session_factory
#     if engine is not None:
#         return engine
#     db_url = get_database_url()
#     engine = create_async_engine(
#         db_url,
#         pool_size=pool_size,
#         max_overflow=max_overflow,
#         poolclass=AsyncAdaptedQueuePool,
#         echo=False,
#     )
#     async_session_factory = async_sessionmaker(
#         engine,
#         class_=AsyncSession,
#         expire_on_commit=False,
#         autoflush=False,
#     )
#     if run_migrations:
#         try:
#             await run_alembic_migrations()
#         except Exception:
#             pass
#     return engine


# async def get_engine() -> Optional[AsyncEngine]:
#     return engine


# async def get_session() -> AsyncGenerator[AsyncSession, None]:
#     if async_session_factory is None:
#         await init_db()
#     async with async_session_factory() as session:
#         try:
#             yield session
#             await session.commit()
#         except Exception:
#             await session.rollback()
#             raise
#         finally:
#             await session.close()


# async def get_session_context() -> AsyncSession:
#     if async_session_factory is None:
#         await init_db()
#     session = async_session_factory()
#     return session


# async def close_db() -> None:
#     global engine, async_session_factory
#     if engine:
#         await engine.dispose()
#         engine = None
#         async_session_factory = None


import asyncio
import logging
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

logger = logging.getLogger(__name__)


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
        logger.info("Database migrations complete")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialised. Call db.init() first.")

        async with self._session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    logger.exception("Session rolled back due to exception")
                    raise

    async def close_db(self) -> None:
        async with self._lock:
            if self.engine is not None:
                await self.engine.dispose()
                self.engine = None
                self._session_factory = None
                logger.info("Database connection closed")


db = Database()