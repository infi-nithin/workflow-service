from typing import Optional
from db.database import db
import time
from db.models import ToolRegistry
from sqlalchemy import select

class ToolRegistryService:
    def __init__(self, cache_ttl: int = 300):
        self._cache: dict[str, tuple[Optional[str], float]] = {}
        self._cache_ttl = cache_ttl

    async def get_url(self, sys_id: str) -> Optional[str]:
        if not sys_id:
            return None

        now = time.time()

        if sys_id in self._cache:
            url, cached_at = self._cache[sys_id]
            if now - cached_at < self._cache_ttl:
                return url

        url = await self._fetch(sys_id)
        self._cache[sys_id] = (url, now)
        return url

    async def _fetch(self, sys_id: str) -> Optional[str]:
        async with db.session() as session:
            result = await session.execute(
                select(ToolRegistry).where(
                    ToolRegistry.sys_id == sys_id,
                    ToolRegistry.is_active,
                )
            )
            tool_registry = result.scalar_one_or_none()
            return tool_registry.tool_registry_url if tool_registry else None

    def invalidate(self, sys_id: Optional[str] = None) -> None:
        if sys_id:
            self._cache.pop(sys_id, None)
        else:
            self._cache.clear()
