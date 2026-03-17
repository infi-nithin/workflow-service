from typing import Optional
from db.database import db
import time
from db.models import Prompt
from sqlalchemy import select


class PromptService:
    def __init__(self, cache_ttl: int = 300):
        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = cache_ttl

    async def get(self, key: str, **format_kwargs) -> Optional[str]:
        if not key:
            return None

        now = time.time()

        if key in self._cache:
            content, cached_at = self._cache[key]
            if now - cached_at < self._cache_ttl:
                return content.format(**format_kwargs) if format_kwargs else content

        content = await self._fetch(key)
        if content is None:
            return None

        self._cache[key] = (content, now)
        return content.format(**format_kwargs) if format_kwargs else content

    async def _fetch(self, key: str) -> Optional[str]:
        async with db.session() as session:
            result = await session.execute(
                select(Prompt).where(Prompt.prompt_name == key)
            )
            prompt = result.scalar_one_or_none()
            return prompt.prompt_content if prompt else None

    def invalidate(self, key: Optional[str] = None) -> None:
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()
