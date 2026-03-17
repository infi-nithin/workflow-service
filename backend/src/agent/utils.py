from typing import Optional
from db.database import db
from langchain_aws import ChatBedrock
from config.config import config
import time
from db.models import Prompt, ToolRegistry
from sqlalchemy import select


def get_llm_client() -> ChatBedrock:
    return ChatBedrock(
        model_id=config.aws.bedrock_model_id,
        aws_secret_access_key=config.aws.secret_access_key,
        aws_access_key_id=config.aws.access_key_id,
        aws_session_token=config.aws.session_token,
        region_name=config.aws.region,
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )


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
