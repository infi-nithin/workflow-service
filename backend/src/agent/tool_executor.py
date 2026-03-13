import uuid
from typing import Dict, Any, Optional, List
from agent.circuit_breaker import CircuitBreakerRegistry
from agent.models import ToolExecutionResponse
import os
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()


class MCPToolExecutor:
    def __init__(
        self,
        tool_registry_url: Optional[str] = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: int = 30,
    ):
        self.tool_registry_url = tool_registry_url or os.getenv("TOOL_REGISTRY_URL")
        self.circuit_breakers = CircuitBreakerRegistry(
            threshold=circuit_breaker_threshold,
            cooldown=circuit_breaker_cooldown,
        )
        self._session: Optional[ClientSession] = None

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        try:
            tools = []
            async with streamable_http_client(self.tool_registry_url + "/mcp") as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in tools.tools
            ]
        except Exception:
            return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ToolExecutionResponse:
        trace_id = trace_id or str(uuid.uuid4())
        workflow_id = workflow_id or "unknown"
        if not self.circuit_breakers.can_execute(tool_name):
            return ToolExecutionResponse(
                success=False,
                error=f"Circuit breaker is open for tool: {tool_name}",
            )
        try:
            result = None
            async with streamable_http_client(self.tool_registry_url + "/mcp") as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    print(result)
            self.circuit_breakers.record_success(tool_name)
            result_content = None
            if result.content:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    result_content = content_item.text
                else:
                    result_content = str(content_item)
            return ToolExecutionResponse(
                success=True,
                result=result_content,
                error=None,
            )
        except Exception as e:
            self.circuit_breakers.record_failure(tool_name)
            return ToolExecutionResponse(
                success=False,
                error=str(e),
            )

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        max_retries: int = 2,
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ToolExecutionResponse:
        last_response: Optional[ToolExecutionResponse] = None
        for attempt in range(max_retries + 1):
            response = await self.execute_tool(
                tool_name, arguments, trace_id, workflow_id
            )
            if response.success:
                return response
            last_response = response
        return last_response or ToolExecutionResponse(
            success=False,
            error="Max retries exceeded",
        )

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        return self.circuit_breakers.get_status()

    def reset_circuit_breaker(self, tool_name: str) -> None:
        breaker = self.circuit_breakers.get_breaker(tool_name)
        breaker.reset()

    def reset_all_circuit_breakers(self) -> None:
        self.circuit_breakers.reset_all()


_executor: Optional[MCPToolExecutor] = None


def get_tool_executor() -> MCPToolExecutor:
    global _executor
    if _executor is None:
        _executor = MCPToolExecutor(
            circuit_breaker_threshold=os.getenv("CIRCUIT_BREAKER_THRESHOLD"),
            circuit_breaker_cooldown=os.getenv("CIRCUIT_BREAKER_COOLDOWN"),
        )
    return _executor
