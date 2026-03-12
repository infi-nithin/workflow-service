"""MCP Tool Executor with circuit breaker integration."""

import logging
from typing import Dict, Any, Optional, List
from agent.circuit_breaker import CircuitBreakerRegistry
from agent.models import ToolExecutionResponse
import os
from dotenv import load_dotenv


# Import MCP ClientSession and Streamable HTTP transport for HTTP-based MCP servers
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()
logger = logging.getLogger(__name__)


class MCPToolExecutor:
    """MCP Tool Executor with circuit breaker protection.

    Executes tools via the Tool Registry service with circuit breaker
    pattern to prevent cascading failures.
    Uses MCP ClientSession for tool calls instead of direct HTTP POST.
    """

    def __init__(
        self,
        tool_registry_url: Optional[str] = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: int = 30,
    ):
        """Initialize the MCP Tool Executor.

        Args:
            tool_registry_url: Base URL for the tool registry
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_cooldown: Seconds to wait before attempting recovery
        """
        self.tool_registry_url = tool_registry_url or os.getenv("TOOL_REGISTRY_URL")
        self.circuit_breakers = CircuitBreakerRegistry(
            threshold=circuit_breaker_threshold,
            cooldown=circuit_breaker_cooldown,
        )
        self._session: Optional[ClientSession] = None

    async def close(self):
        """Close the MCP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the registry.

        Returns:
            List of available tools
        """
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
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolExecutionResponse:
        """Execute a tool via the Tool Registry.

        Uses circuit breaker to prevent calling failing tools.
        Uses MCP ClientSession to call tools.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            ToolExecutionResponse with result or error
        """
        # Check circuit breaker
        if not self.circuit_breakers.can_execute(tool_name):
            breaker = self.circuit_breakers.get_breaker(tool_name)
            logger.warning(
                f"Circuit breaker is {breaker.state.value} for tool: {tool_name}"
            )
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
                    logger.warning(f"Execting tool: {tool_name}")
                    result = await session.call_tool(tool_name, arguments)
                    print(result)
            # Record success
            self.circuit_breakers.record_success(tool_name)

            # Extract result from MCP response
            # MCP call_tool returns a list of content items
            result_content = None
            if result.content:
                # Get text content from the first content item
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
            # Record failure
            self.circuit_breakers.record_failure(tool_name)

            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return ToolExecutionResponse(
                success=False,
                error=str(e),
            )

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        max_retries: int = 2,
    ) -> ToolExecutionResponse:
        """Execute a tool with retry logic.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            max_retries: Maximum number of retries

        Returns:
            ToolExecutionResponse with result or error
        """
        last_response: Optional[ToolExecutionResponse] = None

        for attempt in range(max_retries + 1):
            response = await self.execute_tool(tool_name, arguments)

            if response.success:
                return response

            last_response = response

            if attempt < max_retries:
                logger.info(
                    f"Retrying tool {tool_name} (attempt {attempt + 1}/{max_retries})"
                )

        return last_response or ToolExecutionResponse(
            success=False,
            error="Max retries exceeded",
        )

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get the status of all circuit breakers.

        Returns:
            Dictionary with circuit breaker status
        """
        return self.circuit_breakers.get_status()

    def reset_circuit_breaker(self, tool_name: str) -> None:
        """Reset the circuit breaker for a specific tool.

        Args:
            tool_name: Name of the tool
        """
        breaker = self.circuit_breakers.get_breaker(tool_name)
        breaker.reset()
        logger.info(f"Circuit breaker reset for tool: {tool_name}")

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self.circuit_breakers.reset_all()
        logger.info("All circuit breakers reset")


# Global executor instance
_executor: Optional[MCPToolExecutor] = None


def get_tool_executor() -> MCPToolExecutor:
    """Get or create the global tool executor instance.

    Returns:
        MCPToolExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = MCPToolExecutor(
            circuit_breaker_threshold=os.getenv("CIRCUIT_BREAKER_THRESHOLD"),
            circuit_breaker_cooldown=os.getenv("CIRCUIT_BREAKER_COOLDOWN"),
        )
    return _executor
