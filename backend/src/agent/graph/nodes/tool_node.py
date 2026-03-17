import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


class ToolNode:
    def __init__(self, mcp_server_url: Optional[str] = None):
        self.mcp_server_url = mcp_server_url
        self._mcp_client = None

    async def _get_mcp_client(self) -> Optional[MultiServerMCPClient]:
        if not self.mcp_server_url:
            return None

        if self._mcp_client is None:
            try:
                config = {
                    "tool-registry": {
                        "url": self.mcp_server_url,
                        "transport": "streamable-http",
                    }
                }
                self._mcp_client = MultiServerMCPClient(config)
            except Exception:
                return None

        return self._mcp_client

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        mcp_client = await self._get_mcp_client()
        try:
            mcp_tools = await mcp_client.get_tools()
            tool = next(t for t in mcp_tools if t.name == tool_name)
            result = await tool.ainvoke(arguments)
            if hasattr(result, "content"):
                return {"result": result.content}
            return {"result": str(result)}
        except Exception:
            raise

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        decision = state.get("decision")
        tool_results = dict(state.get("tool_results", {}))
        execution_history = list(state.get("execution_history", []))
        node_outputs = dict(state.get("node_outputs", {}))
        raw_input = dict(state.get("raw_input", {}))

        # Only process if decision is to call a tool
        from agent.models.models import SupervisorAction
        if not decision or decision.action != SupervisorAction.TOOL:
            return {
                "messages": messages,
                "tool_results": tool_results,
                "node_outputs": node_outputs,
            }

        tool_name = decision.tool_name
        args = decision.tool_arguments or {}

        # Merge arguments from various sources
        merged_args = {**raw_input, **node_outputs, **args}

        try:
            result = await self._call_mcp_tool(tool_name, merged_args)
        except Exception as e:
            result = {"error": str(e)}

        tool_results[tool_name] = result

        current_node = state.get("current_node", tool_name)

        node_outputs[current_node] = {
            "type": "tool",
            "tool_name": tool_name,
            "result": result,
        }

        messages.append(
            ToolMessage(content=json.dumps(result), tool_call_id=tool_name)
        )

        execution_history.append(
            {
                "node": "tool",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return {
            "messages": messages,
            "tool_results": tool_results,
            "node_outputs": node_outputs,
            "execution_history": execution_history,
        }
