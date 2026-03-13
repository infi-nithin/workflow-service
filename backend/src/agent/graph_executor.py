import json
import time
import uuid
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from agent.models import AgentState, NodeExecutionLog
from agent.tool_executor import get_tool_executor
from agent.llm_utils import get_llm_client


class GraphExecutor:
    def __init__(
        self,
        tool_executor=None,
        llm_client=None,
    ):
        self.tool_executor = tool_executor or get_tool_executor()
        self.llm_client = llm_client or get_llm_client()

    def build_graph(
        self,
        graph_definition: Dict[str, Any],
    ) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        nodes = graph_definition.get("nodes", [])
        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            node_func = self._create_node_function(node_type, node)
            graph.add_node(node_id, node_func)
        edges = graph_definition.get("edges", [])
        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            if from_node and to_node:
                graph.add_edge(from_node, to_node)
        first_node = nodes[0]["id"]
        last_node = nodes[-1]["id"]
        graph.add_edge(START, first_node)
        graph.add_edge(last_node, END)
        return graph.compile()

    def _create_node_function(
        self,
        node_type: str,
        node: Dict[str, Any],
    ) -> Callable:
        if node_type == "mcp_tool":
            return self._create_tool_node(node)
        elif node_type == "llm":
            return self._create_llm_node(node)
        elif node_type == "sub_agent":
            return self._create_sub_agent_node(node)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _create_tool_node(self, node: Dict[str, Any]) -> Callable:
        tool_name = node.get("tool_name") or node.get("config", {}).get("tool_name")
        input_mapping = node.get("input_mapping") or node.get("config", {}).get(
            "input_mapping", {}
        )
        output_key = node.get("output_key") or node.get("config", {}).get(
            "output_key", "tool_result"
        )
        node_id = node.get("id", "tool_node")

        async def tool_node(state: AgentState) -> AgentState:
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="mcp_tool",
                duration_ms=0,
            )
            try:
                arguments = {}
                for param, source in input_mapping.items():
                    if isinstance(source, str) and source in state:
                        arguments[param] = state[source]
                    else:
                        arguments[param] = source
                result = await self.tool_executor.execute_tool(tool_name, arguments)
                state["node_outputs"][output_key] = (
                    result.dict() if hasattr(result, "dict") else result
                )
                node_log.tool_call = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "success": result.success if hasattr(result, "success") else True,
                }
            except Exception as e:
                node_log.error = str(e)
                state["node_outputs"][output_key] = {"error": str(e)}
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, "dict") else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            return state

        return tool_node

    def _create_llm_node(self, node: Dict[str, Any]) -> Callable:
        prompt_template = node.get("prompt_template") or node.get("config", {}).get(
            "prompt_template"
        )
        system_prompt = node.get("system_prompt") or node.get("config", {}).get(
            "system_prompt", ""
        )
        input_key = node.get("input_key") or node.get("config", {}).get(
            "input_key", "raw_input"
        )
        output_key = node.get("output_key") or node.get("config", {}).get(
            "output_key", "llm_result"
        )
        temperature = node.get("temperature") or node.get("config", {}).get(
            "temperature", 0.7
        )
        node_id = node.get("id", "llm_node")

        async def llm_node(state: AgentState) -> AgentState:
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="llm",
                duration_ms=0,
            )
            try:
                input_data = state.get(input_key, {})
                if isinstance(input_data, dict):
                    user_message = input_data.get("message", "") or json.dumps(
                        input_data
                    )
                else:
                    user_message = str(input_data)
                if prompt_template:
                    try:
                        formatted_prompt = prompt_template.format(**state)
                    except (KeyError, AttributeError):
                        formatted_prompt = prompt_template
                    user_message = formatted_prompt
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=user_message))
                result = self.llm_client.invoke(messages)
                content = result.content if hasattr(result, "content") else str(result)
                state["node_outputs"][output_key] = content
                state["final_output"] = content
                model_used = None
                if hasattr(result, "response_metadata") and result.response_metadata:
                    model_used = result.response_metadata.get("model")
                node_log.llm_call = {
                    "model": model_used,
                    "temperature": temperature,
                    "usage": None,
                }
                execution_log = state.get("execution_log", {})
                current_tokens = execution_log.get("total_tokens", 0)
                execution_log["total_tokens"] = current_tokens
                model_versions = execution_log.get("model_versions_used", [])
                if model_used and model_used not in model_versions:
                    model_versions.append(model_used)
                execution_log["model_versions_used"] = model_versions
                state["execution_log"] = execution_log
            except Exception as e:
                node_log.error = str(e)
                state["node_outputs"][output_key] = {"error": str(e)}
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, "dict") else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            return state

        return llm_node

    def _create_sub_agent_node(self, node: Dict[str, Any]) -> Callable:
        agent_name = node.get("agent_name") or node.get("config", {}).get("agent_name")
        node_id = node.get("id", agent_name or "sub_agent")

        async def sub_agent_node(state: AgentState) -> AgentState:
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="sub_agent",
                duration_ms=0,
            )
            try:
                node_log.error = "Sub-agent execution not yet implemented"
            except Exception as e:
                node_log.error = str(e)
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, "dict") else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            return state

        return sub_agent_node

    async def execute(
        self,
        initial_state: AgentState,
    ) -> AgentState:
        graph_definition = initial_state.get("graph_definition")
        try:
            graph = self.build_graph(graph_definition)
            result = await graph.ainvoke(initial_state)
            return result
        except Exception as e:
            initial_state["execution_log"]["error"] = str(e)
            initial_state["final_output"] = {"error": str(e)}
            return initial_state


_executor: Optional[GraphExecutor] = None


def get_graph_executor() -> GraphExecutor:
    global _executor
    if _executor is None:
        _executor = GraphExecutor()
    return _executor
