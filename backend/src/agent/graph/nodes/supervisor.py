import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import httpx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from agent.utils.prompt_service import PromptService
from agent.models.models import SupervisorAgentState, SupervisorDecision, SupervisorAction


class SupervisorNode:
    def __init__(self, llm, tool_registry_url: str, prompt_service: PromptService):
        self.llm = llm
        self.tool_registry_url = tool_registry_url
        self.prompt_service = prompt_service

    def _find_entry_point(graph_definition: dict) -> str:
        edges = graph_definition.get("edges", [])
        nodes = graph_definition.get("nodes", [])

        incoming = set()
        for edge in edges:
            incoming.add(edge.get("to"))

        for node in nodes:
            node_id = node.get("id")
            if node_id not in incoming:
                return node_id

        return nodes[0].get("id") if nodes else "START"
    
    async def _create_supervisor_prompt(self, graph_definition: Dict[str, Any]) -> str:
        nodes = graph_definition.get("nodes", [])
        edges = graph_definition.get("edges", [])
        version = graph_definition.get("version", "unknown")
        intent = graph_definition.get("intent", "unknown")

        nodes_description = []
        for node in nodes:
            node_id = node.get("id", "")
            node_type = node.get("type", "unknown")
            next_nodes = [
                edge.get("to") for edge in edges if edge.get("from") == node_id
            ]
            node_desc = f"- Node: {node_id}, Type: {node_type}"
            if next_nodes:
                node_desc += f", Can proceed to: {', '.join(next_nodes)}"
            if node.get("tool_name"):
                node_desc += f", Tool: {node.get('tool_name')}"
            if node.get("prompt_template"):
                node_desc += ", Has prompt template"
            if node.get("agent_name"):
                node_desc += f", Agent: {node.get('agent_name')}"
            nodes_description.append(node_desc)

        nodes_text = (
            "\n".join(nodes_description) if nodes_description else "No nodes defined"
        )

        return await self.prompt_service.get(
            "supervisor_system",
            intent=intent,
            version=version,
            nodes_text=nodes_text,
        )

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.tool_registry_url}/api/v1/mcp/tools"
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("tools", [])
        except Exception:
            raise
        return []

    async def _create_supervisor_decision_prompt(
        self,
        current_node: str,
        current_node_def: Optional[Dict[str, Any]],
        messages: List[Any],
        execution_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        node_outputs: Dict[str, Any],
        graph_definition: Dict[str, Any],
    ) -> str:
        recent_messages = []
        for msg in messages[-5:]:
            if isinstance(msg, HumanMessage):
                recent_messages.append(f"Human: {msg.content[:200]}...")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    recent_messages.append(f"AI (tool call): {msg.tool_calls}")
                else:
                    recent_messages.append(f"AI: {msg.content[:200]}...")
            elif isinstance(msg, ToolMessage):
                recent_messages.append(
                    f"Tool ({msg.tool_call_id}): {msg.content[:200]}..."
                )

        messages_text = (
            "\n".join(recent_messages) if recent_messages else "No messages yet"
        )

        tools_text = ""
        if available_tools:
            tools_info = []
            for tool in available_tools[:10]:
                tools_info.append(
                    f"- {tool.get('name')}: {tool.get('description', 'No description')}"
                )
            tools_text = "\nAvailable tools:\n" + "\n".join(tools_info)

        history_text = ""
        if execution_history:
            history_entries = []
            for entry in execution_history[-5:]:
                node = entry.get("node", "unknown")
                timestamp = entry.get("timestamp", "")
                history_entries.append(f"- {node} at {timestamp}")
            history_text = "\nExecution history:\n" + "\n".join(history_entries)

        outputs_text = ""
        if node_outputs:
            outputs_info = []
            for node_id, output in node_outputs.items():
                outputs_info.append(
                    f"- {node_id}: {json.dumps(output, indent=2)[:300]}..."
                )
            outputs_text = "\nPrevious node outputs:\n" + "\n".join(outputs_info)

        node_info = ""
        if current_node_def:
            node_type = current_node_def.get("type", "unknown")
            tool_name = current_node_def.get("tool_name")
            prompt_template = current_node_def.get("prompt_template")
            agent_name = current_node_def.get("agent_name")
            node_info = f"""
Current Node: {current_node}
Node Type: {node_type}
"""
            if tool_name:
                node_info += f"Requires Tool: {tool_name}\n"
            if prompt_template:
                node_info += f"Has Prompt Template: {prompt_template[:100]}...\n"
            if agent_name:
                node_info += f"Requires Agent: {agent_name}\n"

        edges = graph_definition.get("edges", [])
        possible_next = [
            edge.get("to") for edge in edges if edge.get("from") == current_node
        ]
        if possible_next:
            node_info += f"Possible next nodes: {', '.join(possible_next)}\n"

        return await self.prompt_service.get(
            "supervisor_decision",
            node_info=node_info,
            messages_text=messages_text,
            outputs_text=outputs_text,
            tools_text=tools_text,
            history_text=history_text,
        )

    async def execute(self, state: SupervisorAgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        current_node = state.get("current_node")
        execution_history = state.get("execution_history", [])
        graph_definition = state.get("graph_definition")
        node_outputs = state.get("node_outputs", {})

        # Find entry point if not set
        if not current_node:
            current_node = self._find_entry_point(graph_definition)

        nodes = graph_definition.get("nodes", [])
        current_node_def = None
        for node in nodes:
            if node.get("id") == current_node:
                current_node_def = node
                break

        available_tools = await self._get_available_tools()
        supervisor_decision_prompt = await self._create_supervisor_decision_prompt(
            current_node=current_node,
            current_node_def=current_node_def,
            messages=messages,
            execution_history=execution_history,
            available_tools=available_tools,
            node_outputs=node_outputs,
            graph_definition=graph_definition,
        )

        try:
            llm_with_structure = self.llm.with_structured_output(SupervisorDecision)
            decision_messages = [
                SystemMessage(
                    content=await self._create_supervisor_prompt(graph_definition)
                ),
                HumanMessage(content=supervisor_decision_prompt),
            ]
            decision = llm_with_structure.invoke(decision_messages)

            updates: Dict[str, Any] = {
                "decision": decision,
                "current_node": decision.next_node or current_node,
                "should_continue": decision.action not in (SupervisorAction.END,),
            }

            execution_history = list(execution_history) if execution_history else []
            execution_history.append(
                {
                    "node": "supervisor",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": decision.model_dump(),
                }
            )
            updates["execution_history"] = execution_history
            return updates

        except Exception as e:
            return {
                "should_continue": False,
                "decision": SupervisorDecision(
                    action=SupervisorAction.END,
                    reasoning=f"Error in supervisor: {str(e)}",
                    response="An error occurred during execution.",
                ),
                "execution_history": execution_history + [
                    {
                        "node": "supervisor",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                    }
                ]
                if execution_history
                else [
                    {
                        "node": "supervisor",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                    }
                ],
            }

    def should_route(self, state: SupervisorAgentState) -> str:
        decision = state.get("decision")
        if not decision:
            return "end"

        action = decision.action
        if action == SupervisorAction.LLM:
            return "llm"
        elif action == SupervisorAction.TOOL:
            return "tool"
        else:
            return "end"
