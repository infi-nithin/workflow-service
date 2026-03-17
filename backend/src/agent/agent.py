import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from agent.utils import PromptService
import httpx
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.models import (
    SupervisorAgentState,
    SupervisorDecision,
    SupervisorAction,
)


class OrchestrationAgent:
    def __init__(
        self, graph_definition: Dict[str, Any], llm: ChatBedrock, tool_registry_url: str
    ):
        self.graph_definition = graph_definition
        self.llm = llm
        self.tool_registry_url = tool_registry_url
        self.mcp_server_url = self.tool_registry_url + "/mcp"
        self.intent = graph_definition.get("intent", "unknown")
        self.graph = self._build_graph()
        self.entry_point = self._find_entry_point()
        self._mcp_client = None
        self.prompt_service = PromptService()

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
        prompt = await self.prompt_service.get(
            "supervisor_system",
            intent=intent,
            version=version,
            nodes_text=nodes_text,
        )
        return prompt

    def _find_entry_point(self) -> str:
        edges = self.graph_definition.get("edges", [])
        nodes = self.graph_definition.get("nodes", [])
        incoming = set()
        for edge in edges:
            incoming.add(edge.get("to"))
        for node in nodes:
            node_id = node.get("id")
            if node_id not in incoming:
                return node_id
        return nodes[0].get("id") if nodes else "START"

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(SupervisorAgentState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("llm", self.llm_node)
        graph.add_node("tool", self.tool_node)
        graph.set_entry_point("supervisor")
        graph.add_edge("llm", "supervisor")
        graph.add_edge("tool", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._should_route,
            {
                "llm": "llm",
                "tool": "tool",
                "end": END,
            },
        )
        return graph.compile()

    def _should_route(self, state: SupervisorAgentState) -> str:
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

    async def supervisor_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        current_node = state.get("current_node", self.entry_point)
        execution_history = state.get("execution_history", [])
        graph_definition = state.get("graph_definition", self.graph_definition)
        node_outputs = state.get("node_outputs", {})
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
                SystemMessage(content=await self._create_supervisor_prompt(graph_definition)),
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
                "execution_history": execution_history
                + [
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
        prompt = await self.prompt_service.get(
            "supervisor_decision",
            node_info=node_info,
            messages_text=messages_text,
            outputs_text=outputs_text,
            tools_text=tools_text,
            history_text=history_text,
        )
        return prompt

    async def llm_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        decision = state.get("decision")
        execution_history = list(state.get("execution_history", []))
        node_outputs = dict(state.get("node_outputs", {}))

        if not decision or decision.action != SupervisorAction.LLM:
            return {"messages": messages, "node_outputs": node_outputs}

        llm_prompt = decision.llm_prompt or decision.response or "Process task"

        context = "\n\n".join(
            f"=== {k} ===\n{json.dumps(v, indent=2)}" for k, v in node_outputs.items()
        )

        full_prompt = f"Context:\n{context}\n\nTask:{llm_prompt}"

        system_msg = SystemMessage(content=await self.prompt_service.get("llm_context"))
        user_msg = HumanMessage(content=full_prompt)

        try:
            response = await self.llm.ainvoke([system_msg, user_msg])

            messages.append(AIMessage(content=response.content))

            current_node = state.get("current_node", "llm")

            node_outputs[current_node] = {
                "type": "llm",
                "response": response.content,
            }

            execution_history.append(
                {
                    "node": "llm",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            return {
                "messages": messages,
                "node_outputs": node_outputs,
                "execution_history": execution_history,
            }

        except Exception as e:
            execution_history.append({"node": "llm", "error": str(e)})
            return {
                "messages": messages,
                "execution_history": execution_history,
            }

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

    async def tool_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        decision = state.get("decision")
        tool_results = dict(state.get("tool_results", {}))
        execution_history = list(state.get("execution_history", []))
        node_outputs = dict(state.get("node_outputs", {}))
        raw_input = dict(state.get("raw_input", {}))

        if not decision or decision.action != SupervisorAction.TOOL:
            return {
                "messages": messages,
                "tool_results": tool_results,
                "node_outputs": node_outputs,
            }

        tool_name = decision.tool_name
        args = decision.tool_arguments or {}

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

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        initial_state: SupervisorAgentState = {
            "messages": [],
            "graph_definition": self.graph_definition,
            "current_node": self.entry_point,
            "execution_history": [],
            "tool_results": {},
            "workflow_id": input_data.get("workflow_id", str(uuid.uuid4())),
            "intent": self.intent,
            "should_continue": True,
            "node_outputs": {},
            "raw_input": input_data.get("raw_input", input_data),
        }
        user_message = input_data.get("raw_input", {}).get("message", str(input_data))
        initial_state["messages"] = [HumanMessage(content=user_message)]
        try:
            result = await self.graph.ainvoke(initial_state)
            final_output = None
            decision = result.get("decision")
            if decision and decision.response:
                final_output = decision.response
            elif result.get("node_outputs"):
                last_output = list(result["node_outputs"].values())[-1]
                final_output = last_output.get("result") or last_output.get("response")
            return {
                "intent": self.intent,
                "output": final_output,
                "node_outputs": result.get("node_outputs", {}),
                "execution_history": result.get("execution_history", []),
                "status": "completed",
            }
        except Exception as e:
            return {
                "intent": self.intent,
                "output": {"error": str(e)},
                "status": "failed",
            }
