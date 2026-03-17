import uuid
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from agent.utils.prompt_service import PromptService
from agent.graph.nodes.supervisor import SupervisorNode
from agent.graph.nodes.llm_node import LLMNode
from agent.graph.nodes.tool_node import ToolNode
from agent.graph.builder import GraphBuilder
from agent.models.models import SupervisorAgentState


class OrchestrationAgent:
    def __init__(self, graph_definition: Dict[str, Any], llm, tool_registry_url: str):
        self.graph_definition = graph_definition
        self.llm = llm
        self.tool_registry_url = tool_registry_url
        self.intent = graph_definition.get("intent", "unknown")

        # Initialize services
        self.prompt_service = PromptService()

        # Initialize nodes (delegating logic to specialized classes)
        self._init_nodes()

        # Build the graph
        self.graph = GraphBuilder.build(
            supervisor_node=self.supervisor_node,
            llm_node=self.llm_node,
            tool_node=self.tool_node,
        )

        # Find entry point
        self.entry_point = GraphBuilder.find_entry_point(graph_definition)

    def _init_nodes(self) -> None:
        # Supervisor node - handles decision making
        self.supervisor_node = SupervisorNode(
            llm=self.llm,
            tool_registry_url=self.tool_registry_url,
            prompt_service=self.prompt_service,
        )

        # LLM node - handles LLM calls
        self.llm_node = LLMNode(
            llm=self.llm,
            prompt_service=self.prompt_service,
        )

        # Tool node - handles MCP tool calls
        mcp_server_url = (
            self.tool_registry_url + "/mcp" if self.tool_registry_url else None
        )
        self.tool_node = ToolNode(mcp_server_url=mcp_server_url)

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Build initial state
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

        # Add user message
        user_message = input_data.get("raw_input", {}).get("message", str(input_data))
        initial_state["messages"] = [HumanMessage(content=user_message)]

        try:
            # Execute the graph
            result = await self.graph.ainvoke(initial_state)

            # Extract final output
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
