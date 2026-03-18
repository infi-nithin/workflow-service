import uuid
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from agent.utils.prompt_service import PromptService
from agent.graph.nodes.supervisor import SupervisorNode
from agent.graph.nodes.llm_node import LLMNode
from agent.graph.nodes.tool_node import ToolNode
from agent.graph.nodes.HITL import HITLNode
from agent.graph.builder import GraphBuilder
from agent.models import SupervisorAgentState


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
            hitl_node=self.hitl_node,
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

        # HITL node - handles human confirmation
        self.hitl_node = HITLNode(
            prompt_service=self.prompt_service,
        )

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
            "hitl_responses": {},
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
            
            # Check if there's a pending HITL
            pending_hitl = result.get("pending_hitl")
            result_status = "pending_hitl" if pending_hitl else "completed"
            
            # Build response
            response = {
                "intent": self.intent,
                "output": final_output,
                "node_outputs": result.get("node_outputs", {}),
                "execution_history": result.get("execution_history", []),
                "status": result_status,
            }
            
            # If pending HITL, include the pending_hitl and current_state
            if pending_hitl:
                response["pending_hitl"] = pending_hitl
                response["current_state"] = {
                    "graph_definition": self.graph_definition,
                    "messages": [msg.model_dump() for msg in result.get("messages", [])],
                    "current_node": result.get("current_node"),
                    "execution_history": result.get("execution_history", []),
                    "tool_results": result.get("tool_results", {}),
                    "node_outputs": result.get("node_outputs", {}),
                    "hitl_responses": result.get("hitl_responses", {}),
                    "workflow_id": result.get("workflow_id"),
                    "intent": self.intent,
                    "should_continue": result.get("should_continue", True),
                }
            
            return response

        except Exception as e:
            return {
                "intent": self.intent,
                "output": {"error": str(e)},
                "status": "failed",
            }

    async def invoke_resume(self, restored_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resume execution from a saved state (after HITL confirmation).
        
        The restored_state should contain:
        - graph_definition: The workflow graph
        - messages: Previous messages
        - current_node: Current node in workflow
        - execution_history: Previous execution steps
        - tool_results: Results from tool calls
        - node_outputs: Outputs from previous nodes
        - hitl_responses: Previous HITL responses
        - pending_hitl: The pending HITL with human_response filled
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # Convert message dicts back to message objects
        messages = []
        for msg in restored_state.get("messages", []):
            msg_type = msg.get("type")
            if msg_type == "human":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg_type == "ai":
                messages.append(AIMessage(content=msg.get("content", "")))
            elif msg_type == "system":
                messages.append(SystemMessage(content=msg.get("content", "")))
        
        # Build state for resuming
        resume_state: SupervisorAgentState = {
            "messages": messages,
            "graph_definition": restored_state.get("graph_definition", self.graph_definition),
            "current_node": restored_state.get("current_node", self.entry_point),
            "execution_history": restored_state.get("execution_history", []),
            "tool_results": restored_state.get("tool_results", {}),
            "workflow_id": restored_state.get("workflow_id"),
            "intent": restored_state.get("intent", self.intent),
            "should_continue": restored_state.get("should_continue", True),
            "node_outputs": restored_state.get("node_outputs", {}),
            "raw_input": restored_state.get("raw_input", {}),
            "hitl_responses": restored_state.get("hitl_responses", {}),
            "pending_hitl": restored_state.get("pending_hitl"),
        }

        try:
            # Execute the graph from where we left off
            result = await self.graph.ainvoke(resume_state)

            # Extract final output
            final_output = None
            decision = result.get("decision")
            if decision and decision.response:
                final_output = decision.response
            elif result.get("node_outputs"):
                last_output = list(result["node_outputs"].values())[-1]
                final_output = last_output.get("result") or last_output.get("response")
            
            # Check if there's a pending HITL
            pending_hitl = result.get("pending_hitl")
            result_status = "pending_hitl" if pending_hitl else "completed"
            
            # Build response
            response = {
                "intent": self.intent,
                "output": final_output,
                "node_outputs": result.get("node_outputs", {}),
                "execution_history": result.get("execution_history", []),
                "status": result_status,
            }
            
            # If pending HITL, include the pending_hitl and current_state
            if pending_hitl:
                response["pending_hitl"] = pending_hitl
                response["current_state"] = {
                    "graph_definition": self.graph_definition,
                    "messages": [msg.model_dump() for msg in result.get("messages", [])],
                    "current_node": result.get("current_node"),
                    "execution_history": result.get("execution_history", []),
                    "tool_results": result.get("tool_results", {}),
                    "node_outputs": result.get("node_outputs", {}),
                    "hitl_responses": result.get("hitl_responses", {}),
                    "workflow_id": result.get("workflow_id"),
                    "intent": self.intent,
                    "should_continue": result.get("should_continue", True),
                }
            
            return response

        except Exception as e:
            return {
                "intent": self.intent,
                "output": {"error": str(e)},
                "status": "failed",
            }
