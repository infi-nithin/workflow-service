import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage

from agent.utils.prompt_service import PromptService
from agent.models import SupervisorAction


class HITLNode:
    def __init__(self, prompt_service: PromptService):
        self.prompt_service = prompt_service

    async def _create_hitl_prompt(
        self,
        current_node: str,
        current_node_def: Optional[Dict[str, Any]],
        node_outputs: Dict[str, Any],
        graph_definition: Dict[str, Any],
        decision_message: Optional[str] = None,
    ) -> str:
        # Build context from previous node outputs
        context_parts = []
        for node_id, output in node_outputs.items():
            context_parts.append(
                f"=== {node_id} ===\n{json.dumps(output, indent=2)[:500]}"
            )
        context_text = (
            "\n\n".join(context_parts) if context_parts else "No previous context"
        )
        # Get node information
        node_info = ""
        if current_node_def:
            node_type = current_node_def.get("type", "unknown")
            tool_name = current_node_def.get("tool_name")
            agent_name = current_node_def.get("agent_name")

            node_info = f"""
                        Current Node: {current_node}
                        Node Type: {node_type}
                        """
            if tool_name:
                node_info += f"Will execute tool: {tool_name}\n"
            if agent_name:
                node_info += f"Will use agent: {agent_name}\n"

        # Get possible next steps from graph
        edges = graph_definition.get("edges", [])
        possible_next = [
            edge.get("to") for edge in edges if edge.get("from") == current_node
        ]
        next_steps_text = (
            ", ".join(possible_next) if possible_next else "End of workflow"
        )

        return await self.prompt_service.get(
            "hitl_confirmation",
            context=context_text,
            node_info=node_info,
            next_steps=next_steps_text,
            decision_message=decision_message or "Please confirm to proceed",
        )

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        decision = state.get("decision")
        current_node = state.get("current_node", "hitl")
        graph_definition = state.get("graph_definition", {})
        node_outputs = dict(state.get("node_outputs", {}))
        execution_history = list(state.get("execution_history", []))
        hitl_responses = dict(state.get("hitl_responses", {}))

        # Only process if decision is to call HITL
        if not decision or decision.action != SupervisorAction.HITL:
            return {
                "messages": messages,
                "node_outputs": node_outputs,
            }

        # Check if human has already responded (resume from pending)
        pending_hitl = state.get("pending_hitl")

        # If we have a pending HITL with a response from human, process it
        if pending_hitl and pending_hitl.get("human_response"):
            # Human has responded - store the response
            hitl_node_id = pending_hitl.get("node_id", current_node)
            hitl_responses[hitl_node_id] = pending_hitl.get("human_response")

            # Add human response to messages
            messages.append(HumanMessage(content=pending_hitl.get("human_response")))

            # Update node outputs
            node_outputs[current_node] = {
                "type": "hitl",
                "confirmation_received": True,
                "human_response": pending_hitl.get("human_response"),
            }

            execution_history.append(
                {
                    "node": "hitl",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "confirmed",
                    "node_id": hitl_node_id,
                }
            )

            # Clear pending HITL and continue
            return {
                "messages": messages,
                "node_outputs": node_outputs,
                "hitl_responses": hitl_responses,
                "pending_hitl": None,  # Clear the pending request
                "execution_history": execution_history,
                "should_continue": True,
            }

        # No human response yet - create pending HITL request
        # Get current node definition
        nodes = graph_definition.get("nodes", [])
        current_node_def = None
        for node in nodes:
            if node.get("id") == current_node:
                current_node_def = node
                break

        # Get the confirmation message from decision or build one
        hitl_message = decision.hitl_message
        if not hitl_message:
            # Build a contextual message using graph-aware prompting
            try:
                hitl_message = await self._create_hitl_prompt(
                    current_node=current_node,
                    current_node_def=current_node_def,
                    node_outputs=node_outputs,
                    graph_definition=graph_definition,
                    decision_message=decision.response,
                )
            except Exception:
                # Fallback message if prompt service fails
                hitl_message = (
                    f"Node '{current_node}' requires your confirmation to proceed.\n"
                    f"Decision: {decision.reasoning}\n"
                    f"Please confirm or provide input to continue."
                )

        # Get options for the human (if provided)
        hitl_options = decision.hitl_options or []

        # Create pending HITL request
        pending_hitl_request = {
            "node_id": current_node,
            "message": hitl_message,
            "options": hitl_options,
            "reasoning": decision.reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "human_response": None,  # To be filled by human
        }

        # Update node outputs to show waiting state
        node_outputs[current_node] = {
            "type": "hitl",
            "status": "pending",
            "message": hitl_message,
            "options": hitl_options,
        }

        # Add system message about waiting for confirmation
        messages.append(
            AIMessage(
                content=f"Waiting for human confirmation on node '{current_node}': {hitl_message}"
            )
        )

        execution_history.append(
            {
                "node": "hitl",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "node_id": current_node,
            }
        )

        return {
            "messages": messages,
            "node_outputs": node_outputs,
            "pending_hitl": pending_hitl_request,
            "hitl_responses": hitl_responses,
            "execution_history": execution_history,
            "should_continue": False,
        }

    def should_route(self, state: Dict[str, Any]) -> str:
        from agent.models import SupervisorAction

        decision = state.get("decision")
        pending_hitl = state.get("pending_hitl")

        if not decision or decision.action != SupervisorAction.HITL:
            return "end"

        # If there's a human response, continue to supervisor for next action
        if pending_hitl and pending_hitl.get("human_response"):
            return "supervisor"

        # Still waiting for human - could return "wait" or "end" depending on implementation
        return "end"
