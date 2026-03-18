import json
from typing import Dict, Any
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.utils.prompt_service import PromptService
from agent.utils.circuit_breaker import get_circuit_breaker_registry


class LLMNode:
    def __init__(self, llm, prompt_service: PromptService):
        self.llm = llm
        self.prompt_service = prompt_service

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        decision = state.get("decision")
        execution_history = list(state.get("execution_history", []))
        node_outputs = dict(state.get("node_outputs", {}))

        # Only process if decision is to call LLM
        from agent.models import SupervisorAction
        if not decision or decision.action != SupervisorAction.LLM:
            return {"messages": messages, "node_outputs": node_outputs}

        # Get circuit breaker registry for LLM
        circuit_breaker = get_circuit_breaker_registry()
        llm_key = "llm"

        # Check if circuit breaker allows LLM execution
        if not circuit_breaker.can_execute(llm_key):
            execution_history.append(
                {
                    "node": "llm",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "Circuit breaker is OPEN for LLM. Please try again later.",
                    "circuit_breaker_open": True,
                }
            )
            return {
                "messages": messages,
                "execution_history": execution_history,
            }

        llm_prompt = decision.llm_prompt or decision.response or "Process task"

        # Build context from previous node outputs
        context = "\n\n".join(
            f"=== {k} ===\n{json.dumps(v, indent=2)}" for k, v in node_outputs.items()
        )

        full_prompt = f"Context:\n{context}\n\nTask:{llm_prompt}"

        system_msg = SystemMessage(content=await self.prompt_service.get("llm_context"))
        user_msg = HumanMessage(content=full_prompt)

        try:
            response = await self.llm.ainvoke([system_msg, user_msg])
            # Record success in circuit breaker
            circuit_breaker.record_success(llm_key)

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
            # Record failure in circuit breaker
            circuit_breaker.record_failure(llm_key)
            execution_history.append({"node": "llm", "error": str(e)})
            return {
                "messages": messages,
                "execution_history": execution_history,
            }
