import json
import uuid
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm_utils import get_llm_client

DEFAULT_CLASSIFICATION_PROMPT = """You are an intent classifier. Your task is to classify user input into one of the available intents.
Available intents:
{available_intents}
Instructions:
1. Analyze the user's input
2. Select the most appropriate intent from the available intents
3. Return a JSON object with your classification
Return format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
If none of the available intents match, return:
{{
    "intent": "unknown",
    "confidence": 0.0,
    "reasoning": "explanation"
}}
"""


class IntentClassifier:
    def __init__(
        self,
        available_intents: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.available_intents = available_intents or []
        self.system_prompt = system_prompt or DEFAULT_CLASSIFICATION_PROMPT
        self.llm_client = get_llm_client()

    def set_available_intents(self, intents: List[str]) -> None:
        self.available_intents = intents

    async def classify(
        self,
        user_input: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace_id = trace_id or str(uuid.uuid4())
        if isinstance(user_input, dict):
            input_text = user_input.get("message", "") or user_input.get("text", "")
            if not input_text:
                input_text = json.dumps(user_input)
        else:
            input_text = str(user_input)
        intents_str = "\n".join(f"- {intent}" for intent in self.available_intents)
        messages = [
            SystemMessage(
                content=self.system_prompt.format(available_intents=intents_str)
            ),
            HumanMessage(content=f"Classify this input:\n{input_text}"),
        ]
        try:
            response = self.llm_client.invoke(
                messages,
            )
            content = response.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                import re

                json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {
                        "intent": "unknown",
                        "confidence": 0.0,
                        "reasoning": "Failed to parse classification result",
                    }
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            model_used = (
                response.response_metadata.get("model")
                if hasattr(response, "response_metadata")
                else None
            )
            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": reasoning,
                "model_used": model_used,
                "usage": None,
            }
        except Exception as e:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}",
            }

    async def classify_simple(
        self,
        user_input: str,
    ) -> str:
        result = await self.classify({"text": user_input})
        return result.get("intent", "unknown")


_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
