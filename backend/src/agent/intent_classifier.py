import json
import logging
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm_utils import get_llm_client


logger = logging.getLogger(__name__)


# Default system prompt for intent classification
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
    """Intent classifier using LLM for natural language understanding.

    Classifies user input into predefined intents using the LLM.
    """

    def __init__(
        self,
        available_intents: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the intent classifier.

        Args:
            available_intents: List of available intent names
            system_prompt: Custom system prompt for classification
        """
        self.available_intents = available_intents or []
        self.system_prompt = system_prompt or DEFAULT_CLASSIFICATION_PROMPT
        self.llm_client = get_llm_client()

    def set_available_intents(self, intents: List[str]) -> None:
        """Set the available intents.

        Args:
            intents: List of intent names
        """
        self.available_intents = intents


    async def classify(
        self,
        user_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Classify the user input into an intent.

        Args:
            user_input: The user's input data

        Returns:
            Classification result with intent, confidence, and reasoning
        """
        # Convert input to string for classification
        if isinstance(user_input, dict):
            input_text = user_input.get("message", "") or user_input.get("text", "")
            if not input_text:
                input_text = json.dumps(user_input)
        else:
            input_text = str(user_input)

        # Format available intents
        intents_str = "\n".join(f"- {intent}" for intent in self.available_intents)
        # Prepare messages for ChatBedrock invoke
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

            # Parse the response - ChatBedrock returns AIMessage with content attribute
            content = response.content
            try:
                # Try to extract JSON from the response
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the text
                import re

                json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Could not parse LLM response as JSON: {content}")
                    return {
                        "intent": "unknown",
                        "confidence": 0.0,
                        "reasoning": "Failed to parse classification result",
                    }

            return {
                "intent": result.get("intent", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "model_used": response.response_metadata.get("model")
                if hasattr(response, "response_metadata")
                else None,
                "usage": None,  # ChatBedrock invoke doesn't return usage directly
            }

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}",
            }

    async def classify_simple(
        self,
        user_input: str,
    ) -> str:
        """Simple classification returning just the intent name.

        Args:
            user_input: The user's input text

        Returns:
            The classified intent name
        """
        result = await self.classify({"text": user_input})
        return result.get("intent", "unknown")


# Global classifier instance
_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the global intent classifier.

    Returns:
        IntentClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
