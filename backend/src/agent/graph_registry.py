"""Graph Registry client for retrieving graph definitions."""

import httpx
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


class GraphByIntent(BaseModel):
    """Represents a graph definition keyed by intent.

    The intent is the key, and the graph definition is the value.
    This model allows accessing graphs directly by their intent.
    """

    graphs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Graphs keyed by intent. Key is the intent string, value is the graph definition.",
    )


class GraphRegistryClient:
    """Client for interacting with the Graph Registry service.

    Provides methods to fetch graph definitions by intent
    and list available graphs.
    """

    def __init__(self, graph_registry_url: Optional[str] = None):
        """Initialize the Graph Registry client.

        Args:
            graph_registry_url: Base URL for the graph registry
        """
        self.graph_registry_url = os.getenv("GRAPH_REGISTRY_URL")
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()

    async def get_graph_by_intent(self, intent: str) -> Optional[Dict[str, Any]]:
        """Get a graph definition by intent.

        Args:
            intent: The intent to look up

        Returns:
            Graph definition dictionary or None if not found
        """
        url = f"{self.graph_registry_url}/api/v1/graphs/{intent}"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Graph not found for intent: {intent}")
                return None
            logger.error(f"Failed to get graph for intent {intent}: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting graph for intent {intent}: {e}")
            raise

    async def list_graphs(self) -> Dict[str, Dict[str, Any]]:
        """List all available graphs as a key-value pair.

        Returns:
            Dictionary where key is the intent and value is the graph definition.
            Example: {"analyze_data": {...}, "other_intent": {...}}
        """
        url = f"{self.graph_registry_url}/api/v1/graphs"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            data = response.json()

            # Transform from list format to key-value format
            # Original format: {"graphs": [{"intent": "analyze_data", "graph": {...}}, ...]}
            # New format: {"analyze_data": {...}, "other_intent": {...}}
            graphs = data.get("graphs", []) if isinstance(data, dict) else data

            result: Dict[str, Dict[str, Any]] = {}
            for graph_entry in graphs:
                if isinstance(graph_entry, dict):
                    intent = graph_entry.get("intent")
                    graph_def = graph_entry.get("graph") or graph_entry
                    if intent:
                        result[intent] = graph_def

            return result
        except httpx.HTTPError as e:
            logger.error(f"Failed to list graphs: {e}")
            return {}

    async def get_graph_metadata(self, intent: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific graph.

        Args:
            intent: The intent to look up

        Returns:
            Graph metadata dictionary or None if not found
        """
        url = f"{self.graph_registry_url}/api/v1/graphs/{intent}/metadata"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Failed to get metadata for intent {intent}: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting metadata for intent {intent}: {e}")
            raise

    async def validate_graph(self, graph_definition: Dict[str, Any]) -> bool:
        """Validate a graph definition.

        Args:
            graph_definition: The graph definition to validate

        Returns:
            True if valid, False otherwise
        """
        url = f"{self.graph_registry_url}/api/v1/graphs/validate"
        try:
            response = await self.http_client.post(url, json=graph_definition)
            response.raise_for_status()
            return response.json().get("valid", False)
        except httpx.HTTPError as e:
            logger.error(f"Failed to validate graph: {e}")
            return False

    async def get_graphs_by_intent_model(self) -> GraphByIntent:
        """Get all graphs as a Pydantic model with intent as key.

        Returns:
            GraphByIntent model with graphs keyed by intent.
        """
        graphs = await self.list_graphs()
        return GraphByIntent(graphs=graphs)

    async def get_graph_by_intent_from_list(
        self, intent: str
    ) -> Optional[Dict[str, Any]]:
        """Get a graph definition by intent from the list of graphs.

        This method fetches all graphs and filters by intent, useful when
        the registry doesn't support individual graph lookups.

        Args:
            intent: The intent to look up

        Returns:
            Graph definition dictionary or None if not found
        """
        graphs = await self.list_graphs()
        return graphs.get(intent)


# Global client instance
_registry: Optional[GraphRegistryClient] = None


def get_graph_registry() -> GraphRegistryClient:
    """Get or create the global graph registry client.

    Returns:
        GraphRegistryClient instance
    """
    global _registry
    if _registry is None:
        _registry = GraphRegistryClient()
    return _registry
