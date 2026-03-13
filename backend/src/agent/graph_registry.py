import httpx
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class GraphByIntent(BaseModel):
    graphs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Graphs keyed by intent. Key is the intent string, value is the graph definition.",
    )


class GraphRegistryClient:
    def __init__(self):
        self.graph_registry_url = os.getenv("GRAPH_REGISTRY_URL")
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.http_client.aclose()

    async def get_graph_by_intent(self, intent: str) -> Optional[Dict[str, Any]]:
        url = f"{self.graph_registry_url}/api/v1/graphs/{intent}"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except httpx.HTTPError as e:
            raise

    async def list_graphs(self) -> Dict[str, Dict[str, Any]]:
        url = f"{self.graph_registry_url}/api/v1/graphs"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            data = response.json()
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
            return {}

    async def get_graph_metadata(self, intent: str) -> Optional[Dict[str, Any]]:
        url = f"{self.graph_registry_url}/api/v1/graphs/{intent}/metadata"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except httpx.HTTPError as e:
            raise

    async def validate_graph(self, graph_definition: Dict[str, Any]) -> bool:
        url = f"{self.graph_registry_url}/api/v1/graphs/validate"
        try:
            response = await self.http_client.post(url, json=graph_definition)
            response.raise_for_status()
            return response.json().get("valid", False)
        except httpx.HTTPError as e:
            return False

    async def get_graphs_by_intent_model(self) -> GraphByIntent:
        graphs = await self.list_graphs()
        return GraphByIntent(graphs=graphs)

    async def get_graph_by_intent_from_list(
        self, intent: str
    ) -> Optional[Dict[str, Any]]:
        graphs = await self.list_graphs()
        return graphs.get(intent)


_registry: Optional[GraphRegistryClient] = None


def get_graph_registry() -> GraphRegistryClient:
    global _registry
    if _registry is None:
        _registry = GraphRegistryClient()
    return _registry
