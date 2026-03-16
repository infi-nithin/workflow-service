import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from agent.agent import OrchestrationAgent
import httpx
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import config
from agent.models import (
    API,
    GraphNode,
    GraphEdge,
    GraphDefinition,
)
from agent.prompts import (
    CLASSIFY_INTENT_SYSTEM_PROMPT,
    CLASSIFY_INTENT_USER_PROMPT,
)
from db.database import get_session, init_db
from db.models import WorkflowExecution


class WorkflowService:
    def __init__(self):
        self.graph_registry_url = config.registry.graph_registry_url
        self.tool_registry_url = config.registry.tool_registry_url
        self.mcp_server_url = config.registry.mcp_server_url
        self.aws_region = config.aws.region
        self.bedrock_model_id = config.aws.bedrock_model_id
        self.aws_secret_access_key = config.aws.secret_access_key
        self.aws_access_key_id = config.aws.access_key_id
        self.aws_session_token = config.aws.session_token
        self.llm = ChatBedrock(
            model_id=self.bedrock_model_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_access_key_id=self.aws_access_key_id,
            aws_session_token=self.aws_session_token,
            region_name=self.aws_region,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        )

    async def _save_execution_to_db(
        self,
        trace_id: str,
        workflow_id: str,
        graph_version: str,
        intent: str,
        status: str,
        started_at: datetime,
        completed_at: datetime,
        duration_ms: int,
        nodes: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> bool:
        try:
            # Initialize database if not already done
            await init_db(run_migrations=False)

            # Get database session using context manager approach
            async for session in get_session():
                # Create new workflow execution record
                execution = WorkflowExecution(
                    trace_id=trace_id,
                    workflow_id=workflow_id,
                    graph_version=graph_version,
                    intent=intent,
                    model_versions_used=[{"bedrock_model_id": self.bedrock_model_id}],
                    total_tokens=0,
                    started_at=started_at,
                    completed_at=completed_at,
                    status=status,
                    duration_ms=duration_ms,
                    nodes=nodes,
                    error=error,
                )

                session.add(execution)
                await session.commit()
                break

            return True

        except Exception as e:
            # Log error but don't fail the execution
            print(f"Failed to save execution to database: {str(e)}")
            return False

    async def get_available_intents(self) -> List[str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.graph_registry_url}/api/v1/intents")
            response.raise_for_status()
            data = response.json()
            return data.get("intents", [])

    async def get_graph_for_intent(self, intent: str) -> GraphDefinition:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.graph_registry_url}/api/v1/graphs/{intent}"
            )
            response.raise_for_status()
            data = response.json()
            graph_data = data.get("graph", {})
            return GraphDefinition(
                version=graph_data.get("version", "unknown"),
                nodes=[
                    GraphNode(**node) if isinstance(node, dict) else node
                    for node in graph_data.get("nodes", [])
                ],
                edges=[
                    GraphEdge(**edge) if isinstance(edge, dict) else edge
                    for edge in graph_data.get("edges", [])
                ],
            )

    async def classify_intent(
        self, user_input: str, available_intents: List[str]
    ) -> str:
        intents_str = (
            ", ".join(available_intents)
            if available_intents
            else "no intents available"
        )
        system_prompt = CLASSIFY_INTENT_SYSTEM_PROMPT.format(intents=intents_str)
        user_prompt = CLASSIFY_INTENT_USER_PROMPT.format(user_input=user_input)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        intent = response.content.strip().lower()
        if intent not in available_intents:
            for available in available_intents:
                if intent in available or available in intent:
                    intent = available
                    break
            else:
                intent = available_intents[0] if available_intents else "general_query"
        return intent

    async def execute(self, request: API.Request) -> API.Response:
        import asyncio

        trace_id = str(uuid.uuid4())
        start_time = time.time()
        try:
            available_intents = await self.get_available_intents()
            if not available_intents:
                return API.Response(
                    result={"error": "No intents available in registry"},
                    execution_log={
                        "trace_id": trace_id,
                        "workflow_id": request.workflow_id,
                        "status": "failed",
                        "error": "No intents available",
                    },
                )
            user_message = request.input_data.get(
                "message", str(request.input_data)
            )
            intent = await self.classify_intent(user_message, available_intents)
            graph_def = await self.get_graph_for_intent(intent)
            graph_version = graph_def.version
            graph_definition = {
                "version": graph_def.version,
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.type,
                        "tool_name": n.tool_name,
                        "prompt_template": n.prompt_template,
                        "agent_name": n.agent_name,
                    }
                    for n in graph_def.nodes
                ],
                "edges": [{"from": e.from_, "to": e.to} for e in graph_def.edges],
                "intent": intent,
            }
            agent = OrchestrationAgent(
                graph_definition=graph_definition,
                llm=self.llm,
                tool_registry_url=self.tool_registry_url,
            )
            input_data = {
                "workflow_id": request.workflow_id,
                "raw_input": request.input_data,
                "intent": intent,
                "trace_id": trace_id,
            }
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    executor = concurrent.futures.ThreadPoolExecutor()
                    result = executor.submit(agent.invoke, input_data).result()
                else:
                    result = agent.invoke(input_data)
            except Exception:
                result = agent.invoke(input_data)
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            result_status = result.get("status", "completed")
            execution_log = {
                "trace_id": trace_id,
                "workflow_id": request.workflow_id,
                "graph_version": graph_version,
                "intent": intent,
                "model_versions_used": [self.bedrock_model_id],
                "nodes": result.get("execution_history", []),
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "status": result_status,
                "duration_ms": duration_ms,
            }

            # Save execution to database
            started_at_dt = datetime.utcfromtimestamp(start_time)
            completed_at_dt = datetime.utcnow()
            await self._save_execution_to_db(
                trace_id=trace_id,
                workflow_id=request.workflow_id,
                graph_version=graph_version,
                intent=intent,
                status=result_status,
                started_at=started_at_dt,
                completed_at=completed_at_dt,
                duration_ms=duration_ms,
                nodes=result.get("execution_history", []),
            )

            return API.Response(
                result={
                    "intent": intent,
                    "output": result.get("output"),
                    "node_outputs": result.get("node_outputs"),
                },
                execution_log=execution_log,
            )
        except Exception as e:
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            execution_log = {
                "trace_id": trace_id,
                "workflow_id": request.workflow_id,
                "status": "failed",
                "error": str(e),
                "duration_ms": duration_ms,
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            }

            # Save failed execution to database
            started_at_dt = datetime.utcfromtimestamp(start_time)
            completed_at_dt = datetime.utcnow()
            await self._save_execution_to_db(
                trace_id=trace_id,
                workflow_id=request.workflow_id,
                graph_version="unknown",
                intent=request.intent or "unknown",
                status="failed",
                started_at=started_at_dt,
                completed_at=completed_at_dt,
                duration_ms=duration_ms,
                nodes=[],
                error=str(e),
            )

            return API.Response(
                result={"error": str(e)},
                execution_log=execution_log,
            )
