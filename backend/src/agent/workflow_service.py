import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from agent.agent import OrchestrationAgent
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import config
from agent.models import (
    API,
    GraphNode,
    GraphEdge,
    GraphDefinition,
)
from db.database import db
from db.models import WorkflowExecution
from sqlalchemy import select, update

from agent.utils.llm_client import get_llm_client
from agent.utils.tool_registry_service import ToolRegistryService
from agent.utils.prompt_service import PromptService
from aop_logging import log_method

class WorkflowService:
    def __init__(self):
        self.graph_registry_url = config.registry.graph_registry_url
        self.model_id = config.aws.bedrock_model_id
        self.aws_secret_access_key = config.aws.secret_access_key
        self.aws_access_key_id = config.aws.access_key_id
        self.aws_session_token = config.aws.session_token
        self.region_name = config.aws.region
        self.llm = get_llm_client()
        self.tool_registry_service = ToolRegistryService()
        self.prompt_service = PromptService()

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
        pending_hitl: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            async with db.session() as session:
                execution = WorkflowExecution(
                    trace_id=trace_id,
                    workflow_id=workflow_id,
                    graph_version=graph_version,
                    intent=intent,
                    model_versions_used=[{"bedrock_model_id": self.model_id}],
                    total_tokens=0,
                    started_at=started_at,
                    completed_at=completed_at,
                    status=status,
                    duration_ms=duration_ms,
                    nodes=nodes,
                    error=error,
                    pending_hitl=pending_hitl,
                    current_state=current_state,
                )
                session.add(execution)

            return True

        except Exception:
            return False

    async def _update_execution_state(
        self,
        trace_id: str,
        status: str,
        pending_hitl: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update an existing execution record with new state."""
        try:
            async with db.session() as session:
                update_values = {
                    "status": status,
                    "pending_hitl": pending_hitl,
                    "current_state": current_state,
                }
                if nodes is not None:
                    update_values["nodes"] = nodes
                if error:
                    update_values["error"] = error
                if status in ("completed", "failed"):
                    update_values["completed_at"] = datetime.now(timezone.utc)
                
                await session.execute(
                    update(WorkflowExecution)
                    .where(WorkflowExecution.trace_id == trace_id)
                    .values(**update_values)
                )
                await session.commit()
            return True
        except Exception:
            return False

    async def _get_execution_by_trace_id(self, trace_id: str) -> Optional[WorkflowExecution]:
        """Get execution by trace_id."""
        try:
            async with db.session() as session:
                result = await session.execute(
                    select(WorkflowExecution).where(WorkflowExecution.trace_id == trace_id)
                )
                return result.scalar_one_or_none()
        except Exception:
            return None

    @log_method("WorkFlowService")
    async def _resume_from_hitl(self, request: API.Request, start_time: float) -> API.Response:
        """
        Resume execution from a pending HITL state.
        
        This is called when the user provides a human_response for a pending HITL.
        """
        trace_id = request.trace_id
        
        try:
            # Get the existing execution
            execution = await self._get_execution_by_trace_id(trace_id)
            if not execution:
                return API.Response(
                    result={"error": "Execution not found"},
                    execution_log={
                        "trace_id": trace_id,
                        "workflow_id": request.workflow_id,
                        "status": "failed",
                        "error": "Execution not found",
                    },
                )
            
            if not execution.pending_hitl:
                return API.Response(
                    result={"error": "No pending HITL for this execution"},
                    execution_log={
                        "trace_id": trace_id,
                        "workflow_id": request.workflow_id,
                        "status": "failed",
                        "error": "No pending HITL found",
                    },
                )
            
            # Get the saved state
            current_state = execution.current_state
            if not current_state:
                return API.Response(
                    result={"error": "Saved state not found"},
                    execution_log={
                        "trace_id": trace_id,
                        "workflow_id": request.workflow_id,
                        "status": "failed",
                        "error": "Saved state not found",
                    },
                )
            
            # Update the pending HITL with human response
            current_state["pending_hitl"]["human_response"] = request.human_response
            
            # Re-invoke the agent with the restored state
            tool_registry_url = await self.tool_registry_service.get_url(request.sys_id)
            agent = OrchestrationAgent(
                graph_definition=current_state.get("graph_definition"),
                llm=self.llm,
                tool_registry_url=tool_registry_url,
            )
            
            # Invoke with restored state - pass the restored state to the agent
            result = await agent.invoke_resume(current_state)
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            result_status = result.get("status", "completed")
            
            execution_log = {
                "trace_id": trace_id,
                "workflow_id": request.workflow_id,
                "graph_version": execution.graph_version,
                "intent": execution.intent,
                "model_versions_used": [self.model_id],
                "nodes": result.get("execution_history", []),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": result_status,
                "duration_ms": duration_ms,
                "resumed_from_hitl": True,
            }
            
            # Handle new HITL pending state
            pending_hitl = result.get("pending_hitl")
            new_current_state = result.get("current_state") if result_status == "pending_hitl" else None
            
            if result_status == "pending_hitl":
                # Update with new pending state
                await self._update_execution_state(
                    trace_id=trace_id,
                    status="pending_hitl",
                    pending_hitl=pending_hitl,
                    current_state=new_current_state,
                    nodes=result.get("execution_history", []),
                )
                return API.Response(
                    result={
                        "intent": execution.intent,
                        "output": result.get("output"),
                        "pending_hitl": pending_hitl,
                        "trace_id": trace_id,
                    },
                    execution_log=execution_log,
                )
            else:
                # Execution completed
                await self._update_execution_state(
                    trace_id=trace_id,
                    status=result_status,
                    pending_hitl=None,
                    current_state=None,
                    nodes=result.get("execution_history", []),
                )
                return API.Response(
                    result={
                        "intent": execution.intent,
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
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            return API.Response(
                result={"error": str(e)},
                execution_log=execution_log,
            )

    @log_method("WorkFlowService")
    async def get_available_intents(self) -> List[str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.graph_registry_url}/api/v1/intents")
            response.raise_for_status()
            data = response.json()
            return data.get("intents", [])

    @log_method("WorkFlowService")
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

    @log_method("WorkFlowService")
    async def classify_intent(
        self, user_input: str, available_intents: List[str]
    ) -> str:
        intents_str = (
            ", ".join(available_intents)
            if available_intents
            else "no intents available"
        )
        system_prompt = await self.prompt_service.get(
            "classify_intent_system", intents=intents_str
        )
        user_prompt = await self.prompt_service.get(
            "classify_intent_user", user_input=user_input
        )
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

    @log_method("WorkFlowService")
    async def execute(self, request: API.Request) -> API.Response:
        start_time = time.time()
        
        # Check if this is a HITL resume request
        if request.trace_id and request.human_response:
            return await self._resume_from_hitl(request, start_time)
        
        # Check if there's a pending HITL for this trace_id
        if request.trace_id:
            existing_execution = await self._get_execution_by_trace_id(request.trace_id)
            if existing_execution:
                if existing_execution.pending_hitl:
                    # There's a pending HITL but no human_response provided
                    return API.Response(
                        result={
                            "pending_hitl": existing_execution.pending_hitl,
                            "trace_id": request.trace_id,
                        },
                        execution_log={
                            "trace_id": request.trace_id,
                            "workflow_id": request.workflow_id,
                            "status": "pending_hitl",
                        },
                    )
                else:
                    # Trace exists but no pending HITL - return error
                    return API.Response(
                        result={
                            "error": "Execution exists but no pending HITL for this trace_id",
                            "trace_id": request.trace_id,
                        },
                        execution_log={
                            "trace_id": request.trace_id,
                            "workflow_id": request.workflow_id,
                            "status": "failed",
                            "error": "No pending HITL found for this trace_id",
                        },
                    )
            else:
                # Trace doesn't exist - return error
                return API.Response(
                    result={
                        "error": "Execution not found for trace_id",
                        "trace_id": request.trace_id,
                    },
                    execution_log={
                        "trace_id": request.trace_id,
                        "workflow_id": request.workflow_id,
                        "status": "failed",
                        "error": "Execution not found",
                    },
                )
        
        # Start fresh execution
        trace_id = str(uuid.uuid4())
        
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
            user_message = request.input_data.get("message", str(request.input_data))
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
            tool_registry_url = await self.tool_registry_service.get_url(request.sys_id)
            agent = OrchestrationAgent(
                graph_definition=graph_definition,
                llm=self.llm,
                tool_registry_url=tool_registry_url,
            )
            input_data = {
                "workflow_id": request.workflow_id,
                "raw_input": request.input_data,
                "intent": intent,
                "trace_id": trace_id,
            }
            result = await agent.invoke(input_data)
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            result_status = result.get("status", "completed")
            
            # Handle HITL pending state
            pending_hitl = result.get("pending_hitl")
            current_state = result.get("current_state") if result_status == "pending_hitl" else None
            
            execution_log = {
                "trace_id": trace_id,
                "workflow_id": request.workflow_id,
                "graph_version": graph_version,
                "intent": intent,
                "model_versions_used": [self.model_id],
                "nodes": result.get("execution_history", []),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": result_status,
                "duration_ms": duration_ms,
            }
            
            # Save execution to database
            started_at_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
            completed_at_dt = datetime.now(timezone.utc)
            
            if result_status == "pending_hitl":
                # Update existing record with pending state
                await self._update_execution_state(
                    trace_id=trace_id,
                    status="pending_hitl",
                    pending_hitl=pending_hitl,
                    current_state=current_state,
                    nodes=result.get("execution_history", []),
                )
                # Return pending HITL response
                return API.Response(
                    result={
                        "intent": intent,
                        "output": result.get("output"),
                        "pending_hitl": pending_hitl,
                        "trace_id": trace_id,
                    },
                    execution_log=execution_log,
                )
            else:
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
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            # Save failed execution to database
            started_at_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
            completed_at_dt = datetime.now(timezone.utc)
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
