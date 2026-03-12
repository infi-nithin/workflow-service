from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from agent.service import AgentService
from agent.models import ExecuteRequest

router = APIRouter()

# Initialize agent service
agent_service = AgentService()


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(default="user")
    content: str


class ChatRequest(BaseModel):
    """Request model for chat execution endpoint."""

    workflow_id: str = Field(
        ..., description="Unique identifier for the workflow/thread"
    )
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    intent: Optional[str] = Field(
        default=None, 
        description="Pre-classified intent (optional, will be auto-classified if not provided)"
    )
    graph_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Graph definition (optional, will be fetched from registry if not provided)"
    )


class ChatResponse(BaseModel):
    """Response model for chat execution endpoint."""

    result: Dict[str, Any]
    execution_log: Dict[str, Any]


class IntentListResponse(BaseModel):
    """Response model for listing available intents."""

    intents: List[str]
    total_count: int


@router.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/execute", response_model=ChatResponse)
async def execute(request: ChatRequest):
    """
    Execute the generalized LangGraph agent with the given input.

    This endpoint uses a LangGraph supervisor agent that:
    1. Classifies the user intent (if not provided)
    2. Fetches the graph workflow from registry (if not provided)
    3. Uses an embedded LLM to decide what step to take next
    4. The LLM decides what prompt to give when making LLM calls
    5. The LLM decides what parameter values to give when making MCP tool calls
    6. Returns the result and execution log
    """
    try:
        # Create execute request with optional intent and graph_definition
        exec_request = ExecuteRequest(
            workflow_id=request.workflow_id, 
            input_data=request.input_data,
            intent=request.intent,
            graph_definition=request.graph_definition,
        )

        # Execute agent
        result = await agent_service.execute(exec_request)

        return ChatResponse(result=result.result, execution_log=result.execution_log)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    description: str = Form(""),
):
    """
    Upload a file for processing.

    This endpoint handles file uploads for the agent.
    The file can be processed by the agent based on the description.
    """
    try:
        # Read file content
        content = await file.read()

        # Process the file (this is a placeholder - implement as needed)
        # In production, you might save to storage, process with OCR, etc.

        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "thread_id": thread_id,
            "description": description,
            "message": "File uploaded successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}",
        )
