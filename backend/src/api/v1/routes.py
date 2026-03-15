from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from agent.service import AgentService
from agent.models import ExecuteRequest
from api.v1.api_models import ChatRequest, ChatMessage, ChatResponse

router = APIRouter()
agent_service = AgentService()


@router.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/execute", response_model=ChatResponse)
async def execute(request: ChatRequest):
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
    try:
        content = await file.read()
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
