from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from datetime import datetime, timezone
from agent.workflow_service import WorkflowService
from agent.models import API
from api.v1.api_models import ChatRequest, ChatResponse

router = APIRouter()
workflow_service = WorkflowService()


@router.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/execute", response_model=ChatResponse)
async def execute(request: ChatRequest):
    try:
        # Create execute request with optional intent and graph_definition
        exec_request = API.Request(
            workflow_id=request.workflow_id,
            input_data=request.input_data,
            sys_id=request.input_data.get("sys_id", "MFS"),
            trace_id=request.trace_id,
            human_response=request.human_response,
        )
        # Execute agent
        result = await workflow_service.execute(exec_request)
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
