from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: str


class ChatRequest(BaseModel):
    workflow_id: str = Field(
        ..., description="Unique identifier for the workflow/thread"
    )
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    intent: Optional[str] = Field(
        default=None,
        description="Pre-classified intent (optional, will be auto-classified if not provided)",
    )
    graph_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Graph definition (optional, will be fetched from registry if not provided)",
    )


class ChatResponse(BaseModel):
    result: Dict[str, Any]
    execution_log: Dict[str, Any]
