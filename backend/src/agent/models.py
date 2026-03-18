from typing import TypedDict, Any, Dict, List, Optional, Sequence
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from config.config import config


class SupervisorAction(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    END = "end"
    CONTINUE = "continue"


class SupervisorDecision(BaseModel):
    action: SupervisorAction = Field(
        description="The action to take: 'llm', 'tool', 'end', or 'continue'"
    )
    reasoning: str = Field(description="Reasoning for the decision")
    tool_name: Optional[str] = Field(
        default=None, description="Name of tool to execute (if action is 'tool')"
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Arguments for tool (if action is 'tool')"
    )
    next_node: Optional[str] = Field(
        default=None,
        description="Next node to execute in the graph (if action is 'continue')",
    )
    response: Optional[str] = Field(
        default=None, description="Response to return to user (if action is 'end')"
    )
    llm_prompt: Optional[str] = Field(
        default=None, description="Prompt to give to LLM (if action is 'llm')"
    )


class SupervisorAgentState(TypedDict, total=False):
    trace_id: Optional[str]
    messages: Sequence[BaseMessage]
    graph_definition: Dict[str, Any]
    current_node: Optional[str]
    execution_history: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    workflow_id: Optional[str]
    intent: Optional[str]
    decision: Optional[SupervisorDecision]
    should_continue: bool
    node_outputs: Dict[str, Any]
    raw_input: Dict[str, Any]


class AgentState(TypedDict, total=False):
    trace_id: str
    workflow_id: str
    raw_input: Dict[str, Any]
    intent: str
    graph_definition: Dict[str, Any]
    node_outputs: Dict[str, Any]
    final_output: Any
    execution_log: Dict[str, Any]


class API:
    class Request(BaseModel):
        workflow_id: str = Field(..., description="Unique identifier for the workflow")
        input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
        sys_id: str = Field(default="MFS", description="The system id from which the request is coming")

    class Response(BaseModel):
        result: Dict[str, Any] = Field(..., description="The execution result")
        execution_log: Dict[str, Any] = Field(
            ..., description="Execution metadata and logs"
        )


class GraphNode(BaseModel):
    id: str
    type: str
    tool_name: Optional[str] = None
    prompt_template: Optional[str] = None
    agent_name: Optional[str] = None


class GraphEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str


class GraphDefinition(BaseModel):
    version: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class IntentListResponse(BaseModel):
    intents: List[str]
    total_count: int


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig:
    threshold: int = config.circuit_breaker.threshold
    cooldown: int = config.circuit_breaker.cooldown
