from typing import TypedDict, Any, Dict, List, Optional, Sequence
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class SupervisorAction(str, Enum):
    """Actions the supervisor can decide to take."""

    LLM = "llm"
    TOOL = "tool"
    END = "end"
    CONTINUE = "continue"


class SupervisorDecision(BaseModel):
    """Structured output for supervisor decision making."""

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
    """State that flows through the supervisor agent graph.

    Attributes:
        messages: List of conversation messages
        graph_definition: The fetched graph definition from registry
        current_node: Current node being executed
        execution_history: History of node executions
        tool_results: Results from tool executions
        workflow_id: Unique identifier for this workflow execution
        intent: Classified intent from the input
        decision: Current supervisor decision
        should_continue: Whether the agent should continue execution
        node_outputs: Dictionary of outputs from each node execution
        raw_input: The raw input data from the user request
    """

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
    """
    Main agent state that flows through execution.

    Attributes:
        workflow_id: Unique identifier for this workflow execution
        raw_input: The raw input data from the user request
        intent: Classified intent from the input
        graph_definition: The graph definition from the registry
        node_outputs: Dictionary of outputs from each node execution
        final_output: The final output after graph execution
        execution_log: Dictionary containing execution metadata and logs
    """

    workflow_id: str
    raw_input: Dict[str, Any]
    intent: str
    graph_definition: Dict[str, Any]
    node_outputs: Dict[str, Any]
    final_output: Any
    execution_log: Dict[str, Any]


class ExecuteRequest(BaseModel):
    """Request model for agent execution endpoint."""

    workflow_id: str = Field(..., description="Unique identifier for the workflow")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    intent: Optional[str] = Field(
        default=None, description="Pre-classified intent (optional)"
    )
    graph_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Graph definition (optional, will fetch from registry if not provided)",
    )


class ExecuteResponse(BaseModel):
    """Response model for agent execution endpoint."""

    result: Dict[str, Any] = Field(..., description="The execution result")
    execution_log: Dict[str, Any] = Field(
        ..., description="Execution metadata and logs"
    )


class NodeExecutionLog(BaseModel):
    """Log entry for a single node execution."""

    node_id: str
    node_type: str
    duration_ms: int
    tool_call: Optional[Dict[str, Any]] = None
    llm_call: Optional[Dict[str, Any]] = None
    sub_agent_call: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ExecutionLog(BaseModel):
    """Main execution log structure."""

    trace_id: str
    workflow_id: str
    graph_version: Optional[str] = None
    intent: Optional[str] = None
    model_versions_used: List[str] = Field(default_factory=list)
    total_tokens: int = 0
    nodes: List[NodeExecutionLog] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    status: str = "running"  # running, completed, failed


class GraphNode(BaseModel):
    """Represents a node in the graph definition."""

    id: str
    type: str  # mcp_tool, llm, sub_agent
    tool_name: Optional[str] = None
    prompt_template: Optional[str] = None
    agent_name: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge definition connecting two nodes."""

    from_: str = Field(..., alias="from")
    to: str

    class Config:
        populate_by_name = True


class GraphDefinition(BaseModel):
    """Graph definition from the registry."""

    version: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class GraphSubmission(BaseModel):
    """Graph submission from registry."""

    intent: str
    graph: GraphDefinition


class IntentListResponse(BaseModel):
    """Response for listing intents."""

    intents: List[str]
    total_count: int


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Response model for tool execution."""

    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
