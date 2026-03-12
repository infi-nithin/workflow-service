"""Dynamic Graph execution with LangGraph."""
import json
import time
import logging
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from agent.models import AgentState, NodeExecutionLog
from agent.tool_executor import get_tool_executor
from agent.llm_utils import get_llm_client


logger = logging.getLogger(__name__)


class GraphExecutor:
    """Dynamic graph executor using LangGraph.
    
    Builds and executes graphs dynamically based on graph definitions
    from the registry. Supports node types: mcp_tool, llm, sub_agent.
    """
    
    def __init__(
        self,
        tool_executor=None,
        llm_client=None,
    ):
        """Initialize the graph executor.
        
        Args:
            tool_executor: MCP Tool Executor instance
            llm_client: LLM Client instance
        """
        self.tool_executor = tool_executor or get_tool_executor()
        self.llm_client = llm_client or get_llm_client()
    
    def build_graph(
        self,
        graph_definition: Dict[str, Any],
    ) -> CompiledStateGraph:
        """Build a LangGraph from a graph definition.
        
        Args:
            graph_definition: Graph definition from the registry
            
        Returns:
            Compiled LangGraph
        """
        # Create state graph
        graph = StateGraph(AgentState)
        
        # Add nodes from definition
        nodes = graph_definition.get("nodes", [])
        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            
            # Create node function - pass the whole node dict for new format
            node_func = self._create_node_function(node_type, node)
            
            # Add node to graph
            graph.add_node(node_id, node_func)
        
        # Add edges from separate edges array (new format)
        edges = graph_definition.get("edges", [])
        
        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            
            if from_node and to_node:
                graph.add_edge(from_node, to_node)

        first_node = nodes[0]["id"]
        last_node = nodes[-1]["id"]
        graph.add_edge(START, first_node)
        graph.add_edge(last_node, END)
        
        # Compile the graph
        return graph.compile()
    
    def _create_node_function(
        self,
        node_type: str,
        node: Dict[str, Any],
    ) -> Callable:
        """Create a node function based on type.
        
        Args:
            node_type: Type of node (mcp_tool, llm, sub_agent)
            node: Node definition with all properties
            
        Returns:
            Node function
        """
        if node_type == "mcp_tool":
            return self._create_tool_node(node)
        elif node_type == "llm":
            return self._create_llm_node(node)
        elif node_type == "sub_agent":
            return self._create_sub_agent_node(node)
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    def _create_tool_node(self, node: Dict[str, Any]) -> Callable:
        """Create a tool node function.
        
        Args:
            node: Node definition with 'tool_name', 'input_mapping', etc. at top level
            
        Returns:
            Node function
        """
        # Get properties from node level (new format) or config (legacy format)
        tool_name = node.get("tool_name") or node.get("config", {}).get("tool_name")
        input_mapping = node.get("input_mapping") or node.get("config", {}).get("input_mapping", {})
        output_key = node.get("output_key") or node.get("config", {}).get("output_key", "tool_result")
        node_id = node.get("id", "tool_node")
        
        async def tool_node(state: AgentState) -> AgentState:
            """Execute a tool node."""
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="mcp_tool",
                duration_ms=0,
            )
            
            try:
                # Map inputs from state
                arguments = {}
                for param, source in input_mapping.items():
                    if isinstance(source, str) and source in state:
                        arguments[param] = state[source]
                    else:
                        arguments[param] = source
                
                # Execute tool
                result = await self.tool_executor.execute_tool(tool_name, arguments)
                
                # Update state
                state["node_outputs"][output_key] = result.dict() if hasattr(result, 'dict') else result
                
                node_log.tool_call = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "success": result.success if hasattr(result, 'success') else True,
                }
                
            except Exception as e:
                logger.error(f"Tool node execution failed: {e}")
                node_log.error = str(e)
                state["node_outputs"][output_key] = {"error": str(e)}
            
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            
            # Add to execution log
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, 'dict') else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            
            return state
        
        return tool_node
    
    def _create_llm_node(self, node: Dict[str, Any]) -> Callable:
        """Create an LLM node function.
        
        Args:
            node: Node definition with 'prompt_template', 'system_prompt', etc. at top level
            
        Returns:
            Node function
        """
        # Get properties from node level (new format) or config (legacy format)
        prompt_template = node.get("prompt_template") or node.get("config", {}).get("prompt_template")
        system_prompt = node.get("system_prompt") or node.get("config", {}).get("system_prompt", "")
        input_key = node.get("input_key") or node.get("config", {}).get("input_key", "raw_input")
        output_key = node.get("output_key") or node.get("config", {}).get("output_key", "llm_result")
        temperature = node.get("temperature") or node.get("config", {}).get("temperature", 0.7)
        node_id = node.get("id", "llm_node")
        
        async def llm_node(state: AgentState) -> AgentState:
            """Execute an LLM node."""
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="llm",
                duration_ms=0,
            )
            
            try:
                # Get input
                input_data = state.get(input_key, {})
                if isinstance(input_data, dict):
                    user_message = input_data.get("message", "") or json.dumps(input_data)
                else:
                    user_message = str(input_data)
                
                # Build the prompt: use prompt_template if provided, otherwise use system_prompt
                if prompt_template:
                    # Format the prompt template with state data
                    try:
                        formatted_prompt = prompt_template.format(**state)
                    except (KeyError, AttributeError):
                        # If formatting fails, use the template as-is
                        formatted_prompt = prompt_template
                    user_message = formatted_prompt
                
                # Build messages for ChatBedrock invoke
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=user_message))
                
                # Call LLM using invoke
                result = self.llm_client.invoke(messages)
                
                # Update state - ChatBedrock returns AIMessage with content attribute
                content = result.content if hasattr(result, 'content') else str(result)
                state["node_outputs"][output_key] = content
                state["final_output"] = content
                
                node_log.llm_call = {
                    "model": result.response_metadata.get("model") if hasattr(result, "response_metadata") else None,
                    "temperature": temperature,
                    "usage": None,  # ChatBedrock invoke doesn't return usage directly
                }
                
                # Update token count (not available with invoke)
                execution_log = state.get("execution_log", {})
                current_tokens = execution_log.get("total_tokens", 0)
                execution_log["total_tokens"] = current_tokens
                
                # Track model versions
                model_versions = execution_log.get("model_versions_used", [])
                model = result.response_metadata.get("model") if hasattr(result, "response_metadata") else None
                if model and model not in model_versions:
                    model_versions.append(model)
                execution_log["model_versions_used"] = model_versions
                state["execution_log"] = execution_log
                
            except Exception as e:
                logger.error(f"LLM node execution failed: {e}")
                node_log.error = str(e)
                state["node_outputs"][output_key] = {"error": str(e)}
            
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            
            # Add to execution log
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, 'dict') else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            
            return state
        
        return llm_node
    
    def _create_sub_agent_node(self, node: Dict[str, Any]) -> Callable:
        """Create a sub-agent node function.
        
        Args:
            node: Node definition with 'agent_name' at top level
            
        Returns:
            Node function
        """
        # Get properties from node level (new format) or config (legacy format)
        agent_name = node.get("agent_name") or node.get("config", {}).get("agent_name")
        node_id = node.get("id", agent_name or "sub_agent")
        
        async def sub_agent_node(state: AgentState) -> AgentState:
            """Execute a sub-agent node."""
            start_time = time.time()
            node_log = NodeExecutionLog(
                node_id=node_id,
                node_type="sub_agent",
                duration_ms=0,
            )
            
            try:
                # TODO: Implement sub-agent execution
                # For now, just pass through
                node_log.error = "Sub-agent execution not yet implemented"
                
            except Exception as e:
                logger.error(f"Sub-agent node execution failed: {e}")
                node_log.error = str(e)
            
            node_log.duration_ms = int((time.time() - start_time) * 1000)
            
            # Add to execution log
            execution_log = state.get("execution_log", {})
            nodes_log = execution_log.get("nodes", [])
            nodes_log.append(node_log.dict() if hasattr(node_log, 'dict') else node_log)
            execution_log["nodes"] = nodes_log
            state["execution_log"] = execution_log
            
            return state
        
        return sub_agent_node
    
    async def execute(
        self,
        initial_state: AgentState,
    ) -> AgentState:
        """Execute a graph with the given initial state.
        
        Args:
            graph_definition: Graph definition from the registry
            initial_state: Initial agent state
            
        Returns:
            Updated agent state after execution
        """
        try:
            graph_definition = initial_state.get("graph_definition")
            # Build the graph
            graph = self.build_graph(graph_definition)
            # Execute the graph
            result = await graph.ainvoke(initial_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            initial_state["execution_log"]["error"] = str(e)
            initial_state["final_output"] = {"error": str(e)}
            return initial_state


# Global executor instance
_executor: Optional[GraphExecutor] = None


def get_graph_executor() -> GraphExecutor:
    """Get or create the global graph executor.
    
    Returns:
        GraphExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = GraphExecutor()
    return _executor

