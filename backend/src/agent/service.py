import os
import json
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

import httpx
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.models import (
    ExecuteRequest,
    ExecuteResponse,
    GraphNode,
    GraphEdge,
    GraphDefinition,
    SupervisorAgentState,
    SupervisorDecision,
    SupervisorAction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def _create_supervisor_prompt(graph_definition: Dict[str, Any]) -> str:
    """Create the supervisor system prompt with graph structure.
    
    This prompt is injected with the graph workflow and guides the LLM
    to decide what step to take next based on the graph structure.
    
    Args:
        graph_definition: The graph definition from the registry
        
    Returns:
        System prompt string with graph structure injected
    """
    nodes = graph_definition.get("nodes", [])
    edges = graph_definition.get("edges", [])
    version = graph_definition.get("version", "unknown")
    intent = graph_definition.get("intent", "unknown")
    
    # Format nodes for the prompt
    nodes_description = []
    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "unknown")
        
        # Find next nodes from edges
        next_nodes = [edge.get("to") for edge in edges if edge.get("from") == node_id]
        
        node_desc = f"- Node: {node_id}, Type: {node_type}"
        if next_nodes:
            node_desc += f", Can proceed to: {', '.join(next_nodes)}"
        
        # Add node-specific config
        if node.get("tool_name"):
            node_desc += f", Tool: {node.get('tool_name')}"
        if node.get("prompt_template"):
            node_desc += f", Has prompt template"
        if node.get("agent_name"):
            node_desc += f", Agent: {node.get('agent_name')}"
            
        nodes_description.append(node_desc)
    
    nodes_text = "\n".join(nodes_description) if nodes_description else "No nodes defined"
    
    prompt = f"""You are a supervisor agent that orchestrates graph-based agent execution.

## Graph Workflow
- Intent: {intent}
- Version: {version}
- Available Nodes:
{nodes_text}

## Your Role
You are responsible for deciding what action to take next in the execution flow.
You have access to the graph structure and must decide whether to:
1. Call the LLM to process information or generate responses
2. Execute a tool from the tool registry  
3. Continue to the next node in the graph
4. End the execution

## Decision Making (Non-deterministic)
The graph provides a workflow, but YOU decide:
- Whether to execute a node now or skip to another
- What specific parameters to pass to tools (based on context)
- What prompt to give the LLM when processing
- Whether to branch to different paths based on results

## Instructions
1. Start from the entry point of the graph (first node with no incoming edges)
2. When a node requires LLM processing, use the 'llm' action and provide a specific prompt
3. When a node requires tool execution, use the 'tool' action with:
   - tool_name: The exact name of the tool
   - tool_arguments: A dictionary of parameters extracted from context
4. After tool execution, analyze results and decide next action
5. When all required work is complete, use 'end' action

## Decision Format
You must respond with a JSON object containing:
- action: One of 'llm', 'tool', 'end', or 'continue'
- reasoning: Why you made this decision based on current state
- tool_name: (optional) Name of tool to execute
- tool_arguments: (optional) Arguments for the tool - extract from context
- llm_prompt: (optional) Specific prompt to give LLM when action is 'llm'
- next_node: (optional) Next node to execute (for 'continue')
- response: (optional) Final response to return

## Tool Parameter Extraction
When calling tools, extract parameter values from:
- The original user input
- Previous tool execution results
- LLM responses in the conversation history
Provide specific values, not placeholders.

## Non-deterministic Behavior
You may choose different paths based on:
- Results from previous executions
- Analysis of intermediate results
- User input context
The graph is a guide, you make the final decision.

Always respond with valid JSON."""
    
    return prompt


class LangGraphSupervisorAgent:
    """LangGraph Supervisor Agent that orchestrates graph execution with embedded LLM.
    
    This agent uses LangGraph to create a state machine where:
    1. A supervisor node uses an embedded LLM to decide actions
    2. The LLM decides what step to take next based on the graph workflow
    3. The LLM decides what prompt to give when making LLM calls
    4. The LLM decides what parameter values to give when making MCP tool calls
    """
    
    def __init__(
        self,
        graph_definition: Dict[str, Any],
        llm: ChatBedrock,
        tool_registry_url: str
    ):
        """Initialize the supervisor agent.
        
        Args:
            graph_definition: The graph definition from the registry
            llm: The Bedrock LLM instance
            tool_registry_url: URL of the tool registry
        """
        self.graph_definition = graph_definition
        self.llm = llm
        self.tool_registry_url = tool_registry_url
        self.mcp_server_url = self.tool_registry_url+"/mcp"
        self.intent = graph_definition.get("intent", "unknown")
        
        # Create the supervisor prompt with graph structure
        self.system_prompt = _create_supervisor_prompt(graph_definition)
        
        # Build the LangGraph
        self.graph = self._build_graph()
        
        # Get entry point
        self.entry_point = self._find_entry_point()
        
        # Initialize MCP client (lazy initialization)
        self._mcp_client = None
    
    def _find_entry_point(self) -> str:
        """Find the entry point node (node with no incoming edges)."""
        edges = self.graph_definition.get("edges", [])
        nodes = self.graph_definition.get("nodes", [])
        
        # Find nodes with incoming edges
        incoming = set()
        for edge in edges:
            incoming.add(edge.get("to"))
        
        # Entry point is a node with no incoming edges
        for node in nodes:
            node_id = node.get("id")
            if node_id not in incoming:
                return node_id
        
        # Fallback to first node
        return nodes[0].get("id") if nodes else "START"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.
        
        Returns:
            Compiled StateGraph for the supervisor agent
        """
        graph = StateGraph(SupervisorAgentState)
        
        # Add nodes
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("llm", self.llm_node)
        graph.add_node("tool", self.tool_node)
        
        # Set entry point
        graph.set_entry_point("supervisor")
        
        # Add edges - return to supervisor after llm or tool
        graph.add_edge("llm", "supervisor")
        graph.add_edge("tool", "supervisor")
        
        # Conditional routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._should_route,
            {
                "llm": "llm",
                "tool": "tool",
                "end": END,
            }
        )
        
        return graph.compile()
    
    def _should_route(self, state: SupervisorAgentState) -> str:
        """Determine where to route after supervisor decision.
        
        Args:
            state: Current agent state
            
        Returns:
            Route key: 'llm', 'tool', or 'end'
        """
        decision = state.get("decision")
        if not decision:
            return "end"
        
        action = decision.action
        if action == SupervisorAction.LLM:
            return "llm"
        elif action == SupervisorAction.TOOL:
            return "tool"
        else:
            return "end"
    
    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the tool registry.
        
        Returns:
            List of available tools
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.tool_registry_url}/api/v1/mcp/tools")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("tools", [])
        except Exception as e:
            logger.warning(f"Could not fetch tools from registry: {e}")
        return []
    
    def supervisor_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        """Supervisor node that uses LLM to decide the next action.
        
        This is where the embedded LLM decides:
        - What step to do next based on the graph workflow
        - What prompt to give when making LLM calls
        - What parameter values to give when making MCP tool calls
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with decision
        """
        messages = state.get("messages", [])
        current_node = state.get("current_node", self.entry_point)
        execution_history = state.get("execution_history", [])
        graph_definition = state.get("graph_definition", self.graph_definition)
        node_outputs = state.get("node_outputs", {})
        
        logger.info(f"Supervisor node executing. Current node: {current_node}")
        
        # Find the current node definition
        nodes = graph_definition.get("nodes", [])
        current_node_def = None
        for node in nodes:
            if node.get("id") == current_node:
                current_node_def = node
                break
        
        # Get available tools synchronously
        try:
            # Create new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                available_tools = loop.run_until_complete(self._get_available_tools())
            finally:
                loop.close()
        except Exception:
            available_tools = []
        
        # Create the supervisor prompt with current state
        supervisor_prompt = self._create_supervisor_decision_prompt(
            current_node=current_node,
            current_node_def=current_node_def,
            messages=messages,
            execution_history=execution_history,
            available_tools=available_tools,
            node_outputs=node_outputs,
            graph_definition=graph_definition,
        )
        
        # Call the LLM to make a decision
        try:
            # Use structured output with the LLM
            llm_with_structure = self.llm.with_structured_output(SupervisorDecision)
            
            decision_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=supervisor_prompt),
            ]
            
            # Get the decision from LLM - use sync call (bedrock boto3 is thread-safe)
            decision = llm_with_structure.invoke(decision_messages)
            
            logger.info(f"Supervisor decision: {decision.action} - {decision.reasoning}")
            
            # Update state with decision
            updates: Dict[str, Any] = {
                "decision": decision,
                "current_node": decision.next_node or current_node,
                "should_continue": decision.action not in (SupervisorAction.END,),
            }
            
            # Add to execution history
            execution_history = list(execution_history) if execution_history else []
            execution_history.append({
                "node": "supervisor",
                "timestamp": datetime.utcnow().isoformat(),
                "decision": decision.model_dump(),
            })
            updates["execution_history"] = execution_history
            
            return updates
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {e}")
            # Default to ending on error
            return {
                "should_continue": False,
                "decision": SupervisorDecision(
                    action=SupervisorAction.END,
                    reasoning=f"Error in supervisor: {str(e)}",
                    response="An error occurred during execution."
                ),
                "execution_history": execution_history + [{
                    "node": "supervisor",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                }] if execution_history else [{
                    "node": "supervisor",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                }],
            }
    
    def _create_supervisor_decision_prompt(
        self,
        current_node: str,
        current_node_def: Optional[Dict[str, Any]],
        messages: List[Any],
        execution_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        node_outputs: Dict[str, Any],
        graph_definition: Dict[str, Any],
    ) -> str:
        """Create the prompt for supervisor decision making.
        
        This is where the LLM gets the context to decide:
        - What action to take next
        - What parameters to pass to tools
        - What prompt to give to subsequent LLM calls
        
        Args:
            current_node: The current node being executed
            current_node_def: The current node definition
            messages: Conversation messages
            execution_history: History of executions
            available_tools: List of available tools
            node_outputs: Outputs from previous node executions
            graph_definition: The full graph definition
            
        Returns:
            Prompt string for decision making
        """
        # Format recent messages
        recent_messages = []
        for msg in messages[-5:]:  # Last 5 messages
            if isinstance(msg, HumanMessage):
                recent_messages.append(f"Human: {msg.content[:200]}...")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    recent_messages.append(f"AI (tool call): {msg.tool_calls}")
                else:
                    recent_messages.append(f"AI: {msg.content[:200]}...")
            elif isinstance(msg, ToolMessage):
                recent_messages.append(f"Tool ({msg.tool_call_id}): {msg.content[:200]}...")
        
        messages_text = "\n".join(recent_messages) if recent_messages else "No messages yet"
        
        # Format available tools
        tools_text = ""
        if available_tools:
            tools_info = []
            for tool in available_tools[:10]:  # Limit to 10 tools
                tools_info.append(f"- {tool.get('name')}: {tool.get('description', 'No description')}")
            tools_text = "\nAvailable tools:\n" + "\n".join(tools_info)
        
        # Format execution history
        history_text = ""
        if execution_history:
            history_entries = []
            for entry in execution_history[-5:]:  # Last 5 entries
                node = entry.get("node", "unknown")
                timestamp = entry.get("timestamp", "")
                history_entries.append(f"- {node} at {timestamp}")
            history_text = "\nExecution history:\n" + "\n".join(history_entries)
        
        # Format node outputs
        outputs_text = ""
        if node_outputs:
            outputs_info = []
            for node_id, output in node_outputs.items():
                outputs_info.append(f"- {node_id}: {json.dumps(output, indent=2)[:300]}...")
            outputs_text = "\nPrevious node outputs:\n" + "\n".join(outputs_info)
        
        # Node information
        node_info = ""
        if current_node_def:
            node_type = current_node_def.get("type", "unknown")
            tool_name = current_node_def.get("tool_name")
            prompt_template = current_node_def.get("prompt_template")
            agent_name = current_node_def.get("agent_name")
            
            node_info = f"""
Current Node: {current_node}
Node Type: {node_type}
"""
            if tool_name:
                node_info += f"Requires Tool: {tool_name}\n"
            if prompt_template:
                node_info += f"Has Prompt Template: {prompt_template[:100]}...\n"
            if agent_name:
                node_info += f"Requires Agent: {agent_name}\n"
        
        # Find possible next nodes
        edges = graph_definition.get("edges", [])
        possible_next = [edge.get("to") for edge in edges if edge.get("from") == current_node]
        if possible_next:
            node_info += f"Possible next nodes: {', '.join(possible_next)}\n"
        
        prompt = f"""Current state:
{node_info}

Recent conversation:
{messages_text}
{outputs_text}
{tools_text}
{history_text}

Based on the graph workflow above and current state, decide what action to take next.
Think about:
1. What does the current node require?
2. What information do you have from previous executions?
3. What parameters can you extract from the context to pass to tools?
4. What should the LLM process next?

Remember: You make the final decision - the graph is a guide, you decide the execution path.

Respond with a JSON object containing your decision."""
        
        return prompt
    
    def llm_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        """LLM node that processes information with a specific prompt.
        
        The supervisor decides what prompt to give to the LLM.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with LLM response
        """
        messages = list(state.get("messages", [])) if state.get("messages") else []
        decision = state.get("decision")
        execution_history = list(state.get("execution_history", [])) if state.get("execution_history") else []
        node_outputs = dict(state.get("node_outputs", {})) if state.get("node_outputs") else {}
        
        logger.info("LLM node executing")
        
        if not decision or decision.action != SupervisorAction.LLM:
            logger.warning("LLM node called but no LLM decision in state")
            return {"messages": messages, "node_outputs": node_outputs}
        
        # Get the prompt from decision (set by supervisor)
        llm_prompt = decision.llm_prompt or decision.response or "Process the following task"
        
        try:
            # Build context from previous node outputs
            context_parts = []
            for node_id, output in node_outputs.items():
                context_parts.append(f"=== {node_id} ===\n{json.dumps(output, indent=2)}")
            
            context = "\n\n".join(context_parts)
            
            # Build full prompt with context
            full_prompt = f"""Context from previous steps:
{context}

Task: {llm_prompt}"""

            # Add system and user messages
            system_msg = SystemMessage(content="You are a corporate actions expert assistant.")
            user_msg = HumanMessage(content=full_prompt)
            
            # Call the LLM synchronously - it uses boto3 which is thread-safe
            response = self.llm.invoke([system_msg, user_msg])
            
            # Add response to messages
            messages = list(messages)
            messages.append(AIMessage(content=response.content))
            
            # Store output from this LLM call
            current_node = state.get("current_node", "llm_node")
            node_outputs[current_node] = {"type": "llm", "response": response.content}
            
            # Add to execution history
            execution_history.append({
                "node": "llm",
                "timestamp": datetime.utcnow().isoformat(),
                "prompt": llm_prompt,
                "response": response.content[:500],
            })
            
            logger.info(f"LLM response: {response.content[:100]}...")
            
            return {
                "messages": messages,
                "node_outputs": node_outputs,
                "execution_history": execution_history,
            }
            
        except Exception as e:
            logger.error(f"Error in LLM node: {e}")
            execution_history.append({
                "node": "llm",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            })
            return {
                "messages": messages + [AIMessage(content=f"Error: {str(e)}")],
                "node_outputs": node_outputs,
                "execution_history": execution_history,
            }
    
    async def _get_mcp_client(self) -> Optional[MultiServerMCPClient]:
        """Get or create the MCP client.
        
        Returns:
            MultiServerMCPClient instance or None if not configured
        """
        if not self.mcp_server_url:
            return None
        
        if self._mcp_client is None:
            try:
                # Configure MCP server connection
                config = {
                    "tool-registry": {
                        "url": self.mcp_server_url,
                        "transport": "streamable-http",
                    }
                }
                self._mcp_client = MultiServerMCPClient(config)
                logger.info(f"MCP client connected to {self.mcp_server_url}")
            except Exception as e:
                logger.error(f"Failed to create MCP client: {e}")
                return None
        
        return self._mcp_client
    
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool via the MultiServerMCPClient.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        # Try to get MCP client
        mcp_client = await self._get_mcp_client()
        
        try:
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"fetched mcp tools: {mcp_tools}")
            tool = next(t for t in mcp_tools if t.name == tool_name)
            logger.info(f"Executing MCP tool: {tool_name} with args: {arguments}")
            result = await tool.ainvoke(arguments)
            logger.info(f"Executed MCP tool '{tool_name}' and the result was '{str(result)}'")
            if hasattr(result, 'content'):
                return {"result": result.content}
            return {"result": str(result)}
            
        except Exception as e:
            logger.error(f"Error executing MCP tool '{tool_name}': {e}")
    
    def tool_node(self, state: SupervisorAgentState) -> Dict[str, Any]:
        """Tool node that executes MCP tools.
        
        The supervisor decides what parameters to pass to the tool.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with tool execution results
        """
        messages = list(state.get("messages", [])) if state.get("messages") else []
        decision = state.get("decision")
        tool_results = dict(state.get("tool_results", {})) if state.get("tool_results") else {}
        execution_history = list(state.get("execution_history", [])) if state.get("execution_history") else []
        node_outputs = dict(state.get("node_outputs", {})) if state.get("node_outputs") else {}
        raw_input = dict(state.get("raw_input", {})) if state.get("raw_input") else {}
        
        logger.info("Tool node executing")
        
        if not decision or decision.action != SupervisorAction.TOOL:
            logger.warning("Tool node called but no tool decision in state")
            return {"messages": messages, "tool_results": tool_results, "node_outputs": node_outputs}
        
        tool_name = decision.tool_name
        tool_arguments = decision.tool_arguments or {}
        
        if not tool_name:
            logger.warning("Tool decision but no tool name specified")
            return {"messages": messages, "tool_results": tool_results, "node_outputs": node_outputs}
        
        # Merge context: raw_input + previous outputs + tool arguments
        # The supervisor has already decided what parameters to use
        merged_arguments = {
            **raw_input,
            **node_outputs,
            **tool_arguments,
        }
        
        # Call the tool - run async method in event loop
        try:
            # Create new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._call_mcp_tool(tool_name, merged_arguments))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error calling tool: {e}")
            result = {"error": str(e), "tool_name": tool_name}
        
        # Store tool result
        tool_results[tool_name] = result
        
        # Also store in node_outputs for context
        current_node = state.get("current_node", tool_name)
        node_outputs[current_node] = {"type": "tool", "tool_name": tool_name, "result": result}
        
        # Add to messages for context
        messages = list(messages)
        messages.append(ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_name
        ))
        
        # Add to execution history
        execution_history.append({
            "node": "tool",
            "timestamp": datetime.utcnow().isoformat(),
            "tool_name": tool_name,
            "arguments": merged_arguments,
            "result": str(result)[:500],
        })
        
        logger.info(f"Tool {tool_name} executed")
        
        return {
            "messages": messages,
            "tool_results": tool_results,
            "node_outputs": node_outputs,
            "execution_history": execution_history,
        }
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the supervisor agent with input data.
        
        Args:
            input_data: The input data including raw_input and intent
            
        Returns:
            Execution result with final output and execution history
        """
        # Initialize state
        initial_state: SupervisorAgentState = {
            "messages": [],
            "graph_definition": self.graph_definition,
            "current_node": self.entry_point,
            "execution_history": [],
            "tool_results": {},
            "workflow_id": input_data.get("workflow_id", str(uuid.uuid4())),
            "intent": self.intent,
            "should_continue": True,
            "node_outputs": {},
            "raw_input": input_data.get("raw_input", input_data),
        }
        
        # Add initial user message
        user_message = input_data.get("raw_input", {}).get("message", str(input_data))
        initial_state["messages"] = [HumanMessage(content=user_message)]
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Extract final output
            final_output = None
            decision = result.get("decision")
            if decision and decision.response:
                final_output = decision.response
            elif result.get("node_outputs"):
                # Get last node output
                last_output = list(result["node_outputs"].values())[-1]
                final_output = last_output.get("result") or last_output.get("response")
            
            return {
                "intent": self.intent,
                "output": final_output,
                "node_outputs": result.get("node_outputs", {}),
                "execution_history": result.get("execution_history", []),
                "status": "completed",
            }
            
        except Exception as e:
            logger.error(f"Error invoking supervisor agent: {e}")
            return {
                "intent": self.intent,
                "output": {"error": str(e)},
                "status": "failed",
            }


class AgentService:
    """
    Generalized LangGraph Agent Service.
    
    This agent uses LangGraph to create a state machine with an embedded LLM
    that decides what step to take next based on the graph workflow.
    
    Key features:
    1. Non-deterministic execution - LLM decides the path
    2. Graph workflow is injected into the prompt
    3. LLM decides what prompt to give when making LLM calls
    4. LLM decides what parameter values to give when making MCP tool calls
    """
    
    def __init__(self):
        """Initialize the agent service."""
        self.graph_registry_url = os.getenv("GRAPH_REGISTRY_URL", "http://localhost:8002")
        self.tool_registry_url = os.getenv("TOOL_REGISTRY_URL", "http://localhost:8001")
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
        self.aws_region = os.getenv("AWS_REGION")
        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        
        # Initialize LLM
        self.llm = ChatBedrock(
            model_id=self.bedrock_model_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_access_key_id=self.aws_access_key_id,
            aws_session_token=self.aws_session_token,
            region_name=self.aws_region,
            model_kwargs={
                "temperature": 0.7,  # Higher temperature for non-deterministic behavior
                "max_tokens": 4096,
            },
        )
    
    async def get_available_intents(self) -> List[str]:
        """Fetch all available intents from the graph registry."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.graph_registry_url}/api/v1/intents")
            response.raise_for_status()
            data = response.json()
            return data.get("intents", [])
    
    async def get_graph_for_intent(self, intent: str) -> GraphDefinition:
        """Fetch the graph definition for a given intent."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.graph_registry_url}/api/v1/graphs/{intent}")
            response.raise_for_status()
            data = response.json()
            
            # Parse the graph definition
            graph_data = data.get("graph", {})
            return GraphDefinition(
                version=graph_data.get("version", "unknown"),
                nodes=[GraphNode(**node) if isinstance(node, dict) else node for node in graph_data.get("nodes", [])],
                edges=[GraphEdge(**edge) if isinstance(edge, dict) else edge for edge in graph_data.get("edges", [])]
            )
    
    async def classify_intent(self, user_input: str, available_intents: List[str]) -> str:
        """
        Use LLM to classify the user intent from available intents.
        
        Args:
            user_input: The user input message
            available_intents: List of available intents from registry
            
        Returns:
            The best matching intent
        """
        intents_str = ", ".join(available_intents) if available_intents else "no intents available"
        
        system_prompt = """You are an intent classifier. Given a user message and a list of available intents,
you must select the single best matching intent. If no intent matches well, return 'general_query'.

Available intents: {intents}

Respond ONLY with the intent name, nothing else."""

        user_prompt = f"""User message: {user_input}

Select the best matching intent from the available intents. 
Consider the semantic meaning of the user message and match it to the most appropriate intent.
If the message is a general greeting or doesn't match any specific intent, return 'general_query'."""

        messages = [
            SystemMessage(content=system_prompt.format(intents=intents_str)),
            HumanMessage(content=user_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        intent = response.content.strip().lower()
        
        # Validate the intent is in the list
        if intent not in available_intents:
            # Try to find a fuzzy match
            for available in available_intents:
                if intent in available or available in intent:
                    intent = available
                    break
            else:
                intent = available_intents[0] if available_intents else "general_query"
        
        return intent
    
    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """
        Execute the generalized LangGraph agent with the given request.
        
        This method:
        1. Creates a LangGraph supervisor agent with the graph workflow
        2. The embedded LLM decides what step to take next
        3. Returns result and execution log
        
        Args:
            request: The execution request
            
        Returns:
            ExecuteResponse with result and execution log
        """
        import asyncio
        
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Step 1: Get intent (either provided or classify)
            intent = request.intent
            if not intent:
                available_intents = await self.get_available_intents()
                if not available_intents:
                    return ExecuteResponse(
                        result={"error": "No intents available in registry"},
                        execution_log={
                            "trace_id": trace_id,
                            "workflow_id": request.workflow_id,
                            "status": "failed",
                            "error": "No intents available",
                        }
                    )
                
                user_message = request.input_data.get("message", str(request.input_data))
                intent = await self.classify_intent(user_message, available_intents)
            
            # Step 2: Get graph definition (either provided or fetch from registry)
            graph_definition = request.graph_definition
            if not graph_definition:
                graph_def = await self.get_graph_for_intent(intent)
                graph_definition = {
                    "version": graph_def.version,
                    "nodes": [{"id": n.id, "type": n.type, "tool_name": n.tool_name, 
                               "prompt_template": n.prompt_template, "agent_name": n.agent_name} 
                              for n in graph_def.nodes],
                    "edges": [{"from": e.from_, "to": e.to} for e in graph_def.edges],
                    "intent": intent,
                }
            
            # Step 3: Create LangGraph supervisor agent
            agent = LangGraphSupervisorAgent(
                graph_definition=graph_definition,
                llm=self.llm,
                tool_registry_url=self.tool_registry_url,
            )
            
            # Step 4: Invoke the agent
            input_data = {
                "workflow_id": request.workflow_id,
                "raw_input": request.input_data,
                "intent": intent,
            }
            
            # Run in async context
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
            
            # Build execution log
            execution_log = {
                "trace_id": trace_id,
                "workflow_id": request.workflow_id,
                "graph_version": graph_definition.get("version"),
                "intent": intent,
                "model_versions_used": [self.bedrock_model_id],
                "nodes": result.get("execution_history", []),
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "status": result.get("status", "completed"),
                "duration_ms": duration_ms,
            }
            
            return ExecuteResponse(
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
            
            return ExecuteResponse(
                result={"error": str(e)},
                execution_log={
                    "trace_id": trace_id,
                    "workflow_id": request.workflow_id,
                    "status": "failed",
                    "error": str(e),
                    "duration_ms": duration_ms,
                }
            )
