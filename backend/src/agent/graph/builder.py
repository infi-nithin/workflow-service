from langgraph.graph import StateGraph, END

from agent.models.models import SupervisorAgentState
from agent.graph.nodes.supervisor import SupervisorNode
from agent.graph.nodes.llm_node import LLMNode
from agent.graph.nodes.tool_node import ToolNode


class GraphBuilder:
    @staticmethod
    def build(
        supervisor_node: SupervisorNode,
        llm_node: LLMNode,
        tool_node: ToolNode,
    ) -> StateGraph:
        graph = StateGraph(SupervisorAgentState)

        # Add nodes
        graph.add_node("supervisor", supervisor_node.execute)
        graph.add_node("llm", llm_node.execute)
        graph.add_node("tool", tool_node.execute)

        # Set entry point
        graph.set_entry_point("supervisor")

        # Add edges from LLM and Tool back to supervisor
        graph.add_edge("llm", "supervisor")
        graph.add_edge("tool", "supervisor")

        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            supervisor_node.should_route,
            {
                "llm": "llm",
                "tool": "tool",
                "end": END,
            },
        )

        return graph.compile()

    @staticmethod
    def find_entry_point(graph_definition: dict) -> str:
        edges = graph_definition.get("edges", [])
        nodes = graph_definition.get("nodes", [])

        incoming = set()
        for edge in edges:
            incoming.add(edge.get("to"))

        for node in nodes:
            node_id = node.get("id")
            if node_id not in incoming:
                return node_id

        return nodes[0].get("id") if nodes else "START"
