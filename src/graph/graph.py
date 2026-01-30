"""
Graph Compilation - Build and compile the LangGraph workflow
Following architecture from agentic_rag_with_multiple_tools.ipynb
"""
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.state.state_graph import AgentState
from src.nodes.nodes import agent, generate, rewrite
from src.edges.edges import grade_documents


def create_graph(tools: list):
    """
    Create and compile the LangGraph workflow with custom agentic architecture.
    
    Architecture:
    - Agent decides to use tools or end
    - Retrieve executes tools (ToolNode)
    - Grade checks document relevance
    - Generate creates answer from relevant docs
    - Rewrite reformulates query if docs not relevant
    
    Args:
        tools: List of available tools
        
    Returns:
        Compiled graph
    """
    # Define a new graph
    workflow = StateGraph(AgentState)
    
    # Define the nodes we will cycle between
    workflow.add_node("agent", lambda state: agent(state, tools))
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    
    # After retrieval, grade documents
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    
    # Connect generate and rewrite to end/agent
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    # Compile
    graph = workflow.compile()
    
    return graph
