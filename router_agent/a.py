"""
Router Graph - Simple semantic routing graph for the Router Agent
This is different from the Agentic RAG graph - it uses a linear workflow:
Route → Retrieve → Generate
"""
from typing import Literal, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from configuration.llm import get_llm, get_llm_with_structured_output


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal[
        "langgraph_docs",
        "pdf_whitepaper", 
        "personal_info",
        "wikipedia",
        "arxiv",
        "web_search"
    ] = Field(
        description="Given a user question, choose which datasource would be most relevant for answering their question"
    )


class RouterState(TypedDict):
    """State for the router graph"""
    question: str
    selected_route: str
    context: str
    answer: str


def route_node(state: RouterState, tools: Dict[str, Any]) -> RouterState:
    """
    Node 1: Route the query to the most relevant datasource using semantic routing
    """
    question = state["question"]
    print(f"[ROUTE NODE] Routing query: {question}")
    
    # Router prompt
    router_prompt = PromptTemplate(
        template="""You are an expert at routing user questions to the appropriate data source.

Based on the question, route it to the most relevant source:

- langgraph_docs: Questions about LangGraph, LangChain workflows, state graphs, agents
- pdf_whitepaper: Questions about agent quality, evaluation, benchmarks
- personal_info: Questions about Abhiram Kumar Soni, personal information
- wikipedia: General knowledge questions, historical facts, definitions
- arxiv: Academic research papers, scientific topics
- web_search: Current events, recent information, general web queries

Question: {question}

Return only the datasource name.""",
        input_variables=["question"],
    )
    
    # Create structured output LLM
    structured_llm = get_llm_with_structured_output(RouteQuery)
    
    # Create and invoke chain
    router_chain = router_prompt | structured_llm
    route_result = router_chain.invoke({"question": question})
    selected_route = route_result.datasource
    
    print(f"[ROUTE NODE] Selected route: {selected_route}")
    
    return {
        **state,
        "selected_route": selected_route
    }


def retrieve_node(state: RouterState, tools: Dict[str, Any]) -> RouterState:
    """
    Node 2: Execute the selected tool to retrieve context
    """
    question = state["question"]
    selected_route = state["selected_route"]
    
    print(f"[RETRIEVE NODE] Executing tool: {selected_route}")
    
    tool = tools.get(selected_route)
    
    if not tool:
        context = f"Tool '{selected_route}' not available"
    else:
        try:
            context = tool.invoke(question)
        except Exception as e:
            print(f"[ERROR] Tool execution failed: {e}")
            context = f"Error executing tool: {str(e)}"
    
    print(f"[RETRIEVE NODE] Retrieved {len(context)} characters of context")
    
    return {
        **state,
        "context": context
    }


def generate_node(state: RouterState) -> RouterState:
    """
    Node 3: Generate final answer using retrieved context
    """
    question = state["question"]
    context = state["context"]
    
    print(f"[GENERATE NODE] Generating answer...")
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say so. Keep the answer concise and informative.

Question: {question}

Context: {context}

Answer:""",
        input_variables=["question", "context"],
    )
    
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({"question": question, "context": context})
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    print(f"[GENERATE NODE] Answer generated")
    
    return {
        **state,
        "answer": answer
    }


def create_router_graph(tools: Dict[str, Any]):
    """
    Create the router graph with a simple linear flow:
    START → route → retrieve → generate → END
    
    Args:
        tools: Dictionary mapping route names to tool instances
        
    Returns:
        Compiled LangGraph graph
    """
    print("[INFO] Creating router graph...")
    
    # Create the graph
    workflow = StateGraph(RouterState)
    
    # Add nodes with tool access
    workflow.add_node("route", lambda state: route_node(state, tools))
    workflow.add_node("retrieve", lambda state: retrieve_node(state, tools))
    workflow.add_node("generate", generate_node)
    
    # Define the linear flow
    workflow.add_edge(START, "route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    print("[INFO] Router graph created successfully")
    
    return graph
