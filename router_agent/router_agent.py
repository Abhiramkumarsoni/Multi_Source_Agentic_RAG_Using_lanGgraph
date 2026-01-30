"""
Router-Based Agent - Semantic routing for intelligent tool selection
"""
import sys
from pathlib import Path
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration.llm import get_llm, get_llm_with_structured_output
from configuration.configuration import PDF_FILE, TEXT_FILE, URLS
from src.tools import (
    create_wikipedia_tool,
    create_arxiv_tool,
    create_duckgo_search_tool,
    create_url_retriever_tool,
    create_pdf_retriever_tool,
    create_text_retriever_tool,
)


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


class RouterAgent:
    """
    Router-based agent that uses semantic routing to select the best tool
    """
    
    def __init__(self):
        """Initialize the router agent"""
        print("[INFO] Initializing Router Agent...")
        self.llm = get_llm()
        self.tools = self._initialize_tools()
        self.router = self._create_router()
        print("[INFO] Router Agent initialized successfully")
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all available tools"""
        print("[INFO] Initializing tools...")
        
        tools = {
            "langgraph_docs": create_url_retriever_tool(URLS),
            "pdf_whitepaper": create_pdf_retriever_tool(PDF_FILE),
            "personal_info": create_text_retriever_tool(TEXT_FILE),
            "wikipedia": create_wikipedia_tool(),
            "arxiv": create_arxiv_tool(),
            "web_search": create_duckgo_search_tool()
        }
        
        # Filter out None values
        tools = {k: v for k, v in tools.items() if v is not None}
        print(f"[INFO] {len(tools)} tools initialized")
        
        return tools
    
    def _create_router(self):
        """Create the routing chain"""
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
        
        # Create chain
        router_chain = router_prompt | structured_llm
        
        return router_chain
    
    def _execute_tool(self, tool_name: str, query: str) -> str:
        """Execute the selected tool"""
        tool = self.tools.get(tool_name)
        
        if not tool:
            return f"Tool '{tool_name}' not available"
        
        try:
            print(f"[INFO] Executing tool: {tool_name}")
            result = tool.invoke(query)
            return result
        except Exception as e:
            print(f"[ERROR] Tool execution failed: {e}")
            return f"Error executing tool: {str(e)}"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate final answer using retrieved context"""
        prompt = PromptTemplate(
            template="""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say so. Keep the answer concise and informative.

Question: {question}

Context: {context}

Answer:""",
            input_variables=["question", "context"],
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"question": question, "context": context})
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query using semantic routing
        
        Returns:
            Dict with 'answer', 'route', and 'context'
        """
        print(f"\n[QUERY] {question}")
        
        # Step 1: Route the query
        route_result = self.router.invoke({"question": question})
        selected_route = route_result.datasource
        print(f"[ROUTE] Selected: {selected_route}")
        
        # Step 2: Execute the tool
        context = self._execute_tool(selected_route, question)
        
        # Step 3: Generate answer
        answer = self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "route": selected_route,
            "context": context[:500] + "..." if len(context) > 500 else context
        }
    
    def get_available_routes(self):
        """Get list of available routes"""
        return list(self.tools.keys())


def main():
    """Test the router agent"""
    agent = RouterAgent()
    
    # Test queries
    test_queries = [
        "Who is Abhiram Kumar Soni?",
        "What is LangGraph?",
        "What is agent quality?",
        "Who won the Nobel Prize in Physics in 2023?"
    ]
    
    for query in test_queries:
        result = agent.query(query)
        print(f"\nQuestion: {query}")
        print(f"Route: {result['route']}")
        print(f"Answer: {result['answer']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
