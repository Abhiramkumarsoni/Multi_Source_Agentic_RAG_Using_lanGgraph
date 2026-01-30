"""
Agent Module - Agentic RAG with custom graph architecture
Following agentic_rag_with_multiple_tools.ipynb
"""
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from configuration.configuration import PDF_FILE, TEXT_FILE, URLS
from src.graph.graph import create_graph
from src.tools import (
    create_wikipedia_tool,
    create_arxiv_tool,
    create_duckgo_search_tool,
    create_url_retriever_tool,
    create_pdf_retriever_tool,
    create_text_retriever_tool,
)


class AgenticRAGAgent:
    """Agentic RAG agent with custom graph workflow"""
    
    def __init__(self):
        """Initialize the agent with tools and custom graph"""
        self.tools = self._initialize_tools()
        self.graph = create_graph(self.tools)
        
    def _initialize_tools(self) -> List:
        """Initialize all available tools"""
        print("[INFO] Initializing tools...")
        
        tools_raw = [
            create_url_retriever_tool(URLS),
            create_pdf_retriever_tool(PDF_FILE),
            create_text_retriever_tool(TEXT_FILE),
            create_wikipedia_tool(),
            create_arxiv_tool(),
            create_duckgo_search_tool()
        ]
        
        tools = [t for t in tools_raw if t is not None]
        print(f"[INFO] {len(tools)} tools initialized successfully")
        
        if not tools:
            raise RuntimeError("No tools were initialized!")
        
        return tools
    
    def _extract_response(self, messages: list) -> str:
        """
        Extract the final response from messages.
        Matches notebook's ask_agent logic exactly.
        """
        # Find the last AI message that is NOT a tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                # Skip messages with tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue
                # Skip empty or tool-call formatted responses
                if content and isinstance(content, str):
                    content_stripped = content.strip()
                    # Skip if it looks like a tool call XML
                    if content_stripped.startswith('<') and '</function>' in content_stripped:
                        continue
                    if content_stripped:
                        return content_stripped
        
        return "Sorry, I couldn't generate a response."
    
    def query(self, question: str) -> str:
        """
        Process a query and return the response
        """
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 25}
        )
        
        messages = result.get("messages", [])
        return self._extract_response(messages)
    
    def query_with_details(self, question: str) -> tuple[str, dict]:
        """
        Process a query and return response with execution details
        
        Returns:
            Tuple of (response, details_dict)
        """
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 25}
        )
        
        messages = result.get("messages", [])
        
        # Extract tools used
        tools_used = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        # Extract response from the SAME result (don't run again!)
        response = self._extract_response(messages)
        
        details = {
            "tools_used": tools_used,
            "total_messages": len(messages)
        }
        
        return response, details
    
    def get_tool_count(self) -> int:
        """Get the number of initialized tools"""
        return len(self.tools)


def create_agent() -> AgenticRAGAgent:
    """Factory function to create an agent"""
    return AgenticRAGAgent()
