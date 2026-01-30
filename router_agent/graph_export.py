"""
Graph Export - Exports the compiled graph for LangGraph Studio
"""
import sys
from pathlib import Path

# Add project root to Python path to allow imports from src and configuration
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configuration.configuration import PDF_FILE, TEXT_FILE, URLS
from src.tools import (
    create_wikipedia_tool,
    create_arxiv_tool,
    create_duckgo_search_tool,
    create_url_retriever_tool,
    create_pdf_retriever_tool,
    create_text_retriever_tool,
)
from src.graph.graph import create_graph


# Initialize tools at module level (required for LangGraph Studio)
print("[INFO] Initializing tools for LangGraph Studio...")
_tools_raw = [
    create_url_retriever_tool(URLS),
    create_pdf_retriever_tool(PDF_FILE),
    create_text_retriever_tool(TEXT_FILE),
    create_wikipedia_tool(),
    create_arxiv_tool(),
    create_duckgo_search_tool()
]
tools = [t for t in _tools_raw if t is not None]
print(f"[INFO] {len(tools)} tools initialized successfully")

if not tools:
    raise RuntimeError("No tools were initialized! Check your data files and network connection.")

# Create and export the compiled graph
graph = create_graph(tools)
