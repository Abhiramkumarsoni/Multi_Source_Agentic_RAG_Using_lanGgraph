"""
Arxiv Tool - Academic papers search
"""
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


def create_arxiv_tool():
    """
    Create an Arxiv search tool
    
    Returns:
        Arxiv tool or None if failed
    """
    try:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        print("✓ Arxiv Tool created")
        return arxiv_tool
    except Exception as e:
        print(f"✗ Arxiv tool failed: {e}")
        return None
