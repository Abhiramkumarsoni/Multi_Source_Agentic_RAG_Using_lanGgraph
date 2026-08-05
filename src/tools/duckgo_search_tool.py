"""
DuckDuckGo Search Tool - Web search
"""
from langchain_community.tools import DuckDuckGoSearchRun


def create_duckgo_search_tool():
    """
    Create a DuckDuckGo search tool
    
    Returns:
        DuckDuckGo tool or None if failed
    """
    try:
        duckduckgo_tool = DuckDuckGoSearchRun()
        print("[OK] DuckDuckGo Tool created")
        return duckduckgo_tool
    except Exception as e:
        print(f"[FAIL] DuckDuckGo tool failed: {e}")
        return None
