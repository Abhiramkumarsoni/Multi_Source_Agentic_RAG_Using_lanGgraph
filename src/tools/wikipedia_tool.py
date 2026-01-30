"""
Wikipedia Tool - General knowledge search
"""
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def create_wikipedia_tool():
    """
    Create a Wikipedia search tool
    
    Returns:
        Wikipedia tool or None if failed
    """
    try:
        wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
        print("✓ Wikipedia Tool created")
        return wikipedia_tool
    except Exception as e:
        print(f"✗ Wikipedia tool failed: {e}")
        return None
