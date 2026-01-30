"""
LLM Configuration - Centralized LLM model initialization with dynamic API key support
"""
import os
from langchain_groq import ChatGroq
from configuration.configuration import DEFAULT_MODEL

# Module-level API key storage
_API_KEY = None


def set_api_key(api_key: str):
    """Set the API key dynamically (for Streamlit UI)"""
    global _API_KEY
    _API_KEY = api_key
    os.environ["GROQ_API_KEY"] = api_key


def get_api_key() -> str:
    """Get the current API key"""
    global _API_KEY
    return _API_KEY or os.getenv("GROQ_API_KEY")


def get_llm(model: str = None, temperature: float = 0):
    """
    Get a ChatGroq LLM instance
    
    Args:
        model: Model name (defaults to DEFAULT_MODEL from config)
        temperature: Temperature for generation (default: 0)
        
    Returns:
        ChatGroq instance
    """
    model_name = model or DEFAULT_MODEL
    api_key = get_api_key()
    
    if api_key:
        return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)
    else:
        return ChatGroq(model=model_name, temperature=temperature)


def get_llm_with_tools(tools: list, model: str = None, temperature: float = 0):
    """
    Get a ChatGroq LLM instance with tools bound
    
    Args:
        tools: List of tools to bind to the model
        model: Model name (defaults to DEFAULT_MODEL from config)
        temperature: Temperature for generation (default: 0)
        
    Returns:
        ChatGroq instance with tools bound
    """
    llm = get_llm(model, temperature)
    return llm.bind_tools(tools)


def get_llm_with_structured_output(output_schema, model: str = None, temperature: float = 0):
    """
    Get a ChatGroq LLM instance with structured output
    
    Args:
        output_schema: Pydantic model or schema for structured output
        model: Model name (defaults to DEFAULT_MODEL from config)
        temperature: Temperature for generation (default: 0)
        
    Returns:
        ChatGroq instance with structured output
    """
    llm = get_llm(model, temperature)
    return llm.with_structured_output(output_schema)
