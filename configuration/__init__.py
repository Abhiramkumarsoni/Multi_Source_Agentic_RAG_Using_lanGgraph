"""
Configuration package - Environment variables, paths, and LLM settings
"""
from .configuration import (
    GROQ_API_KEY,
    LANGSMITH_API_KEY,
    PROJECT_ROOT,
    DATA_DIR,
    PDF_FILE,
    TEXT_FILE,
    URLS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    RECURSION_LIMIT,
    WIKIPEDIA_TOP_K,
    WIKIPEDIA_DOC_CONTENT_CHARS_MAX,
    ARXIV_TOP_K,
    ARXIV_DOC_CONTENT_CHARS_MAX,
)

from .llm import (
    get_llm,
    get_llm_with_tools,
    get_llm_with_structured_output,
    set_api_key,
    get_api_key,
)

__all__ = [
    # Configuration constants
    "GROQ_API_KEY",
    "LANGSMITH_API_KEY",
    "PROJECT_ROOT",
    "DATA_DIR",
    "PDF_FILE",
    "TEXT_FILE",
    "URLS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVER_K",
    "RECURSION_LIMIT",
    "WIKIPEDIA_TOP_K",
    "WIKIPEDIA_DOC_CONTENT_CHARS_MAX",
    "ARXIV_TOP_K",
    "ARXIV_DOC_CONTENT_CHARS_MAX",
    # LLM functions
    "get_llm",
    "get_llm_with_tools",
    "get_llm_with_structured_output",
]
