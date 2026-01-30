"""
Configuration - Environment variables and paths
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== API Configuration ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# ==================== LangSmith Tracing ====================
# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "multi-source-rag-agent")

# ==================== Path Configuration ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_FILE = DATA_DIR / "Agent Quality Whitepaper.pdf"
TEXT_FILE = DATA_DIR / "about_me.txt"

# ==================== URL Configuration ====================
# URLs for web retrieval
URLS = [
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]

# ==================== LLM Configuration ====================
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0

# ==================== Embedding Configuration ====================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== Text Splitter Configuration ====================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ==================== Retrieval Configuration ====================
# Number of documents to retrieve
RETRIEVER_K = 4

# ==================== Graph Configuration ====================
# Maximum recursion limit for graph execution
RECURSION_LIMIT = 25

# ==================== Tool Configuration ====================
# Wikipedia settings
WIKIPEDIA_TOP_K = 1
WIKIPEDIA_DOC_CONTENT_CHARS_MAX = 500

# Arxiv settings
ARXIV_TOP_K = 1
ARXIV_DOC_CONTENT_CHARS_MAX = 500
