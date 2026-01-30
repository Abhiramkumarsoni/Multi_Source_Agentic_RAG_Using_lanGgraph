"""
Agentic RAG Agent - Following 1-AgenticRAG_langraph.ipynb pattern exactly
Simple agent with tools: Wikipedia, Arxiv, PDF, Text, URL retrievers
"""
import os
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_FILE = DATA_DIR / "Agent Quality Whitepaper.pdf"
TEXT_FILE = DATA_DIR / "about_me.txt"

# URLs
URLS = [
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]


# ==================== State ====================
class State(TypedDict):
    """State with messages key for LangGraph Studio chat interface"""
    messages: Annotated[list, add_messages]


# ==================== Create Tools ====================

# 1. URL/Web Retriever Tool
def create_url_retriever_tool():
    """Create retriever tool from web URLs"""
    try:
        docs_list = []
        for url in URLS:
            try:
                docs = WebBaseLoader(url).load()
                docs_list.extend(docs)
            except Exception as e:
                print(f"[WARN] Failed to load {url}: {e}")
        
        if not docs_list:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)
        
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        retriever = vectorstore.as_retriever()
        
        return create_retriever_tool(
            retriever,
            "langgraph_docs_search",
            "Search for information about LangGraph. Use this for questions about LangGraph concepts, tutorials, and features."
        )
    except Exception as e:
        print(f"[WARN] URL retriever failed: {e}")
        return None


# 2. PDF Retriever Tool
def create_pdf_retriever_tool():
    """Create retriever tool from PDF file"""
    if not PDF_FILE.exists():
        print(f"[INFO] PDF not found: {PDF_FILE}")
        return None
    try:
        loader = PyPDFLoader(str(PDF_FILE))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        retriever = vectorstore.as_retriever()
        
        return create_retriever_tool(
            retriever,
            "pdf_search",
            "Search the Agent Quality Whitepaper PDF. Use for questions about agent quality."
        )
    except Exception as e:
        print(f"[WARN] PDF retriever failed: {e}")
        return None


# 3. Text File Retriever Tool
def create_text_retriever_tool():
    """Create retriever tool from text file"""
    if not TEXT_FILE.exists():
        print(f"[INFO] Text file not found: {TEXT_FILE}")
        return None
    try:
        loader = TextLoader(str(TEXT_FILE))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        retriever = vectorstore.as_retriever()
        
        return create_retriever_tool(
            retriever,
            "about_me_search",
            "Search personal information about Abhiram. Use for questions about Abhiram."
        )
    except Exception as e:
        print(f"[WARN] Text retriever failed: {e}")
        return None


# 4. Wikipedia Tool
def create_wikipedia_tool():
    """Create Wikipedia search tool"""
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        return WikipediaQueryRun(api_wrapper=api_wrapper)
    except Exception as e:
        print(f"[WARN] Wikipedia failed: {e}")
        return None


# 5. Arxiv Tool
def create_arxiv_tool():
    """Create Arxiv search tool"""
    try:
        api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        return ArxivQueryRun(api_wrapper=api_wrapper)
    except Exception as e:
        print(f"[WARN] Arxiv failed: {e}")
        return None


# Initialize all tools - filter out None values
print("[INFO] Initializing tools...")
_tools_raw = [
    create_url_retriever_tool(),
    create_pdf_retriever_tool(),
    create_text_retriever_tool(),
    create_wikipedia_tool(),
    create_arxiv_tool()
]
tools = [t for t in _tools_raw if t is not None]
print(f"[INFO] {len(tools)} tools initialized successfully")

if not tools:
    raise RuntimeError("No tools were initialized! Check your data files and network connection.")


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state.get("messages", [])
    
    # Handle empty messages gracefully
    if not messages:
        msg = "No input provided. Please ask a question about LangGraph, Agent Quality, or Abhiram."
        return {"messages": [AIMessage(content=msg)]}

    model = ChatGroq(model="llama-3.1-8b-instant")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatGroq(model="llama-3.1-8b-instant")

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    # Get the last human message (question)
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if user_messages:
        question = user_messages[-1].content
    else:
        # Fallback to first message if no HumanMessage found (shouldn't happen in normal flow)
        question = messages[0].content

    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated message
    """
    print("---GENERATE---")
    messages = state["messages"]
    
    # Get the last human message (question)
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if user_messages:
         question = user_messages[-1].content
    else:
         question = messages[0].content

    last_message = messages[-1]

    docs = last_message.content

    # Prompt - RAG prompt template (replaces hub.pull("rlm/rag-prompt"))
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:""",
        input_variables=["question", "context"],
    )

    # LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    
    # Get the last human message (question)
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if user_messages:
         question = user_messages[-1].content
    else:
         question = messages[0].content


    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatGroq(model="llama-3.1-8b-instant")
    response = model.invoke(msg)
    return {"messages": [response]}



from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Define a new graph
workflow = StateGraph(State)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
"agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

