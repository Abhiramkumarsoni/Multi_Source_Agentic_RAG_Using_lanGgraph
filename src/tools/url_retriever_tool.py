"""
URL Retriever Tool - LangGraph documentation search
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool


def create_url_retriever_tool(urls: list):
    """
    Create a retriever tool from URLs
    
    Args:
        urls: List of URLs to load
        
    Returns:
        Retriever tool or None if failed
    """
    try:
        print("Loading URLs...")
        docs_list = []
        for url in urls:
            try:
                docs = WebBaseLoader(url).load()
                docs_list.extend(docs)
                print(f"✓ Loaded: {url}")
            except Exception as e:
                print(f"✗ Failed: {url} - {e}")
        
        if not docs_list:
            print("No documents loaded from URLs")
            return None
        
        # Split and vectorize
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        url_retriever_tool = create_retriever_tool(
            retriever,
            "langgraph_docs_search",
            "Search for information about LangGraph. Use this for questions about LangGraph concepts, tutorials, and features."
        )
        
        print(f"✓ URL Retriever Tool created ({len(doc_splits)} chunks)")
        return url_retriever_tool
        
    except Exception as e:
        print(f"✗ Failed to create URL retriever: {e}")
        return None
