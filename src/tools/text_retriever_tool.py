"""
Text Retriever Tool - About Abhiram search
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool


def create_text_retriever_tool(text_path: str):
    """
    Create a retriever tool from a text file
    
    Args:
        text_path: Path to the text file
        
    Returns:
        Retriever tool or None if failed
    """
    try:
        text_file = Path(text_path)
        
        if not text_file.exists():
            print(f"✗ Text file not found: {text_path}")
            return None
        
        loader = TextLoader(str(text_file))
        text_docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_splits = text_splitter.split_documents(text_docs)
        
        text_vectorstore = FAISS.from_documents(
            documents=text_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        text_retriever = text_vectorstore.as_retriever(search_kwargs={"k": 4})
        
        text_retriever_tool = create_retriever_tool(
            text_retriever,
            "about_abhiram_search",
            "Search information about Abhiram. Use for questions about Abhiram's background, experience, or profile."
        )
        
        print(f"✓ Text Retriever Tool created ({len(text_splits)} chunks)")
        return text_retriever_tool
        
    except Exception as e:
        print(f"✗ Failed to create text retriever: {e}")
        return None
