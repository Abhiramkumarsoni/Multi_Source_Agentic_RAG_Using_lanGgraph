"""
PDF Retriever Tool - Agent Quality Whitepaper search
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool


def create_pdf_retriever_tool(pdf_path: str):
    """
    Create a retriever tool from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Retriever tool or None if failed
    """
    try:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"✗ PDF not found: {pdf_path}")
            return None
        
        loader = PyPDFLoader(str(pdf_file))
        pdf_docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pdf_splits = text_splitter.split_documents(pdf_docs)
        
        pdf_vectorstore = FAISS.from_documents(
            documents=pdf_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 4})
        
        pdf_retriever_tool = create_retriever_tool(
            pdf_retriever,
            "pdf_search",
            "Search the Agent Quality Whitepaper PDF. Use for questions about agent quality."
        )
        
        print(f"✓ PDF Retriever Tool created ({len(pdf_splits)} chunks)")
        return pdf_retriever_tool
        
    except Exception as e:
        print(f"✗ Failed to create PDF retriever: {e}")
        return None
