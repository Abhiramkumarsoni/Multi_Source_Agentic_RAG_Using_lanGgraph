"""
Streamlit UI - Simple interface for the Agentic RAG Agent
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import create_agent

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Agentic RAG Agent")
st.caption("Multi-Source Tool Retriever using LangGraph")

# Sidebar
with st.sidebar:
    st.header("📚 Available Tools")
    st.markdown("""
    - **LangGraph Docs** - Documentation search
    - **PDF Search** - Whitepaper search
    - **Wikipedia** - General knowledge
    - **Arxiv** - Academic papers
    - **DuckDuckGo** - Web search
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize agent
@st.cache_resource
def get_agent():
    """Initialize the agent (cached)"""
    return create_agent()

try:
    agent = get_agent()
    st.sidebar.success(f"✅ {agent.get_tool_count()} tools ready")
except Exception as e:
    st.error(f"Failed to initialize agent: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Use query_with_details to get tools used
                response, details = agent.query_with_details(prompt)
                
                # Show tools used if any
                if details.get("tools_used"):
                    tools_str = ", ".join(details["tools_used"])
                    st.caption(f"🔧 Tools Used: {tools_str}")
                    st.caption(f"📊 Total Steps: {details.get('total_messages', 0)}")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

