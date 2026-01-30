"""
Streamlit Final App - Complete UI with API Key Input, Agent Mode Toggle, and Streaming
Supports both Agentic RAG and Router Agent modes with streaming responses
"""
import streamlit as st
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration.llm import set_api_key, get_api_key

# Page configuration
st.set_page_config(
    page_title="Multi-Source RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .mode-agentic { background: #4A90E2; color: white; }
    .mode-router { background: #50C878; color: white; }
    .tool-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        background: #2d3748;
        color: #e2e8f0;
        margin: 0.1rem;
    }
    .api-status-ok { color: #50C878; font-weight: bold; }
    .api-status-missing { color: #E94B3C; font-weight: bold; }
    .streaming-cursor {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 Multi-Source RAG Agent</p>', unsafe_allow_html=True)
st.caption("Intelligent Tool Retriever with LangGraph")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "Enter GROQ API Key",
        type="password",
        placeholder="gsk_...",
        help="Your GROQ API key for LLM access"
    )
    
    if api_key_input:
        set_api_key(api_key_input)
        st.markdown('<span class="api-status-ok">✅ API Key Set</span>', unsafe_allow_html=True)
    elif get_api_key():
        st.markdown('<span class="api-status-ok">✅ Using .env Key</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="api-status-missing">❌ No API Key</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Agent Mode Toggle
    st.subheader("🔄 Agent Mode")
    agent_mode = st.toggle(
        "Use Router Agent",
        value=False,
        help="Switch between Agentic RAG (multi-step reasoning) and Router Agent (semantic routing)"
    )
    
    if agent_mode:
        st.markdown('<span class="mode-badge mode-router">🧭 Router Mode</span>', unsafe_allow_html=True)
        st.caption("Fast semantic routing to best data source")
    else:
        st.markdown('<span class="mode-badge mode-agentic">🤖 Agentic RAG Mode</span>', unsafe_allow_html=True)
        st.caption("Multi-step reasoning with document grading")
    
    st.divider()
    
    # Tool Info
    st.subheader("📚 Available Tools")
    tools_list = [
        ("🔷", "LangGraph Docs"),
        ("📄", "PDF Search"),
        ("👤", "About Abhiram"),
        ("🌐", "Wikipedia"),
        ("🎓", "Arxiv"),
        ("🔍", "DuckDuckGo")
    ]
    for icon, name in tools_list:
        st.markdown(f"{icon} **{name}**")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if "agent" in st.session_state:
            del st.session_state["agent"]
        if "router_agent" in st.session_state:
            del st.session_state["router_agent"]
        st.rerun()

# Check API key before proceeding
if not get_api_key():
    st.warning("⚠️ Please enter your GROQ API Key in the sidebar to continue.")
    st.stop()

# Initialize agents based on mode
@st.cache_resource
def get_agentic_rag_agent():
    """Initialize the Agentic RAG agent (cached)"""
    from src.agent import create_agent
    return create_agent()

@st.cache_resource
def get_router_agent():
    """Initialize the Router agent (cached)"""
    from router_agent.router_agent import RouterAgent
    return RouterAgent()

# Load appropriate agent
try:
    if agent_mode:
        agent = get_router_agent()
        st.sidebar.success(f"✅ Router Agent Ready")
    else:
        agent = get_agentic_rag_agent()
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
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if "tools_used" in meta and meta["tools_used"]:
                tools_html = " ".join([f'<span class="tool-badge">{t}</span>' for t in meta["tools_used"]])
                st.markdown(tools_html, unsafe_allow_html=True)
            if "route" in meta:
                st.caption(f"📍 Route: {meta['route']}")


def stream_response(text: str, delay: float = 0.02):
    """Generator to stream text character by character"""
    for char in text:
        yield char
        time.sleep(delay)


def stream_response_words(text: str, delay: float = 0.05):
    """Generator to stream text word by word"""
    words = text.split(' ')
    for i, word in enumerate(words):
        if i > 0:
            yield ' '
        yield word
        time.sleep(delay)


# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        mode_name = "Router" if agent_mode else "Agentic RAG"
        
        # Show processing status
        status_placeholder = st.empty()
        status_placeholder.markdown(f"🔄 *{mode_name} is processing...*")
        
        try:
            if agent_mode:
                # Router Agent Mode
                status_placeholder.markdown("🧭 *Routing query to best source...*")
                result = agent.query(prompt)
                response = result["answer"]
                metadata = {
                    "route": result["route"],
                    "tools_used": [result["route"]]
                }
                
                # Clear status and show route
                status_placeholder.empty()
                st.caption(f"📍 Route: {result['route']}")
                
                # Stream the response
                response_placeholder = st.empty()
                streamed_text = ""
                for word in stream_response_words(response):
                    streamed_text += word
                    response_placeholder.markdown(streamed_text + "▌")
                response_placeholder.markdown(streamed_text)
                
                # Show context
                with st.expander("📄 Retrieved Context"):
                    st.text(result.get("context", ""))
                    
            else:
                # Agentic RAG Mode
                status_placeholder.markdown("🤖 *Agent is reasoning...*")
                response, details = agent.query_with_details(prompt)
                metadata = {
                    "tools_used": details.get("tools_used", []),
                    "total_messages": details.get("total_messages", 0)
                }
                
                # Clear status and show tools
                status_placeholder.empty()
                if details.get("tools_used"):
                    tools_html = " ".join([f'<span class="tool-badge">{t}</span>' for t in details["tools_used"]])
                    st.markdown(tools_html, unsafe_allow_html=True)
                    st.caption(f"📊 Steps: {details.get('total_messages', 0)}")
                
                # Stream the response
                response_placeholder = st.empty()
                streamed_text = ""
                for word in stream_response_words(response):
                    streamed_text += word
                    response_placeholder.markdown(streamed_text + "▌")
                response_placeholder.markdown(streamed_text)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata
            })
            
        except Exception as e:
            status_placeholder.empty()
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Agentic RAG - Multi-step reasoning")
with col2:
    st.caption("🧭 Router - Fast semantic routing")
with col3:
    current_mode = "Router" if agent_mode else "Agentic RAG"
    st.caption(f"Current: **{current_mode}**")
