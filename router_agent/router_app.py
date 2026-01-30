"""
Streamlit UI for Router-Based Agent
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router_agent import RouterAgent

# Page configuration
st.set_page_config(
    page_title="Router-Based RAG Agent",
    page_icon="🧭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .route-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .route-langgraph { background: #4A90E2; color: white; }
    .route-pdf { background: #E94B3C; color: white; }
    .route-personal { background: #50C878; color: white; }
    .route-wikipedia { background: #9B59B6; color: white; }
    .route-arxiv { background: #F39C12; color: white; }
    .route-web { background: #34495E; color: white; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧭 Router-Based RAG Agent")
st.caption("Intelligent semantic routing to the right data source")

# Sidebar
with st.sidebar:
    st.header("📚 Available Routes")
    
    routes_info = {
        "🔷 LangGraph Docs": "LangGraph, workflows, state graphs",
        "📄 PDF Whitepaper": "Agent quality, evaluation",
        "👤 Personal Info": "About Abhiram Kumar Soni",
        "🌐 Wikipedia": "General knowledge, facts",
        "🎓 Arxiv": "Academic papers, research",
        "🔍 Web Search": "Current events, web queries"
    }
    
    for route, desc in routes_info.items():
        st.markdown(f"**{route}**")
        st.caption(desc)
        st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize agent
@st.cache_resource
def get_agent():
    """Initialize the router agent (cached)"""
    return RouterAgent()

try:
    agent = get_agent()
    st.sidebar.success(f"✅ {len(agent.get_available_routes())} routes ready")
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
        
        # Show route badge if available
        if message["role"] == "assistant" and "route" in message:
            route_class = message["route"].replace("_", "-")
            st.markdown(
                f'<span class="route-badge route-{route_class}">📍 {message["route"]}</span>',
                unsafe_allow_html=True
            )

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🧭 Routing your query..."):
            try:
                result = agent.query(prompt)
                
                # Display answer
                st.markdown(result["answer"])
                
                # Display route badge
                route_class = result["route"].replace("_", "-")
                st.markdown(
                    f'<span class="route-badge route-{route_class}">📍 {result["route"]}</span>',
                    unsafe_allow_html=True
                )
                
                # Show context in expander
                with st.expander("📄 Retrieved Context"):
                    st.text(result["context"])
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "route": result["route"]
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
