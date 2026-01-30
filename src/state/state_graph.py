"""
State Definition - Graph state with messages
Following architecture from agentic_rag_with_multiple_tools.ipynb
"""
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent state with messages"""
    messages: Annotated[list[BaseMessage], add_messages]
