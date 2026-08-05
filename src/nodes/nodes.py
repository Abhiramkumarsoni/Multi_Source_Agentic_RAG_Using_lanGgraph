"""
Node Functions - Agent, Generate, and Rewrite nodes
Following architecture from agentic_rag_with_multiple_tools.ipynb
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from configuration.llm import get_llm


# System prompt that guides the model to ALWAYS use tools first
AGENT_SYSTEM_PROMPT = """You are a helpful research assistant with access to multiple search tools.

IMPORTANT RULES:
1. ALWAYS use a search tool to retrieve information before answering any question.
2. Never answer from memory alone - always retrieve relevant documents first.
3. Choose the correct tool based on the question:
   - about_abhiram_search / personal_info_search: Questions about Abhiram Kumar Soni
   - langgraph_docs_search: Questions about LangGraph, LangChain, state graphs, agents
   - pdf_whitepaper_search: Questions about agent quality, benchmarks, evaluation
   - wikipedia: General knowledge, history, definitions
   - arxiv_search: Academic papers, scientific research
   - duckduckgo_search: Current events, recent news, general web queries
4. If unsure which tool to use, pick the most relevant one and search.
"""


def _get_content_str(content) -> str:
    """Safely extract a string from AIMessage content (handles list-of-blocks)."""
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return " ".join(p for p in parts if p).strip()
    return str(content) if content else ""


def agent(state, tools):
    """
    Invokes the agent model to generate a response based on the current state.
    Decides whether to retrieve using tools or simply end.

    Uses a system prompt to guide the LLM to always prefer tool use.
    If the model still returns empty content with no tool calls (known llama-3.1-8b
    behavior with many tools), it retries with an explicit nudge message.

    Args:
        state: The current state with messages
        tools: List of available tools

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = get_llm().bind_tools(tools)

    # Always prepend the system message to guide tool selection
    full_messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)
    response = model.invoke(full_messages)

    # Detect empty-response with no tool calls (known llama-3.1-8b issue)
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    content_str = _get_content_str(response.content)

    if not has_tool_calls and not content_str:
        print("---AGENT: empty response, retrying with explicit tool nudge---")
        # Add an explicit instruction to use a tool
        nudge_messages = full_messages + [
            HumanMessage(
                content="You must use one of the available search tools to look up "
                        "the answer. Please call the most relevant tool now."
            )
        ]
        response = model.invoke(nudge_messages)

        # If still empty, use plain LLM as last resort
        content_str2 = _get_content_str(response.content)
        if not getattr(response, "tool_calls", None) and not content_str2:
            print("---AGENT: still empty, falling back to plain LLM---")
            response = get_llm().invoke(messages)

    return {"messages": [response]}


def generate(state):
    """
    Generate answer using retrieved documents

    Args:
        state: The current state with messages

    Returns:
        dict: The updated message with generated answer
    """
    print("---GENERATE---")
    messages = state["messages"]

    # Get question from user messages
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    question = user_messages[-1].content if user_messages else messages[0].content

    # Get retrieved documents from last message (handle list or string content)
    last_message = messages[-1]
    docs = _get_content_str(last_message.content)

    # RAG Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:""",
        input_variables=["question", "context"],
    )

    # Chain
    rag_chain = prompt | get_llm() | StrOutputParser()

    # Generate
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [AIMessage(content=response)]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state: The current state with messages

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]

    # Get question
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    question = user_messages[-1].content if user_messages else messages[0].content

    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent / meaning.
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    response = get_llm().invoke(msg)
    return {"messages": [response]}
