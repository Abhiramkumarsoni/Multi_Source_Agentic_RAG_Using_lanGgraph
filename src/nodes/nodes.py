"""
Node Functions - Agent, Generate, and Rewrite nodes
Following architecture from agentic_rag_with_multiple_tools.ipynb
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from configuration.llm import get_llm


def agent(state, tools):
    """
    Invokes the agent model to generate a response based on the current state.
    Decides whether to retrieve using tools or simply end.

    Args:
        state: The current state with messages
        tools: List of available tools

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = get_llm().bind_tools(tools)
    response = model.invoke(messages)
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

    # Get retrieved documents from last message
    last_message = messages[-1]
    docs = last_message.content

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
