# Agentic RAG with Multiple Tools

> **Multi-Source Retrieval Augmented Generation System using LangGraph**

An intelligent agent that retrieves information from multiple sources (URLs, PDFs, text files, Wikipedia, Arxiv, and web search) to answer questions accurately using LangGraph's agentic workflow architecture.

---

## 🎯 Project Overview

This project implements an **Agentic RAG (Retrieval Augmented Generation)** system that:

- Intelligently selects the right tool(s) for each query
- Grades document relevance before generating answers
- Rewrites queries when retrieved documents aren't relevant
- Uses 6 different knowledge sources

**Architecture Reference:** Based on `1-AgenticRAG_langraph.ipynb`

---

## 🏗️ Architecture

### Core Components

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       v
┌─────────────┐      ┌──────────────┐
│   AGENT     │─────▶│   RETRIEVE   │
│  (Decide)   │      │   (Tools)    │
└──────┬──────┘      └──────┬───────┘
       │                    │
       │                    v
       │             ┌──────────────┐
       │             │   GRADE      │
       │             │  DOCUMENTS   │
       │             └──────┬───────┘
       │                    │
       │         ┌──────────┴──────────┐
       │         │                     │
       │         v                     v
       │   ┌──────────┐         ┌──────────┐
       │   │ GENERATE │         │ REWRITE  │
       │   │ (Answer) │         │ (Query)  │
       │   └────┬─────┘         └────┬─────┘
       │        │                    │
       v        v                    │
    ┌──────────┐                    │
    │   END    │◀───────────────────┘
    └──────────┘
```

### Workflow Nodes

1. **AGENT Node**: Decides whether to use tools or respond directly
2. **RETRIEVE Node**: Executes selected tools to gather information
3. **GRADE DOCUMENTS Edge**: Assesses relevance of retrieved content
4. **GENERATE Node**: Creates answer from relevant documents
5. **REWRITE Node**: Reformulates query if documents aren't relevant

---

## 🛠️ Tools Integrated

| Tool | Purpose | Source |
|------|---------|--------|
| **URL Retriever** | LangGraph documentation | Web URLs |
| **PDF Retriever** | Agent Quality Whitepaper | Local PDF |
| **Text Retriever** | About Abhiram info | Local text file |
| **Wikipedia** | General knowledge | Wikipedia API |
| **Arxiv** | Research papers | Arxiv API |
| **DuckDuckGo** | Web search | DuckDuckGo API |

---

## 🔄 How It Works

### Step-by-Step Process

1. **User asks a question** → Agent receives query
2. **Agent decides** → Determines if tools are needed
3. **Tool selection** → Agent calls appropriate tool(s)
4. **Document retrieval** → Tools fetch relevant information
5. **Relevance grading** → LLM checks if docs match the question
   - ✅ **Relevant** → Generate answer
   - ❌ **Not relevant** → Rewrite query and retry
6. **Answer generation** → Creates concise response using retrieved context

### Example Query Flow

**Question:** "What is LangGraph?"

```
1. AGENT: Decides to use langgraph_docs_search tool
2. RETRIEVE: Fetches relevant documentation chunks
3. GRADE: Documents are relevant ✓
4. GENERATE: "LangGraph is a framework for building 
   stateful, multi-actor applications with LLMs..."
```

---

## 💻 Technology Stack

- **LangChain & LangGraph**: Agent orchestration
- **Groq LLM**: `llama-3.1-8b-instant` for fast inference
- **FAISS**: Vector similarity search
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Python 3.12**: Core language

---

## 📋 Setup Instructions

### 1. Clone Repository

```bash
cd Multi_source_tool_retriever_using_langgraph
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install langchainhub  # For RAG prompts
```

### 4. Set Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run Notebook

```bash
jupyter notebook agentic_rag_with_multiple_tools.ipynb
```

---

## 🎤 Interview Questions & Answers

### Q1: What is the main difference between traditional RAG and Agentic RAG?

**Answer:**
Traditional RAG blindly retrieves documents and generates answers, while **Agentic RAG**:

- Uses an **agent** to decide when to retrieve
- **Grades** document relevance before answering
- Can **rewrite** queries for better results
- Supports **multiple tools** and sources

This makes it more intelligent and context-aware.

---

### Q2: How does the agent decide which tool to use?

**Answer:**
The agent uses **LLM tool calling** with `bind_tools()`. The LLM:

1. Analyzes the user's question
2. Reviews tool descriptions
3. Selects the most appropriate tool(s)
4. Can call multiple tools if needed

Example: "Tell me about Abhiram" → selects `about_abhiram_search` tool

---

### Q3: What happens if retrieved documents aren't relevant?

**Answer:**
The system has a **feedback loop**:

1. **Grade Documents** edge checks relevance using structured LLM output
2. If score = "no":
   - Routes to **REWRITE node**
   - Reformulates the question for better semantic matching
   - Returns to **AGENT** to retry with improved query
3. If score = "yes":
   - Routes to **GENERATE node**
   - Creates answer from relevant context

This prevents hallucinations from irrelevant context.

---

### Q4: Why use LangGraph instead of simple LangChain chains?

**Answer:**
LangGraph provides:

- **State management**: Tracks conversation history with `add_messages`
- **Conditional routing**: Different paths based on relevance grading
- **Cycles**: Can loop back (rewrite → agent → retrieve)
- **Transparency**: Can visualize and debug the graph
- **Flexibility**: Easy to add new nodes/edges

Traditional chains are linear; LangGraph handles complex, branching workflows.

---

### Q5: How does document grading work technically?

**Answer:**
Uses **structured output** with Pydantic:

```python
class grade(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

llm_with_tool = llm.with_structured_output(grade)
result = llm_with_tool.invoke({"question": q, "context": docs})
```

This forces the LLM to return validated JSON, ensuring reliability.

---

### Q6: What's the role of the ToolNode?

**Answer:**
`ToolNode` is a pre-built LangGraph component that:

- Receives tool calls from the agent
- Executes the actual tool functions
- Returns results back to the graph
- Handles multiple tool calls in parallel if needed

```python
retrieve = ToolNode(tools)  # Wraps all 6 tools
workflow.add_node("retrieve", retrieve)
```

---

### Q7: How do you handle different embedding models for different sources?

**Answer:**
Currently uses **same embedding model** (`all-MiniLM-L6-v2`) for consistency across:

- URL documents
- PDF chunks
- Text file content

This ensures:

- Semantic similarity works across sources
- No dimension mismatch issues
- Faster retrieval with unified index

For production, could use different embeddings per source if needed.

---

### Q8: What's the purpose of `add_messages` in AgentState?

**Answer:**
`add_messages` is a **reducer function** that:

- **Appends** new messages instead of replacing
- Maintains full conversation history
- Enables multi-turn dialogues
- Allows nodes to see previous context

```python
messages: Annotated[list[BaseMessage], add_messages]
```

Without it, each node would only see the latest message.

---

### Q9: How would you add a new tool to this system?

**Answer:**
**3 Steps:**

1. **Create the tool** (e.g., SQL database retriever):

```python
sql_tool = create_retriever_tool(
    sql_retriever,
    "sql_search",
    "Search SQL database for data"
)
```

1. **Add to tools list**:

```python
tools = [url_tool, pdf_tool, ..., sql_tool]
```

1. **Done!** The agent automatically learns about it from the description.

No code changes needed in nodes/edges.

---

### Q10: What are potential bottlenecks in this architecture?

**Answer:**

1. **LLM API calls**: Multiple calls per query (agent → grade → generate)
   - *Solution*: Cache results, use faster models for grading

2. **Embedding generation**: Creating vectors for large documents
   - *Solution*: Batch processing, pre-compute and store embeddings

3. **Sequential grading**: Checks docs one at a time
   - *Solution*: Parallel grading, batch relevance checking

4. **Rewrite loop**: Could cycle indefinitely
   - *Solution*: Add max retry limit in recursion_limit

---

### Q11: How does this differ from AutoGPT or similar agents?

**Answer:**

| Feature | This System | AutoGPT |
|---------|-------------|---------|
| **Focus** | RAG + Multi-source retrieval | General task automation |
| **Tools** | Retrieval-specific | Web, code execution, file ops |
| **Control** | Explicit graph structure | Autonomous decision-making |
| **Loops** | Controlled (rewrite only) | Unlimited planning loops |
| **Output** | Always returns answer | May fail or loop forever |

This is **more constrained** but **more reliable** for Q&A tasks.

---

### Q12: How do you prevent the agent from hallucinating?

**Answer:**
**4 Mechanisms:**

1. **Grounded retrieval**: Only uses retrieved documents
2. **Relevance grading**: Verifies context before answering
3. **Explicit prompting**: "If you don't know, say you don't know"
4. **Rewrite fallback**: Improves query instead of guessing

The RAG prompt explicitly states:

```
"If you don't know the answer, just say that you don't know."
```

---

## 🚀 Running the System

### Quick Test

```python
result = ask_agent("What is LangGraph?")
# Output:
# 🔧 Tools Used: langgraph_docs_search
# 📊 Total Steps: 5
# Answer: LangGraph is a framework...
```

### Expected Behavior

- **LangGraph questions** → Uses URL retriever
- **Agent quality questions** → Uses PDF retriever
- **Personal questions** → Uses text retriever
- **General knowledge** → Uses Wikipedia/DuckDuckGo

---

## 📁 Project Structure

```
├── agentic_rag_with_multiple_tools.ipynb  # Main implementation
├── test_all_tools.ipynb                   # Individual tool tests
├── data/
│   ├── Agent Quality Whitepaper.pdf       # PDF source
│   └── about_me.txt                       # Text source
├── src/                                    # Modular code (optional)
│   ├── tools/                             # Tool implementations
│   ├── nodes/                             # Node functions
│   └── edges/                             # Edge functions
└── .env                                    # Environment variables
```

---

## 🎯 Key Takeaways

1. **Agentic RAG** adds intelligence to traditional RAG
2. **LangGraph** enables complex, stateful workflows
3. **Multi-source** retrieval improves answer quality
4. **Grading** prevents irrelevant context from polluting answers
5. **Rewriting** creates a self-improving feedback loop

---

## 📚 Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Vector Database](https://faiss.ai/)

---

## 👤 Author

**Abhiram**  
Multi-Source Agentic RAG System Implementation

---

## 📝 License

MIT License - Feel free to use for learning and interviews!
