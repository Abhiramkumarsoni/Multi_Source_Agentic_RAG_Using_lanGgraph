# Debugging Folder - Agentic RAG Agent

This folder contains the Agentic RAG agent for LangGraph Studio debugging.

## Structure

```
debugging/
├── agentic_rag_agent.py   # Main agent with graph definition
├── langgraph.json         # LangGraph configuration
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Workflow 

```
START → agent → retrieve → grade_documents → [rewrite] → generate → END
```

- **agent**: Extracts question from user message
- **retrieve**: Searches FAISS vector store with LangGraph docs
- **grade_documents**: Filters relevant documents
- **rewrite**: Rewrites query if no documents found (max 2 attempts)
- **generate**: Creates answer from retrieved documents

## Usage

### Run with LangGraph Dev/Studio

```bash
cd debugging
langgraph dev
```

This starts LangGraph Studio where you can visualize and test the workflow.

## LangGraph Configuration

`langgraph.json` tells the LangGraph CLI how to load this agent:

- **`dependencies`** lists the project paths whose dependencies should be installed. `"."` refers to this `debugging` folder.
- **`graphs`** maps the graph name shown in LangGraph Studio to the Python module and exported graph object. Here, `agentic_rag_agent` loads `graph` from `agentic_rag_agent.py`.
- **`env`** points to the environment-variable file, relative to this configuration file. `"../.env"` uses the `.env` file in the repository root.

### Test Queries

- "What is LangGraph?"
- "How to create map-reduce branches in LangGraph?"
- "Explain workflows and agents in LangGraph"

## Key Difference from Previous Version

This implementation:

1. Uses proper document retrieval from FAISS vector store
2. Only generates answers based on retrieved documents
3. Avoids hallucination by not using general LLM knowledge


example query

{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ]
}