"""FastAPI interface for the multi-source LangGraph agent."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from configuration.llm import get_api_key

if TYPE_CHECKING:
    from src.agent import AgenticRAGAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Source RAG Agent API",
    description="Query the LangGraph agent over HTTP.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    include_details: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class QueryDetails(BaseModel):
    tools_used: list[str]
    total_messages: int


class QueryResponse(BaseModel):
    answer: str
    details: QueryDetails | None = None


@lru_cache(maxsize=1)
def _get_agent() -> "AgenticRAGAgent":
    from src.agent import create_agent

    return create_agent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not get_api_key():
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    try:
        agent = _get_agent()
        answer, details = agent.query_with_details(request.question)
    except Exception as exc:
        logger.exception("Agent query failed")
        raise HTTPException(
            status_code=500,
            detail="The agent could not process the question.",
        ) from exc

    return QueryResponse(
        answer=answer,
        details=QueryDetails(**details) if request.include_details else None,
    )
