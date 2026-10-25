"""The state that flows around the graph."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document

Status = Literal["answered", "partial", "insufficient"]


class TraceEvent(TypedDict):
    step: str
    summary: str
    details: dict


class RAGState(TypedDict, total=False):
    question: str
    query: str  # current search query (starts as the question, then gets rewritten)
    queries_tried: Annotated[list[str], operator.add]

    documents: list[Document]  # raw hits from the latest retrieval
    relevant_docs: list[Document]  # graded-relevant chunks, kept across retries

    answer: str
    insufficient: bool  # generator said the context doesn't answer the question
    critique: dict
    verdict: str  # accept | regenerate | re_retrieve
    feedback: str  # what the critic wants fixed
    partial_answer: str  # best grounded-but-incomplete answer so far

    retrieval_attempts: int
    generation_attempts: int  # resets after every retrieval
    llm_calls: Annotated[int, operator.add]

    status: Status
    final_answer: str
    sources: list[str]
    trace: Annotated[list[TraceEvent], operator.add]
