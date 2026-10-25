"""Graph nodes and routing functions.

Each node takes the current state and returns the fields it changes. Routing
functions read the state and pick the next node; together with the retry
counters they guarantee the loop always ends.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from shrag import prompts
from shrag.llm import text_of
from shrag.state import RAGState, TraceEvent
from shrag.structured import Critique, RelevanceGrades, StructuredOutputError, invoke_structured
from shrag.vectorstore import Retriever

MAX_CONTEXT_CHUNKS = 8
_CITATION = re.compile(r"\[(\d+)\]")

FALLBACK_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that confidently."
)


@dataclass(frozen=True)
class LoopLimits:
    max_retrievals: int = 3  # retrieval rounds, including the first
    max_generations: int = 2  # answer attempts per retrieval round
    base_k: int = 4  # chunks fetched on the first retrieval
    k_step: int = 2  # extra chunks fetched on each retry (widens the net)

    @property
    def recursion_limit(self) -> int:
        # Upper bound on node executions; LangGraph aborts past it as a backstop.
        per_round = 3 + 2 * self.max_generations  # retrieve, grade, (generate, critique)*n, rewrite
        return self.max_retrievals * per_round + 5


def _event(step: str, summary: str, **details) -> list[TraceEvent]:
    return [{"step": step, "summary": summary, "details": details}]


def _chunk_id(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.page_content[:80]


def citation_problems(answer: str, n_chunks: int) -> list[int]:
    """Citation numbers in the answer that point at chunks that don't exist."""
    return sorted({int(n) for n in _CITATION.findall(answer) if not 1 <= int(n) <= n_chunks})


class RAGNodes:
    def __init__(self, retriever: Retriever, generator: Runnable, critic: Runnable,
                 limits: LoopLimits = LoopLimits()):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic
        self.limits = limits

    # ----- nodes -----------------------------------------------------------

    def retrieve(self, state: RAGState) -> dict:
        attempt = state.get("retrieval_attempts", 0) + 1
        query = state.get("query") or state["question"]
        k = self.limits.base_k + self.limits.k_step * (attempt - 1)
        docs = self.retriever.search(query, k)
        return {
            "query": query,
            "queries_tried": [query],
            "documents": docs,
            "retrieval_attempts": attempt,
            "generation_attempts": 0,
            "feedback": "",
            "trace": _event(
                "retrieve", f"Round {attempt}: fetched {len(docs)} chunks for {query!r}",
                query=query, k=k,
                hits=[{"id": _chunk_id(d), "score": d.metadata.get("score")} for d in docs],
            ),
        }

    def grade_documents(self, state: RAGState) -> dict:
        docs = state.get("documents", [])
        kept: list[Document] = []
        calls = 0
        if docs:
            try:
                grades, calls = invoke_structured(
                    self.generator, prompts.grade_documents(state["question"], docs), RelevanceGrades
                )
                kept = [docs[i - 1] for i in dict.fromkeys(grades.relevant) if 1 <= i <= len(docs)]
            except StructuredOutputError:
                kept = docs  # can't grade: keep everything, the critic is still downstream
                calls = 2

        # Keep relevant chunks from earlier rounds too, so a multi-part question
        # can be answered from pieces found by different queries.
        merged = {_chunk_id(d): d for d in state.get("relevant_docs", [])}
        for d in kept:
            merged.setdefault(_chunk_id(d), d)
        relevant = list(merged.values())[-MAX_CONTEXT_CHUNKS:]
        return {
            "relevant_docs": relevant,
            "llm_calls": calls,
            "trace": _event(
                "grade_documents",
                f"{len(kept)}/{len(docs)} new chunks relevant; {len(relevant)} in context",
                kept=[_chunk_id(d) for d in kept],
                dropped=[_chunk_id(d) for d in docs if d not in kept],
            ),
        }

    def generate(self, state: RAGState) -> dict:
        docs = state["relevant_docs"]
        feedback = state.get("feedback", "")
        answer = text_of(self.generator.invoke(prompts.generate(state["question"], docs, feedback)))
        insufficient = prompts.INSUFFICIENT in answer and len(answer) < len(prompts.INSUFFICIENT) + 40
        attempt = state.get("generation_attempts", 0) + 1
        summary = ("Context doesn't answer the question" if insufficient
                   else f"Drafted answer (attempt {attempt}{', using critic feedback' if feedback else ''})")
        return {
            "answer": "" if insufficient else answer,
            "insufficient": insufficient,
            "generation_attempts": attempt,
            "llm_calls": 1,
            "trace": _event("generate", summary, answer=answer, used_feedback=bool(feedback)),
        }

    def critique(self, state: RAGState) -> dict:
        docs, answer = state["relevant_docs"], state["answer"]
        bad_citations = citation_problems(answer, len(docs))
        try:
            result, calls = invoke_structured(
                self.critic, prompts.critique(state["question"], docs, answer), Critique
            )
        except StructuredOutputError:
            # An answer we can't verify is treated as rejected, never accepted.
            result, calls = Critique(
                grounded=False, answers_question=False,
                feedback="The answer could not be verified. Stick strictly to the context.",
            ), 2

        unsupported = [c.claim for c in result.claims if not c.supported]
        grounded = result.grounded and not unsupported and not bad_citations
        feedback = result.feedback
        if unsupported:
            feedback = "Unsupported claims: " + "; ".join(unsupported) + ". " + feedback
        if bad_citations:
            feedback = f"Citations {bad_citations} refer to chunks that don't exist. " + feedback

        if grounded and result.answers_question:
            verdict, summary = "accept", "Answer is grounded and complete"
        elif grounded:
            verdict = "re_retrieve"
            summary = f"Grounded but incomplete: {result.missing or 'missing information'}"
        else:
            verdict = "regenerate"
            summary = f"Rejected: {len(unsupported)} unsupported claim(s)" + (
                f", bad citations {bad_citations}" if bad_citations else "")

        update = {
            "critique": result.model_dump(),
            "verdict": verdict,
            "feedback": feedback.strip() or result.missing,
            "llm_calls": calls,
            "trace": _event("critique", summary, verdict=verdict, unsupported=unsupported,
                            missing=result.missing, bad_citations=bad_citations),
        }
        if verdict == "re_retrieve":
            update["partial_answer"] = answer
        return update

    def rewrite_query(self, state: RAGState) -> dict:
        if not state.get("relevant_docs"):
            reason = "The search returned no relevant documents."
        elif state.get("insufficient"):
            reason = "The retrieved documents did not contain the answer."
        else:
            reason = state.get("feedback") or "The answer was not supported by the documents."
        tried = state.get("queries_tried", [])
        reply = text_of(self.generator.invoke(prompts.rewrite_query(state["question"], tried, reason)))
        lines = [line.strip().strip('"') for line in reply.splitlines() if line.strip()]
        new_query = lines[0] if lines else ""
        if not new_query or new_query in tried:
            new_query = f"{state['question']} {state.get('critique', {}).get('missing', '')}".strip()
        return {
            "query": new_query,
            "llm_calls": 1,
            "trace": _event("rewrite_query", f"New query: {new_query!r}", reason=reason),
        }

    def finalize(self, state: RAGState) -> dict:
        return {
            "status": "answered",
            "final_answer": state["answer"],
            "sources": [_chunk_id(d) for d in state["relevant_docs"]],
            "trace": _event("finalize", "Returned verified answer"),
        }

    def fallback(self, state: RAGState) -> dict:
        partial = state.get("partial_answer", "")
        tried = ", ".join(repr(q) for q in state.get("queries_tried", []))
        if partial:
            missing = state.get("critique", {}).get("missing", "")
            return {
                "status": "partial",
                "final_answer": f"{partial}\n\nNote: this answer is incomplete. "
                                f"{'Not found in the knowledge base: ' + missing if missing else ''}".strip(),
                "sources": [_chunk_id(d) for d in state.get("relevant_docs", [])],
                "trace": _event("fallback", "Retry limit reached; returned partial grounded answer"),
            }
        return {
            "status": "insufficient",
            "final_answer": f"{FALLBACK_MESSAGE} I searched for: {tried}.",
            "sources": [],
            "trace": _event("fallback", "Retry limit reached; declined to answer"),
        }

    # ----- routing ---------------------------------------------------------

    def _can_retrieve_again(self, state: RAGState) -> bool:
        return state.get("retrieval_attempts", 0) < self.limits.max_retrievals

    def route_after_grading(self, state: RAGState) -> str:
        if state.get("relevant_docs"):
            return "generate"
        return "rewrite_query" if self._can_retrieve_again(state) else "fallback"

    def route_after_generation(self, state: RAGState) -> str:
        if not state.get("insufficient"):
            return "critique"
        return "rewrite_query" if self._can_retrieve_again(state) else "fallback"

    def route_after_critique(self, state: RAGState) -> str:
        verdict = state.get("verdict")
        if verdict == "accept":
            return "finalize"
        if verdict == "regenerate" and state.get("generation_attempts", 0) < self.limits.max_generations:
            return "generate"
        return "rewrite_query" if self._can_retrieve_again(state) else "fallback"
