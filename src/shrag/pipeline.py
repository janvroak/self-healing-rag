"""High-level entry point: build everything from settings and ask questions."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.runnables import Runnable

from shrag import prompts
from shrag.config import Settings, load_settings
from shrag.graph import build_graph
from shrag.llm import build_llm, text_of
from shrag.nodes import LoopLimits, RAGNodes
from shrag.state import RAGState, TraceEvent
from shrag.vectorstore import Retriever, VectorRetriever, open_store


@dataclass
class RAGResult:
    question: str
    answer: str
    status: str  # answered | partial | insufficient
    sources: list[str]
    retrieval_attempts: int
    llm_calls: int
    seconds: float
    trace: list[TraceEvent] = field(default_factory=list)


class SelfHealingRAG:
    def __init__(self, retriever: Retriever, generator: Runnable, critic: Runnable,
                 limits: LoopLimits = LoopLimits()):
        self.retriever = retriever
        self.generator = generator
        self.limits = limits
        self.graph = build_graph(RAGNodes(retriever, generator, critic, limits))

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **limit_overrides) -> "SelfHealingRAG":
        settings = settings or load_settings()
        retriever = VectorRetriever(open_store(settings))
        if retriever.is_empty():
            raise RuntimeError("The knowledge base is empty. Run `shrag ingest` first.")
        limits = LoopLimits(
            max_retrievals=limit_overrides.get("max_retrievals", settings.max_retrievals),
            max_generations=limit_overrides.get("max_generations", settings.max_generations),
            base_k=settings.base_k,
            k_step=settings.k_step,
        )
        return cls(retriever, build_llm(settings.generator_models, settings),
                   build_llm(settings.critic_models, settings), limits)

    def _config(self) -> dict:
        return {"recursion_limit": self.limits.recursion_limit}

    def ask(self, question: str) -> RAGResult:
        start = time.perf_counter()
        state: RAGState = self.graph.invoke({"question": question}, self._config())
        return self._result(question, state, time.perf_counter() - start)

    def stream(self, question: str) -> Iterator[tuple[str, dict]]:
        """Yield (node name, state update) as each step finishes, then ("result", RAGResult)."""
        start = time.perf_counter()
        final: dict = {"question": question}
        for chunk in self.graph.stream({"question": question}, self._config(), stream_mode="updates"):
            for node, update in chunk.items():
                for key, value in (update or {}).items():
                    if key in ("trace", "queries_tried"):
                        final[key] = final.get(key, []) + value
                    elif key == "llm_calls":
                        final[key] = final.get(key, 0) + value
                    else:
                        final[key] = value
                yield node, update or {}
        yield "result", self._result(question, final, time.perf_counter() - start)

    def baseline(self, question: str) -> RAGResult:
        """Plain retrieve-then-generate RAG with no critic, for comparison."""
        start = time.perf_counter()
        docs = self.retriever.search(question, self.limits.base_k)
        answer = text_of(self.generator.invoke(prompts.naive_rag(question, docs)))
        return RAGResult(question, answer, "answered",
                         [d.metadata.get("chunk_id", "") for d in docs], 1, 1,
                         time.perf_counter() - start)

    @staticmethod
    def _result(question: str, state: dict, seconds: float) -> RAGResult:
        return RAGResult(
            question=question,
            answer=state.get("final_answer", ""),
            status=state.get("status", "insufficient"),
            sources=state.get("sources", []),
            retrieval_attempts=state.get("retrieval_attempts", 0),
            llm_calls=state.get("llm_calls", 0),
            seconds=round(seconds, 1),
            trace=state.get("trace", []),
        )
