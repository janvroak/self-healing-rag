"""Self-Healing RAG: retrieval-augmented generation that checks its own answers."""

from shrag.pipeline import RAGResult, SelfHealingRAG

__all__ = ["RAGResult", "SelfHealingRAG"]
