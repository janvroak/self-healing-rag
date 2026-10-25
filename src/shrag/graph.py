"""The self-correcting RAG workflow as a cyclical LangGraph state machine.

    START -> retrieve -> grade_documents --(relevant)--> generate -> critique --(accept)--> finalize -> END
                ^             |                             ^  |        |
                |       (none relevant)                     |  |  (hallucinated) -> generate again
                |             v                             |  |        |
                +------- rewrite_query <--------------------+--+---(incomplete / out of retries)
                              |
                      (retry limit hit) -> fallback -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from shrag.nodes import RAGNodes
from shrag.state import RAGState


def build_graph(nodes: RAGNodes) -> CompiledStateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("critique", nodes.critique)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("fallback", nodes.fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", nodes.route_after_grading, ["generate", "rewrite_query", "fallback"]
    )
    graph.add_conditional_edges(
        "generate", nodes.route_after_generation, ["critique", "rewrite_query", "fallback"]
    )
    graph.add_conditional_edges(
        "critique", nodes.route_after_critique,
        ["finalize", "generate", "rewrite_query", "fallback"],
    )
    graph.add_edge("rewrite_query", "retrieve")  # the cycle
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)
    return graph.compile()
