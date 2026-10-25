"""Prompts for each LLM step. Every step returns a list of (role, content) messages."""

from langchain_core.documents import Document

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('chunk_id', d.metadata.get('source', '?'))})\n"
        f"{d.page_content.strip()}"
        for i, d in enumerate(docs, start=1)
    )


def grade_documents(question: str, docs: list[Document]) -> list[tuple[str, str]]:
    return [
        ("system",
         "You filter search results for a question-answering system. A chunk is relevant "
         "if it contains information that helps answer the question, even partially. "
         'Reply with only JSON: {"relevant": [chunk numbers]}. Use [] if none are relevant.'),
        ("user", f"Question: {question}\n\nChunks:\n{format_context(docs)}"),
    ]


def generate(question: str, docs: list[Document], feedback: str = "") -> list[tuple[str, str]]:
    system = (
        "You answer questions using ONLY the numbered context chunks provided. Rules:\n"
        "1. Every factual sentence must end with a citation of the chunk(s) it comes from, like [2] or [1][3].\n"
        "2. Never use outside knowledge, and never guess numbers, names or dates that are not in the context.\n"
        "3. If the context answers only part of the question, answer that part and state clearly "
        "what the context does not cover.\n"
        f"4. If the context contains nothing that answers the question, reply with exactly: {INSUFFICIENT}\n"
        "Be concise."
    )
    user = f"Context:\n{format_context(docs)}\n\nQuestion: {question}"
    if feedback:
        user += (
            "\n\nA reviewer rejected your previous answer for this reason:\n"
            f"{feedback}\nWrite a corrected answer that fixes this. Remove any claim the context "
            "does not support."
        )
    return [("system", system), ("user", user)]


def critique(question: str, docs: list[Document], answer: str) -> list[tuple[str, str]]:
    return [
        ("system",
         "You are a strict fact-checker for a retrieval-augmented QA system. You decide whether "
         "an answer is grounded in the provided context chunks or contains hallucinations.\n\n"
         "Steps:\n"
         "1. Split the answer into its individual factual claims.\n"
         "2. For each claim, mark supported=true ONLY if the context explicitly states it "
         "(a faithful paraphrase is fine). Claims based on outside knowledge, assumptions, or "
         "wrong numbers are unsupported, even if they are true in the real world. Give the number "
         "of the chunk that supports it as evidence.\n"
         "3. grounded = true only if every claim is supported.\n"
         "4. answers_question = true if the answer fully addresses what was asked. If something is "
         "missing, describe it in `missing`.\n"
         "5. In `feedback`, tell the writer exactly what to fix.\n\n"
         "Reply with only JSON in this format:\n"
         '{"claims": [{"claim": "...", "supported": true, "evidence": 1}], '
         '"grounded": true, "answers_question": true, "missing": "", "feedback": ""}'),
        ("user",
         f"Question: {question}\n\nContext:\n{format_context(docs)}\n\nAnswer to check:\n{answer}"),
    ]


def rewrite_query(question: str, tried: list[str], reason: str) -> list[tuple[str, str]]:
    tried_list = "\n".join(f"- {q}" for q in tried)
    return [
        ("system",
         "You improve search queries for a vector database of internal company documents. "
         "Write ONE new search query that is more likely to find the missing information: use "
         "synonyms, likely document wording, or focus on the part that was not found. Do not "
         "repeat a query that was already tried. Reply with only the query text."),
        ("user",
         f"Original question: {question}\n\nQueries already tried:\n{tried_list}\n\n"
         f"Why the last attempt failed: {reason}"),
    ]


def naive_rag(question: str, docs: list[Document]) -> list[tuple[str, str]]:
    """The usual one-shot RAG prompt, used as the baseline in evaluations."""
    return [
        ("system", "Answer the user's question using the context below."),
        ("user", f"Context:\n{format_context(docs)}\n\nQuestion: {question}"),
    ]
