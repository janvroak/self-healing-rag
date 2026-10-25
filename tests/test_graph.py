"""Offline tests of the self-correcting loop, using scripted fake LLMs (no API calls)."""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from shrag.nodes import FALLBACK_MESSAGE, LoopLimits, citation_problems
from shrag.pipeline import SelfHealingRAG
from shrag.structured import Critique, StructuredOutputError, extract_json, invoke_structured

DOCS = [
    Document("Employees get 24 days of paid leave per year.", metadata={"chunk_id": "handbook.md#1"}),
    Document("The office cafeteria opens at 8am.", metadata={"chunk_id": "handbook.md#2"}),
]


class FakeRetriever:
    def __init__(self, docs=DOCS):
        self.docs = docs
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Document]:
        self.queries.append(query)
        return self.docs[:k]


def critique(grounded=True, answers=True, unsupported=(), missing=""):
    claims = [{"claim": "24 days of leave", "supported": True, "evidence": 1}]
    claims += [{"claim": c, "supported": False, "evidence": None} for c in unsupported]
    return json.dumps({"claims": claims, "grounded": grounded, "answers_question": answers,
                       "missing": missing, "feedback": "fix it" if unsupported else ""})


def make_rag(generator_replies, critic_replies, retriever=None, **limits):
    return SelfHealingRAG(
        retriever or FakeRetriever(),
        FakeListChatModel(responses=generator_replies),
        FakeListChatModel(responses=critic_replies),
        LoopLimits(**limits),
    )


def steps(result):
    return [event["step"] for event in result.trace]


def test_grounded_answer_is_accepted_first_time():
    rag = make_rag(['{"relevant": [1]}', "Employees get 24 days of paid leave [1]."], [critique()])
    result = rag.ask("How much leave do employees get?")

    assert result.status == "answered"
    assert result.answer == "Employees get 24 days of paid leave [1]."
    assert result.sources == ["handbook.md#1"]
    assert steps(result) == ["retrieve", "grade_documents", "generate", "critique", "finalize"]
    assert result.llm_calls == 3


def test_hallucinated_answer_is_regenerated_with_feedback():
    rag = make_rag(
        ['{"relevant": [1]}',
         "Employees get 24 days of leave [1], plus unlimited sick days [1].",
         "Employees get 24 days of paid leave [1]."],
        [critique(grounded=False, unsupported=["unlimited sick days"]), critique()],
    )
    result = rag.ask("How much leave do employees get?")

    assert result.status == "answered"
    assert "unlimited" not in result.answer
    assert steps(result) == ["retrieve", "grade_documents", "generate", "critique",
                             "generate", "critique", "finalize"]
    assert result.trace[4]["details"]["used_feedback"] is True


def test_no_relevant_docs_rewrites_query_then_declines():
    retriever = FakeRetriever()
    rag = make_rag(
        ['{"relevant": []}', "parking policy", '{"relevant": []}', "car park rules", '{"relevant": []}'],
        [], retriever=retriever, max_retrievals=3,
    )
    result = rag.ask("What is the parking policy?")

    assert result.status == "insufficient"
    assert result.answer.startswith(FALLBACK_MESSAGE)
    assert result.retrieval_attempts == 3
    assert retriever.queries == ["What is the parking policy?", "parking policy", "car park rules"]


def test_retry_limits_stop_endless_rejection():
    # The critic rejects everything; the loop must still terminate.
    rag = make_rag(
        ['{"relevant": [1]}', "bad answer [1]", "bad answer [1]", "new query"] * 2
        + ['{"relevant": [1]}', "bad answer [1]", "bad answer [1]"],
        [critique(grounded=False, unsupported=["made up"])] * 6,
        max_retrievals=3, max_generations=2,
    )
    result = rag.ask("Anything?")

    assert result.status == "insufficient"
    assert result.retrieval_attempts == 3
    assert steps(result).count("generate") == 6
    assert steps(result)[-1] == "fallback"


def test_incomplete_but_grounded_answer_triggers_re_retrieval_then_partial():
    rag = make_rag(
        ['{"relevant": [1]}', "Employees get 24 days of leave [1].", "leave carry over"]
        + ['{"relevant": [1]}', "Employees get 24 days of leave [1]."],
        [critique(answers=False, missing="carry-over rules")] * 2,
        max_retrievals=2,
    )
    result = rag.ask("How much leave do we get and can it carry over?")

    assert result.status == "partial"
    assert "24 days" in result.answer
    assert "carry-over rules" in result.answer
    assert steps(result).count("retrieve") == 2


def test_generator_saying_insufficient_skips_critic():
    rag = make_rag(['{"relevant": [2]}', "INSUFFICIENT_CONTEXT", "q2", '{"relevant": [2]}',
                    "INSUFFICIENT_CONTEXT"], [], max_retrievals=2)
    result = rag.ask("What is the CEO's salary?")

    assert result.status == "insufficient"
    assert "critique" not in steps(result)


def test_unparseable_critic_reply_counts_as_rejection():
    rag = make_rag(['{"relevant": [1]}', "Answer [1].", "Answer [1].", "q2", '{"relevant": [1]}',
                    "Answer [1].", "Answer [1]."],
                   ["looks fine to me!"] * 8, max_retrievals=2)
    result = rag.ask("How much leave?")

    assert result.status == "insufficient"


def test_citation_to_missing_chunk_is_rejected_even_if_critic_approves():
    rag = make_rag(['{"relevant": [1]}', "24 days [7].", "24 days [1]."], [critique(), critique()])
    result = rag.ask("How much leave?")

    assert result.status == "answered"
    assert result.answer == "24 days [1]."
    assert result.trace[3]["details"]["bad_citations"] == [7]


def test_citation_problems():
    assert citation_problems("a [1] b [2][3]", 3) == []
    assert citation_problems("a [0] b [4]", 3) == [0, 4]


def test_extract_json_handles_thinking_fences_and_prose():
    reply = '<think>{"not": "this"}</think> Sure! ```json\n{"relevant": [1, 2]}\n```'
    # <think> is stripped by text_of before parsing; extract_json takes the first object it sees
    assert extract_json(reply.split("</think>")[1]) == {"relevant": [1, 2]}
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_invoke_structured_retries_once_then_succeeds():
    llm = FakeListChatModel(responses=["oops", critique()])
    result, calls = invoke_structured(llm, [("user", "check")], Critique)
    assert result.grounded and calls == 2
