"""JSON output from free models, which don't reliably support tool calling.

We ask for JSON in the prompt, pull the first JSON object out of the reply
(ignoring prose, code fences and <think> blocks), validate it with Pydantic,
and ask once more if it doesn't parse.
"""

import json
from typing import Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from shrag.llm import text_of


class StructuredOutputError(ValueError):
    pass


class RelevanceGrades(BaseModel):
    relevant: list[int] = Field(default_factory=list)


class Claim(BaseModel):
    claim: str
    supported: bool
    evidence: int | None = None  # number of the chunk that supports it


class Critique(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    grounded: bool
    answers_question: bool
    missing: str = ""
    feedback: str = ""


Verdict = Literal["accept", "regenerate", "re_retrieve"]


def extract_json(text: str) -> dict:
    """The first JSON object in `text`."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise StructuredOutputError(f"No JSON object in model reply: {text[:200]!r}")


def invoke_structured[T: BaseModel](
    llm: Runnable, messages: list[tuple[str, str]], schema: type[T], retries: int = 1
) -> tuple[T, int]:
    """Call the model and parse its reply as `schema`. Returns (result, calls made)."""
    conversation = list(messages)
    for attempt in range(retries + 1):
        reply = text_of(llm.invoke(conversation))
        try:
            return schema.model_validate(extract_json(reply)), attempt + 1
        except (StructuredOutputError, ValidationError) as exc:
            error = exc
            conversation += [
                ("assistant", reply or "(empty reply)"),
                ("user", "That was not valid JSON in the requested format. "
                         "Reply with only the JSON object, nothing else."),
            ]
    raise StructuredOutputError(str(error))
