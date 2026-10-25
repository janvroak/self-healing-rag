"""Free LLMs: OpenRouter `:free` models with a local Ollama model as the last fallback."""

import re
from functools import lru_cache

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from shrag.config import Settings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMNotConfiguredError(RuntimeError):
    pass


@lru_cache
def ollama_available(base_url: str, model: str) -> bool:
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    return model in names or f"{model}:latest" in names


def build_llm(models: list[str], settings: Settings) -> Runnable:
    """A chat model that tries each model in turn until one answers."""
    candidates: list[BaseChatModel] = []
    if settings.openrouter_api_key:
        for model in models:
            candidates.append(
                ChatOpenAI(
                    model=model,
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                    temperature=0,
                    timeout=settings.llm_timeout,
                    max_retries=1,  # fail fast so the next model can take over
                    default_headers={"X-Title": "Self-Healing RAG"},
                )
            )
    if settings.ollama_model and ollama_available(settings.ollama_base_url, settings.ollama_model):
        candidates.append(
            ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0,
                       num_ctx=8192)  # the critic prompt holds up to 8 chunks
        )
    if not candidates:
        raise LLMNotConfiguredError(
            "No LLM available: set OPENROUTER_API_KEY in .env or run Ollama "
            f"with `ollama pull {settings.ollama_model or 'llama3'}`."
        )
    primary, *fallbacks = candidates
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def text_of(message: BaseMessage | str) -> str:
    """Plain text of a model reply, without any <think> reasoning blocks."""
    content = message if isinstance(message, str) else message.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return _THINK.sub("", content).strip()
