"""Settings, read from environment variables (and a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # OpenRouter free models. Each list is tried in order; if one fails or is
    # rate-limited the next one takes over, and local Ollama is the last resort.
    openrouter_api_key: str | None
    openrouter_base_url: str
    generator_models: list[str]
    critic_models: list[str]
    ollama_model: str | None
    ollama_base_url: str
    llm_timeout: int

    # Local embeddings (FastEmbed / ONNX, no API needed) and the vector store.
    embedding_model: str
    chroma_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int

    # Self-correction loop limits.
    max_retrievals: int
    max_generations: int
    base_k: int
    k_step: int


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        generator_models=_list(
            "GENERATOR_MODELS",
            "nvidia/nemotron-3.5-lightning:free,google/gemma-4-31b-it:free,"
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ),
        # A different (larger) model judges the answers, so the generator is
        # not grading its own homework.
        critic_models=_list(
            "CRITIC_MODELS",
            "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,"
            "nvidia/nemotron-3.5-lightning:free",
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3") or None,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_timeout=_int("LLM_TIMEOUT", 90),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_dir=Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / ".chroma")),
        collection=os.getenv("CHROMA_COLLECTION", "knowledge_base"),
        chunk_size=_int("CHUNK_SIZE", 700),
        chunk_overlap=_int("CHUNK_OVERLAP", 120),
        max_retrievals=_int("MAX_RETRIEVALS", 3),
        max_generations=_int("MAX_GENERATIONS", 2),
        base_k=_int("BASE_K", 4),
        k_step=_int("K_STEP", 2),
    )
