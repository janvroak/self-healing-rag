"""Document loading, chunking and the Chroma vector store (local embeddings)."""

from pathlib import Path
from typing import Protocol

from langchain_chroma import Chroma
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from shrag.config import PROJECT_ROOT, Settings

SUPPORTED = {".md", ".txt", ".pdf"}


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Document]: ...


def load_documents(path: Path) -> list[Document]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    docs = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED:
            continue
        if file.suffix.lower() == ".pdf":
            for page_no, page in enumerate(PdfReader(file).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(text, metadata={"source": file.name, "page": page_no}))
        else:
            docs.append(Document(file.read_text(encoding="utf-8"), metadata={"source": file.name}))
    return docs


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for chunk in chunks:
        key = f"{chunk.metadata['source']}:{chunk.metadata.get('page', '')}"
        counters[key] = counters.get(key, 0) + 1
        page = f"p{chunk.metadata['page']}-" if "page" in chunk.metadata else ""
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{page}{counters[key]}"
    return chunks


class LocalEmbeddings(Embeddings):
    """FastEmbed (ONNX) embeddings that run on the CPU: free and no API key."""

    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name, cache_dir=str(PROJECT_ROOT / ".cache" / "fastembed"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


_EMBEDDINGS: dict[str, LocalEmbeddings] = {}


def _embeddings(settings: Settings) -> LocalEmbeddings:
    if settings.embedding_model not in _EMBEDDINGS:
        _EMBEDDINGS[settings.embedding_model] = LocalEmbeddings(settings.embedding_model)
    return _EMBEDDINGS[settings.embedding_model]


def open_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest(path: Path, settings: Settings, reset: bool = True) -> tuple[int, int]:
    """Index every supported file under `path`. Returns (documents, chunks)."""
    docs = load_documents(path)
    if not docs:
        raise FileNotFoundError(f"No {', '.join(sorted(SUPPORTED))} files found in {path}")
    chunks = split_documents(docs, settings)
    store = open_store(settings)
    if reset:
        store.delete_collection()
        store = open_store(settings)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(docs), len(chunks)


class VectorRetriever:
    def __init__(self, store: Chroma):
        self.store = store

    def search(self, query: str, k: int) -> list[Document]:
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        docs = []
        for doc, score in results:
            doc.metadata["score"] = round(float(score), 3)
            docs.append(doc)
        return docs

    def is_empty(self) -> bool:
        return not self.store.get(limit=1)["ids"]
