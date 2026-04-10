from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    # "data/python_intro.txt",
    # "data/vector_store_notes.md",
    # "data/rag_system_design.md",
    # "data/customer_support_playbook.txt",
    # "data/chunking_experiment_report.md",
    # "data/vi_retrieval_notes.md",
    "data/giao_trinh.md"
]
LLM_PROVIDER_ENV = "LLM_PROVIDER"
OPENAI_LLM_MODEL_ENV = "OPENAI_LLM_MODEL"
DEFAULT_LLM_PROVIDER = "demo"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CACHE_PATH = Path(".cache") / "manual_test_store.json"


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def chunk_documents(documents: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Document]:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunked_documents: list[Document] = []

    for document in documents:
        chunks = chunker.chunk(document.content)
        for index, chunk in enumerate(chunks):
            metadata = dict(document.metadata)
            metadata["doc_id"] = document.id
            metadata["chunk_index"] = index
            chunked_documents.append(
                Document(
                    id=f"{document.id}::chunk::{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )

    return chunked_documents


def make_preview_text(text: str, limit: int = 120) -> str:
    preview = text[:limit].replace("\n", " ")
    encoding = sys.stdout.encoding or "utf-8"
    return preview.encode(encoding, errors="replace").decode(encoding, errors="replace")


def compute_cache_metadata(
    documents: list[Document],
    embedding_backend: str,
    chunk_size: int,
) -> dict:
    return {
        "embedding_backend": embedding_backend,
        "chunk_size": chunk_size,
        "documents": [
            {
                "id": document.id,
                "source": document.metadata.get("source"),
                "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
            }
            for document in documents
        ],
    }


def load_cached_store_if_valid(store: EmbeddingStore, cache_path: str | Path, expected_metadata: dict) -> bool:
    try:
        stored_metadata = store.load_from_disk(cache_path)
    except Exception:
        return False
    return stored_metadata == expected_metadata


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def make_llm_fn():
    provider = os.getenv(LLM_PROVIDER_ENV, DEFAULT_LLM_PROVIDER).strip().lower()
    if provider != "openai":
        demo_llm._backend_name = "demo_llm"
        return demo_llm

    try:
        from openai import OpenAI

        client_kwargs = {}
        base_url = os.getenv(OPENAI_BASE_URL_ENV)
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        client = OpenAI(**client_kwargs)
        model_name = os.getenv(OPENAI_LLM_MODEL_ENV, DEFAULT_OPENAI_LLM_MODEL)

        def openai_llm(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer questions using the provided context. "
                            "If the context is insufficient, say so clearly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            if content is None:
                return ""
            return str(content)

        openai_llm._backend_name = model_name
        return openai_llm
    except Exception:
        demo_llm._backend_name = "demo_llm_fallback"
        return demo_llm


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    embedding_backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    print(f"\nEmbedding backend: {embedding_backend}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    cache_path = Path(os.getenv("RAG_CACHE_PATH", str(DEFAULT_CACHE_PATH)))
    cache_metadata = compute_cache_metadata(
        documents=docs,
        embedding_backend=embedding_backend,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )

    if load_cached_store_if_valid(store, cache_path, cache_metadata):
        print(f"Loaded cached vector store from {cache_path}")
    else:
        chunked_docs = chunk_documents(docs)
        print(f"\nChunked into {len(chunked_docs)} chunks using RecursiveChunker")
        store.add_documents(chunked_docs)
        store.save_to_disk(cache_path, metadata=cache_metadata)
        print(f"Saved vector store cache to {cache_path}")

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=5)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {make_preview_text(result['content'])}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm_fn = make_llm_fn()
    print(f"LLM backend: {getattr(llm_fn, '_backend_name', getattr(llm_fn, '__name__', 'unknown'))}")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
