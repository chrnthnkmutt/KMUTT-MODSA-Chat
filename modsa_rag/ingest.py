from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.errors import NotFoundError
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modsa_rag.config import Settings


logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
MANIFEST_FILE = "source_manifest.json"


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_uses_ollama:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "langchain-ollama is required for Ollama embeddings. Install it with: pip install langchain-ollama",
            ) from exc
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.resolved_embedding_base_url,
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.resolved_embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def get_vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_dir),
        embedding_function=build_embeddings(settings),
    )


def discover_source_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(child)
    return sorted(files)


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def build_manifest(files: list[Path]) -> dict[str, object]:
    return {
        "files": [file_fingerprint(path) for path in files],
    }


def manifest_path(settings: Settings) -> Path:
    return settings.chroma_dir / MANIFEST_FILE


def load_manifest(settings: Settings) -> dict[str, object] | None:
    path = manifest_path(settings)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(settings: Settings, manifest: dict[str, object]) -> None:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(settings).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collection_document_count(settings: Settings) -> int:
    vector_store = get_vector_store(settings)
    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        return 0
    return int(collection.count())


def load_json_documents(path: Path) -> list[Document]:
    """Load a prepared MOD-SA chunks JSON into pre-chunked Documents.

    Expects the schema produced by ``pipeline/chunk.py``::

        {"doc_id": ..., "metadata": {...}, "chunks": [{content, page, section, ...}]}

    Each chunk becomes one Document with citation metadata merged in. The
    ``_prechunked`` flag tells :func:`split_documents` to leave these as-is.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    doc_meta = data.get("metadata", {})
    doc_id = data.get("doc_id", path.stem)

    documents: list[Document] = []
    for chunk in data.get("chunks", []):
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        metadata: dict[str, object] = {
            "source": doc_meta.get("source_name") or doc_id,
            "doc_id": doc_id,
            "chunk_id": chunk.get("chunk_id", ""),
            "_prechunked": True,
        }
        for key in ("category", "title", "department", "source_url", "language", "last_updated"):
            value = doc_meta.get(key)
            if value:
                metadata[key] = value
        if chunk.get("section"):
            metadata["section"] = chunk["section"]
        if chunk.get("page") is not None:
            # our pages are 1-indexed; rag.py displays page + 1, so store 0-indexed
            metadata["page"] = int(chunk["page"]) - 1
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def load_documents(files: list[Path]) -> tuple[list[Document], list[dict[str, object]]]:
    documents: list[Document] = []
    skipped: list[dict[str, object]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                documents.extend(load_json_documents(path))
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                skipped.append({"source": str(path), "reason": str(exc)})
            continue
        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()
        except ImportError as exc:
            logger.warning("Skipping %s because a loader dependency is missing: %s", path, exc)
            skipped.append({"source": str(path), "reason": str(exc)})
            continue
        for document in loaded:
            document.metadata["source"] = str(path)
            documents.append(document)
    return documents, skipped


def split_documents(settings: Settings, documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # Documents loaded from prepared chunks JSON are already chunked — keep them
    # as-is and only split the rest (PDF/text/markdown).
    prechunked = [doc for doc in documents if doc.metadata.get("_prechunked")]
    to_split = [doc for doc in documents if not doc.metadata.get("_prechunked")]
    for doc in prechunked:
        doc.metadata.pop("_prechunked", None)
    return splitter.split_documents(to_split) + prechunked


def reset_collection(settings: Settings) -> None:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.chroma_collection)
    except (ValueError, NotFoundError):
        pass


def ingest_sources(settings: Settings, force: bool = False) -> dict[str, object]:
    files = discover_source_files(settings.source_paths)
    current_manifest = build_manifest(files)
    previous_manifest = load_manifest(settings)

    if not force and previous_manifest == current_manifest:
        indexed_documents = collection_document_count(settings)
        if files and indexed_documents == 0:
            logger.warning(
                "Source manifest matches but the Chroma collection is empty; rebuilding index.",
            )
        else:
            return {
                "status": "skipped",
                "reason": "source files unchanged",
                "files": len(files),
            }

    reset_collection(settings)

    if not files:
        save_manifest(settings, current_manifest)
        return {
            "status": "empty",
            "reason": "no supported source files found",
            "files": 0,
            "chunks": 0,
        }

    documents, skipped = load_documents(files)
    if not documents:
        save_manifest(settings, current_manifest)
        return {
            "status": "empty",
            "reason": "no documents could be loaded",
            "files": len(files),
            "chunks": 0,
            "skipped": skipped,
        }

    chunks = split_documents(settings, documents)
    vector_store = get_vector_store(settings)
    vector_store.add_documents(chunks)
    save_manifest(settings, current_manifest)

    return {
        "status": "indexed",
        "files": len(files),
        "documents": len(documents),
        "chunks": len(chunks),
        "skipped": skipped,
    }
