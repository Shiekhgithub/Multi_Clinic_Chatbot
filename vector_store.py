"""
vector_store.py
---------------
Creates (or loads) ChromaDB vector stores for each dataset
using the sentence-transformers/all-MiniLM-L6-v2 embedding model.
"""

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Shared embedding model (loaded once)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_fn = None  # lazy singleton


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and cache) the HuggingFace embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn


# ──────────────────────────────────────────────
# Collection names
# ──────────────────────────────────────────────

COLLECTIONS = {
    "heart_disease": "heart_disease_index",
    "dermatology": "dermatology_index",
    "diabetes": "diabetes_index",
}


def _get_or_create_store(
    collection_name: str,
    persist_dir: str,
    documents: list[Document] | None = None,
) -> Chroma:
    """
    Return a Chroma vector store.
    - If `documents` is provided → build from scratch.
    - Otherwise           → load existing persisted store.
    """
    embeddings = get_embeddings()
    store_path = os.path.join(persist_dir, collection_name)

    if documents:
        print(f"[VectorStore] Building '{collection_name}' with {len(documents)} documents …")
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=store_path,
        )
        print(f"[VectorStore] '{collection_name}' saved to {store_path}")
    else:
        print(f"[VectorStore] Loading existing '{collection_name}' from {store_path}")
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=store_path,
        )

    return store


def build_all_stores(
    persist_dir: str,
    heart_docs: list[Document],
    derm_docs: list[Document],
    diabetes_docs: list[Document],
) -> dict[str, Chroma]:
    """Build all three vector stores and return them as a dict."""
    return {
        "heart_disease": _get_or_create_store(
            COLLECTIONS["heart_disease"], persist_dir, heart_docs
        ),
        "dermatology": _get_or_create_store(
            COLLECTIONS["dermatology"], persist_dir, derm_docs
        ),
        "diabetes": _get_or_create_store(
            COLLECTIONS["diabetes"], persist_dir, diabetes_docs
        ),
    }


def load_all_stores(persist_dir: str) -> dict[str, Chroma]:
    """Load all three pre-built vector stores from disk."""
    return {
        "heart_disease": _get_or_create_store(COLLECTIONS["heart_disease"], persist_dir),
        "dermatology": _get_or_create_store(COLLECTIONS["dermatology"], persist_dir),
        "diabetes": _get_or_create_store(COLLECTIONS["diabetes"], persist_dir),
    }


def stores_exist(persist_dir: str) -> bool:
    """Return True if all three ChromaDB collections already exist on disk."""
    for name in COLLECTIONS.values():
        path = os.path.join(persist_dir, name)
        if not os.path.isdir(path):
            return False
    return True
