"""
Módulo que contiene implementación de VectorRepository usando Chroma como backend.
"""

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from indexing.embedders.base import Embedder
from vectorstore.base import VectorRepository


class ChromaRepository(VectorRepository):
    """
    Implementación de VectorRepository usando Chroma como backend.
    """

    def __init__(self, persist_dir: str, embedder: Embedder):
        self._db = Chroma(persist_directory=persist_dir, embedding_function=embedder)

    def add(self, documents: List[Document]) -> None:
        ids = [doc.metadata["id"] for doc in documents]
        self._db.add_documents(documents, ids=ids)

    def get_existing_ids(self) -> set[str]:
        items = self._db.get(include=[])
        return set(items["ids"])

    def similarity_search(self, query: str, k: int) -> List[tuple[Document, float]]:
        return self._db.similarity_search_with_score(query, k=k)

    def delete(self, ids: list[str]) -> None:
        self._db.delete(ids=ids)
