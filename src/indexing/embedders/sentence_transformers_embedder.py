"""
Embedder basado en sentence-transformers (all-MiniLM-L6-v2).
"""

from sentence_transformers import SentenceTransformer

from typing import List
from indexing.embedders.base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """
    Implementación de Embedder usando sentence-transformers/all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
