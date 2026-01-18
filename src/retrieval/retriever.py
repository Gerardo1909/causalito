"""
Módulo que contiene lógica para la recuperación de documentos similares presentes en la base
de datos vectorial.
"""

import os

from typing import List, Tuple

from vectorstore.base import VectorRepository


class Retriever:
    """
    Clase que se encarga de recuperar documentos similares desde un vectorstore.
    """

    def __init__(
        self,
        repository: VectorRepository,
        k: int,
        min_score: float,
        require_multiple_sources: bool,
    ):
        self.repository = repository
        self.k = k
        self.min_score = min_score
        self.require_multiple_sources = require_multiple_sources

    def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Función que recupera documentos similares a una consulta dada.

        :param query: Consulta hecha hacia el retriever.
        :type query: str
        :return: Tupla con dos listas, la primera con los documentos recuperados y la segunda con las fuentes de dichos documentos.
        :rtype: Tuple[List[str], List[str]]
        """

        results = self.repository.similarity_search(query, k=self.k)

        filtered = [doc for doc, score in results if score >= self.min_score]
        sources = [os.path.basename(doc.metadata.get("source", "")) for doc in filtered]
        sources = list(set(sources))

        if self.require_multiple_sources:
            if len(sources) < 2:
                return [], []

        return filtered, sources
