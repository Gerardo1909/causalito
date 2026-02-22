"""
Módulo que contiene lógica para la recuperación de documentos similares presentes en la base
de datos vectorial.
"""

import os
from typing import List, Optional, Tuple

from confidence.feature_extractor import FeatureLogger, RetrievalFeatureExtractor
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
        feature_logger: Optional[FeatureLogger] = None,
    ):
        self.repository = repository
        self.k = k
        self.min_score = min_score
        self.require_multiple_sources = require_multiple_sources
        self.feature_logger = feature_logger
        self._extractor = RetrievalFeatureExtractor()

    def retrieve(self, query: str) -> Tuple[List[str], List[str], List[float]]:
        """
        Recupera documentos similares a una consulta dada.

        :param query: Consulta hecha hacia el retriever.
        :type query: str
        :return: Tupla con tres listas: documentos recuperados, fuentes de dichos documentos
                 y scores de similitud correspondientes a los documentos filtrados.
        :rtype: Tuple[List[str], List[str], List[float]]
        """

        results = self.repository.similarity_search(query, k=self.k)

        # Separar documentos y scores
        docs_with_scores = [(doc, score) for doc, score in results]

        # Extraer features ANTES de filtrar (captura el retrieval completo)
        all_scores = [score for _, score in docs_with_scores]
        features = self._extractor.extract(
            query, [doc for doc, _ in docs_with_scores], all_scores
        )

        # Loguear features si el logger está disponible
        if self.feature_logger:
            self.feature_logger.log(features)

        # Se procede con filtros originales
        filtered_with_scores = [
            (doc, score) for doc, score in results if score >= self.min_score
        ]
        filtered = [doc for doc, _ in filtered_with_scores]
        filtered_scores = [score for _, score in filtered_with_scores]

        sources = [os.path.basename(doc.metadata.get("source", "")) for doc in filtered]
        sources = list(set(sources))

        if self.require_multiple_sources and len(sources) < 2:
            return [], [], []

        return filtered, sources, filtered_scores
