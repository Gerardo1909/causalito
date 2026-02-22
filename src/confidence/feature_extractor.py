"""
Módulo que se encarga de la obtención de métricas del retriever para
analizar desempeño del agente.
"""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document


@dataclass
class RetrievalFeatures:
    """
    Features extraídas del retrieval para el modelo bayesiano.
    """

    query: str
    mean_similarity: float
    max_similarity: float
    min_similarity: float
    n_sources: int
    context_length: int
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class RetrievalFeatureExtractor:
    """
    Extrae features de documentos recuperados por el
    retriever.
    """

    @staticmethod
    def extract(
        query: str, documents: List[Document], scores: List[float]
    ) -> RetrievalFeatures:
        """
        Extrae features del retrieval.

        :param query: Consulta original hecha hacia el retriever.
        :type query: str
        :param documents: Documentos recuperados
        :type documents: List[Document]
        :param scores: Similarity scores
        :type scores: List[float]
        :return: RetrievalFeatures con estadísticas
        :rtype: RetrievalFeatures
        """
        if not documents:
            return RetrievalFeatures(
                query=query,
                mean_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                n_sources=0,
                context_length=0,
                timestamp=datetime.now().isoformat(),
            )

        # Extraer fuentes únicas
        sources = set()
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            sources.add(source)

        # Calcular estadísticas
        context_text = "\n".join([d.page_content for d in documents])

        return RetrievalFeatures(
            query=query,
            mean_similarity=sum(scores) / len(scores) if scores else 0.0,
            max_similarity=max(scores) if scores else 0.0,
            min_similarity=min(scores) if scores else 0.0,
            n_sources=len(sources),
            context_length=len(context_text),
            timestamp=datetime.now().isoformat(),
        )


class FeatureLogger:
    """
    Registra features en CSV para análisis posterior.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Crear CSV si no existe
        if not self.log_path.exists():
            self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Crea archivo CSV con headers."""
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query",
                    "mean_similarity",
                    "max_similarity",
                    "min_similarity",
                    "n_sources",
                    "context_length",
                    "label",
                    "timestamp",
                ]
            )

    def log(self, features: RetrievalFeatures, label: str = "") -> None:
        """
        Registra features en el CSV.

        :param features: Features extraídas
        :type features: RetrievalFeatures
        :param label: Etiqueta manual (1=correct, 0=incorrect, None=pending)
        :type label: str
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    features.query,
                    f"{features.mean_similarity:.4f}",
                    f"{features.max_similarity:.4f}",
                    f"{features.min_similarity:.4f}",
                    features.n_sources,
                    features.context_length,
                    label,
                    features.timestamp,
                ]
            )
