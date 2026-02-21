import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from langchain_core.documents import Document

from confidence.feature_extractor import (
    FeatureLogger,
    RetrievalFeatureExtractor,
    RetrievalFeatures,
)


@pytest.mark.unit
class TestRetrievalFeatureExtractor:
    """
    Clase que define casos de prueba para el extractor de features
    del retriever.
    """

    def test_extract_should_calculate_statistics(self):
        """Verifica que las estadísticas se calculen correctamente."""
        query = "¿Qué es una red bayesiana?"
        docs = [
            Document(page_content="A" * 100, metadata={"source": "doc1.pdf"}),
            Document(page_content="B" * 150, metadata={"source": "doc2.pdf"}),
        ]
        scores = [0.9, 0.7]

        features = RetrievalFeatureExtractor.extract(query, docs, scores)

        assert features.query == query
        assert features.mean_similarity == 0.8
        assert features.max_similarity == 0.9
        assert features.min_similarity == 0.7
        assert features.n_sources == 2
        assert features.context_length == 251  # 100 + 150 + 1 query

    def test_extract_should_handle_empty_documents(self):
        """Maneja caso sin documentos."""
        features = RetrievalFeatureExtractor.extract("query", [], [])

        assert features.mean_similarity == 0.0
        assert features.max_similarity == 0.0
        assert features.n_sources == 0


@pytest.mark.unit
class TestFeatureLogger:
    """
    Clase que contiene casos de prueba para el proceso de logging de respuestas
    y sus correspondientes métricas.
    """

    def test_logger_should_create_csv_with_headers(self):
        """Crea archivo CSV con headers correctos."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = FeatureLogger(log_path)

            # Verificar que existe y tiene headers
            with open(log_path) as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == [
                    "query",
                    "mean_similarity",
                    "max_similarity",
                    "min_similarity",
                    "n_sources",
                    "context_length",
                    "label",
                    "timestamp",
                ]

    def test_logger_should_append_rows(self):
        """Agrega filas al CSV."""
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = FeatureLogger(log_path)

            features = RetrievalFeatures(
                query="test",
                mean_similarity=0.8,
                max_similarity=0.9,
                min_similarity=0.7,
                n_sources=2,
                context_length=250,
                timestamp="2024-01-01T00:00:00",
            )

            logger.log(features, label="1")

            # Verificar que se escribió
            with open(log_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                row = next(reader)
                assert row[0] == "test"
                assert row[6] == "1"  # label
