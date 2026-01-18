"""
Módulo que contiene casos de prueba para el retriever de documentos.
"""

import pytest
from retrieval.retriever import Retriever
from helpers import make_document, mock_vectorstore


@pytest.mark.unit
class TestRetrieverBehavior:
    """
    Clase que define los casos de prueba para el retriever de documentos.
    """

    def test_retrieve_should_return_filtered_documents_when_above_min_score(
        self, soft_assert
    ):
        """
        Verifica que el retriever retorne solo documentos que superen el puntaje mínimo.
        """
        doc1 = make_document("A", source="file1.pdf")
        doc2 = make_document("B", source="file2.pdf")
        repo = mock_vectorstore([(doc1, 0.5), (doc2, 0.1)])

        retriever = Retriever(
            repository=repo, k=2, min_score=0.2, require_multiple_sources=False
        )
        docs, sources = retriever.retrieve("query")

        soft_assert(len(docs) == 1)
        soft_assert(docs[0].page_content == "A")
        soft_assert(sources == ["file1.pdf"])

    def test_retrieve_should_return_empty_when_not_enough_sources(self, soft_assert):
        """
        Verifica que el retriever retorne listas vacías cuando no hay suficientes fuentes.
        """
        doc1 = make_document("A", source="file1.pdf")
        repo = mock_vectorstore([(doc1, 0.5)])

        retriever = Retriever(
            repository=repo, k=2, min_score=0.2, require_multiple_sources=True
        )
        docs, sources = retriever.retrieve("query")

        soft_assert(docs == [])
        soft_assert(sources == [])
