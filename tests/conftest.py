"""
Módulo que contiene fixtures para pruebas unitarias.
"""

import pytest
import pytest_check as check
from langchain_core.documents import Document
from unittest.mock import MagicMock


@pytest.fixture
def mock_documents():
    """
    Fixture que retorna una lista de documentos simulados para pruebas.
    """
    return [
        Document(
            page_content="La probabilidad causal se define como ...",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="En un DAG causal, una intervención do(X) ...",
            metadata={"source": "doc2"},
        ),
    ]


@pytest.fixture
def mock_retriever(mock_documents):
    """
    Fixture que retorna un retriever simulado para pruebas.

    Devuelve un objeto MagicMock con método `retrieve` que emula
    la tupla (documents, sources) que usan varios tests.
    """
    retriever = MagicMock()
    # Default: return (docs, list of sources)
    retriever.retrieve.return_value = (
        mock_documents,
        [d.metadata.get("source", None) for d in mock_documents],
    )
    return retriever


@pytest.fixture
def mock_llm():
    """
    Fixture que retorna un LLM simulado para pruebas. El mock expone
    `generate` y `call_count` compatible con los tests.
    """
    llm = MagicMock()
    llm.generate.return_value = (
        "Según el contexto, la inferencia causal bayesiana se basa en..."
    )
    return llm


@pytest.fixture
def soft_assert():
    """
    Provee una función de aserción "suave" que usa `pytest_check`.

    Uso:
        soft_assert(condición, mensaje_opcional)
    """

    def _soft_assert(expr, msg=None):
        check.is_true(bool(expr), msg)

    return _soft_assert
