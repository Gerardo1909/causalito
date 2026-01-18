"""
Módulo que contiene funciones de utilidad utilizadas por los casos de prueba.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from unittest.mock import MagicMock

from langchain_core.documents import Document


def make_document(page_content: str, source: str = "doc") -> Document:
    """
    Crea un objeto `Document` con metadatos mínimos.
    """
    return Document(page_content=page_content, metadata={"source": source})


def mock_vectorstore(similarities: List[Tuple[Document, float]]) -> MagicMock:
    """
    Construye un `MagicMock` que emula un `VectorRepository`.

    `similarities` debe ser una lista de tuplas `(Document, score)`.
    El mock expone `similarity_search` que retorna esa lista.
    """
    repo = MagicMock()
    repo.similarity_search.return_value = similarities
    return repo


def mock_llm_response(text: str) -> MagicMock:
    """
    Crea un mock de LLM con método `generate` que devuelve `text`.
    """
    llm = MagicMock()
    llm.generate.return_value = text
    return llm


def _normalize(text: str) -> str:
    """
    Noramaliza el texto para comparación (minúsculas y sin espacios alrededor).
    """
    return text.lower().strip()


def load_test_questions(path: Path) -> List[Dict[str, Any]]:
    """
    Carga preguntas de prueba desde un archivo JSON.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def response_contains_any(response: str, expected: Iterable[str]) -> bool:
    """
    Verifica si la respuesta contiene al menos uno de los textos esperados.
    """
    response_norm = _normalize(response)
    return any(_normalize(e) in response_norm for e in expected)


def response_contains_all(response: str, required: Iterable[str]) -> bool:
    """
    Verifica si la respuesta contiene todos los textos requeridos.
    """
    response_norm = _normalize(response)
    return all(_normalize(r) in response_norm for r in required)
