"""
Módulo que contiene casos de pruebas relacionados al módulo de ingestión
"""

import pytest
from helpers import make_document
from langchain_core.documents import Document

from ingestion.text_cleaner import TextCleaner


@pytest.mark.unit
class TestTextCleanerBehavior:
    """
    Casos de prueba del comportamiento de `TextCleaner`.

    Las pruebas comprueban limpieza, filtrado y normalización de texto.
    """

    def test_clean_document_should_return_clean_document_when_valid_input(
        self, soft_assert
    ):
        """
        Verifica que documentos largos y válidos deben retornar un `Document` limpio.
        """
        doc = make_document("Chapter 1\n" + "A" * 300, source="page-1")
        # añadir metadata de ejemplo similar a producción
        doc.metadata["page_label"] = "1"

        cleaner = TextCleaner()
        result = cleaner.clean_document(doc)

        soft_assert(result is not None)
        soft_assert(isinstance(result, Document))
        soft_assert("Chapter" not in result.page_content)

    def test_clean_document_should_return_none_when_short_or_noise(self, soft_assert):
        """
        Verifica que documentos vacíos, muy cortos o con etiquetas romanas deben descartarse.
        """
        cleaner = TextCleaner()
        doc_empty = make_document("", source="empty")
        doc_short = make_document("short text", source="short")
        doc_roman = make_document("Preface", source="roman")
        doc_roman.metadata["page_label"] = "iv"

        soft_assert(cleaner.clean_document(doc_empty) is None)
        soft_assert(cleaner.clean_document(doc_short) is None)
        soft_assert(cleaner.clean_document(doc_roman) is None)

    def test_clean_text_should_remove_headers_and_normalize_when_messy_text(
        self, soft_assert
    ):
        """
        Verifica que la limpieza de texto debe eliminar headers, unir líneas y normalizar espacios.
        """
        cleaner = TextCleaner()
        text = "Chapter 1\nThis is valid.\nSection\nAnother line."
        cleaned = cleaner.clean_text(text)

        soft_assert("Chapter" not in cleaned)
        soft_assert("Section" not in cleaned)
        soft_assert("This is valid." in cleaned)
        soft_assert("Another line." in cleaned)
