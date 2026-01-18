"""
Módulo para la limpieza y filtrado de texto en documentos académicos.
"""

import re
import unicodedata
from typing import Union

from langchain_core.documents import Document


class TextCleaner:
    """
    Clase que se encarga de la limpieza y filtrado de texto para libros y papers científicos
    orientados a RAG académico.
    """

    ROMAN_PAGE_PATTERN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
    HEADER_FOOTER_PATTERN = re.compile(
        r"^(contents|preface|chapter|section|appendix)\b",
        re.IGNORECASE,
    )
    COPYRIGHT_PATTERN = re.compile(
        r"(isbn|copyright|all rights reserved|printed in|library of congress)",
        re.IGNORECASE,
    )
    TABLE_OF_CONTENTS_PATTERN = re.compile(r"\.{3,}|\d{2,}$")
    MULTIPLE_SPACES = re.compile(r"\s+")
    BROKEN_WORD_PATTERN = re.compile(r"(\w+)-\s+(\w+)")

    MIN_CHAR_LENGTH = 200

    def clean_document(self, document: Document) -> Union[Document, None]:
        """
        Se encarga de limpiar y filtrar un documento. Si el documento
        se considera inválido después de la limpieza, devuelve None.

        :param document: Documento que se va a limpiar.
        :type document: Document
        :return: Documento limpio o None si se descarta.
        :rtype: Document | None
        """
        if self._should_drop(document):
            return None

        text = self.clean_text(document.page_content)
        if len(text) < self.MIN_CHAR_LENGTH:
            return None

        return Document(page_content=text, metadata=document.metadata)

    def _should_drop(self, document: Document) -> bool:
        text = document.page_content.strip()
        metadata = document.metadata

        if not text:
            return True

        if len(text) < self.MIN_CHAR_LENGTH:
            return True

        page_label = metadata.get("page_label", "")
        if self.ROMAN_PAGE_PATTERN.match(str(page_label)):
            return True

        if self.COPYRIGHT_PATTERN.search(text):
            return True

        if self._looks_like_table_of_contents(text):
            return True

        return False

    def _looks_like_table_of_contents(self, text: str) -> bool:
        lines = text.splitlines()
        toc_like = sum(
            1 for line in lines if self.TABLE_OF_CONTENTS_PATTERN.search(line)
        )
        return toc_like > len(lines) * 0.4

    def clean_text(self, text: str) -> str:
        """
        Ejecuta el flujo completo de limpieza de texto.

        :param text: Texto a limpiar.
        :type text: str
        :return: Texto limpio.
        :rtype: str
        """
        text = unicodedata.normalize("NFKC", text)
        text = self._fix_broken_words(text)
        text = self._remove_headers(text)
        text = self._fix_line_breaks(text)
        text = self._normalize_whitespace(text)
        text = self._remove_invalid_surrogates(text)
        return text.strip()

    def _fix_broken_words(self, text: str) -> str:
        return self.BROKEN_WORD_PATTERN.sub(r"\1\2", text)

    def _remove_headers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = [
            line for line in lines if not self.HEADER_FOOTER_PATTERN.match(line.strip())
        ]
        return "\n".join(cleaned)

    def _fix_line_breaks(self, text: str) -> str:
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return self.MULTIPLE_SPACES.sub(" ", text)

    def _remove_invalid_surrogates(self, text: str) -> str:
        return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
