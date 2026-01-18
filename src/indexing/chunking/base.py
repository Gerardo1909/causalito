"""
Módulo base para estrategias de chunking de documentos.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class Chunker(ABC):
    """
    Clase base abstracta para estrategias de chunking de documentos.
    """

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Separa una lista de documentos en fragmentos más pequeños.

        :param documents: Lista de documentos a separar en chunks.
        :type documents: List[Document]
        :return: Lista de documentos separados en chunks.
        :rtype: List[Document]
        """
        ...
