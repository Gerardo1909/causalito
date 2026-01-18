"""
Módulo que contiene lógica base para embebedores de texto.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Clase base para embebedores de texto.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embebe una lista de documentos de texto en vectores.

        :param texts: Lista de textos a embeber.
        :type texts: list[str]
        :return: Lista de representaciones vectoriales de los textos.
        :rtype: list[list[float]]
        """

        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embebe una consulta de texto en un vector.

         :param text: Texto de la consulta a embeber.
         :type text: str
         :return: Representación vectorial de la consulta.
         :rtype: list[float]
        """
        ...
