"""
Módulo que contiene la clase base para modelar abstracción de
bases de datos vectoriales.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Set


class VectorRepository(ABC):
    """
    Clase base que define el comportamiento de un repositorio vectorial.
    """

    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        """
        Agrega documentos al repositorio vectorial.

        :param documents: Lista de documentos a agregar.
        :type documents: List[Document]
        """
        ...

    @abstractmethod
    def get_existing_ids(self) -> Set[str]:
        """
        Recupera los IDs existentes en el repositorio vectorial.

        :return: Conjunto de IDs existentes.
        :rtype: Set[str]
        """
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[tuple[Document, float]]:
        """
        Ejecuta una búsqueda de similitud en el repositorio vectorial.

        :param query: Consulta de texto.
        :type query: str
        :param k: Número de resultados a retornar.
        :type k: int
        :return: Lista de tuplas con documentos y sus puntajes de similitud.
        :rtype: List[tuple[Document, float]]
        """
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Elimina documentos del repositorio vectorial por sus IDs.

        :param ids: IDs de los documentos a eliminar.
        :type ids: List[str]
        """
        ...
