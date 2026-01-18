"""
Módulo que se encarga de la carga de datos desde diversas fuentes
hacia el sistema de almacenamiento.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Abstracción para la carga de documentos PDF desde un directorio.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load(self) -> list[Document]:
        """
        Carga documentos PDF desde el directorio especificado.

        :return: Una lista de objetos Document que representa los documentos PDF cargados.
        :rtype: list[Document]
        """
        loader = PyPDFDirectoryLoader(str(self.data_dir))
        return loader.load()
