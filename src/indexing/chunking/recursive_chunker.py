"""
Módulo donde se implementa estrategia de chunking recursivo.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from indexing.chunking.base import Chunker


class RecursiveChunker(Chunker):
    """
    Clase que implementa un generador de chunks recursivo.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
