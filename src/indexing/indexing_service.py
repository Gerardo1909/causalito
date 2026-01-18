"""
Módulo que contiene lógica para el servicio de indexación de documentos.
"""

from typing import List
from langchain_core.documents import Document
from vectorstore.base import VectorRepository
from indexing.chunking.chunking_id_strategy import ChunkIdStrategy
from indexing.chunking.base import Chunker


class IndexingService:
    """
    Se encarga de orquestar el proceso de indexación de documentos:
    documentos -> chunks -> IDs -> vectorstore.
    """

    def __init__(
        self,
        chunker: Chunker,
        chunk_id_strategy: ChunkIdStrategy,
        repository: VectorRepository,
    ):
        self.chunker = chunker
        self.chunk_id_strategy = chunk_id_strategy
        self.repository = repository

    def index(self, documents: List[Document]) -> None:
        """
        Indexa una lista de documentos.

        :param documents: Lista que contiene los documentos a indexar.
        :type documents: List[Document]
        """
        # 1. Primero separamos los documentos en chunks
        chunks = self.chunker.split(documents)

        # 2. Asignamos IDs únicos a los chunks
        chunks = self.chunk_id_strategy.assign_ids(chunks)

        # 3. Filtramos los chunks ya existentes en el repositorio
        existing_ids = self.repository.get_existing_ids()

        new_chunks = [
            chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            self.repository.add(new_chunks)
