"""
Módulo que define la estrategia para asignar IDs únicos a chunks de documentos.
"""

from typing import List
from langchain_core.documents import Document


class ChunkIdStrategy:
    """
    Estrategia para asignar IDs únicos a chunks de documentos.
    """

    def __init__(self, version: str = "v1"):
        self.version = version

    def assign_ids(self, chunks: List[Document]) -> List[Document]:
        """
        Asigna IDs únicos a cada chunk basado en su fuente y página.

        :param chunks: Lista de chunks a los que se les asignarán IDs.
        :type chunks: List[Document]
        :return: Lista de chunks con IDs asignados.
        :rtype: List[Document]
        """

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            # 1. Obtener la fuente y página del chunk
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            if source is None or page is None:
                raise ValueError(
                    "Los metadatos del chunk deben contener 'source' y 'page'"
                )

            # 2. Construir el ID del chunk
            current_page_id = f"{source}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk_id = f"{current_page_id}:{current_chunk_index}:{self.version}"

            # 3. Asignar el ID al chunk
            chunk.metadata["id"] = chunk_id
            last_page_id = current_page_id

        return chunks
