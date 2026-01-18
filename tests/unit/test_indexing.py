"""
Módulo que define casos de prueba relacionados al módulo de indexación.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from indexing.chunking.chunking_id_strategy import ChunkIdStrategy
from indexing.indexing_service import IndexingService


@pytest.mark.unit
class TestIndexingBehavior:
    """
    Clase que define casos de prueba para el servicio de indexación.
    """

    def test_chunk_id_strategy_should_assign_sequential_ids_when_same_page(
        self, soft_assert
    ):
        """
        Verifica que la estrategia de IDs asigne IDs secuenciales
        a chunks que provienen de la misma página.
        """
        chunks = [
            Document(page_content="a", metadata={"source": "file.pdf", "page": 1}),
            Document(page_content="b", metadata={"source": "file.pdf", "page": 1}),
            Document(page_content="c", metadata={"source": "file.pdf", "page": 2}),
        ]

        strategy = ChunkIdStrategy(version="vX")
        assigned = strategy.assign_ids(chunks)

        soft_assert(assigned[0].metadata.get("id") is not None)
        soft_assert(assigned[1].metadata.get("id") is not None)
        soft_assert(assigned[2].metadata.get("id") is not None)

        # Los primeros dos chunks son de la misma página, deben tener IDs 0 y 1
        # El tercer chunk es de otra página, debe reiniciar el contador a 0
        id0 = assigned[0].metadata["id"]
        id1 = assigned[1].metadata["id"]
        id2 = assigned[2].metadata["id"]

        soft_assert(id0.split(":")[2] == "0")
        soft_assert(id1.split(":")[2] == "1")
        soft_assert(id2.split(":")[2] == "0")

    def test_chunk_id_strategy_should_raise_value_error_when_missing_metadata(self):
        """
        Verifica que se lance ValueError si falta metadata necesaria.
        """
        chunks = [
            Document(page_content="x", metadata={"source": "s"}),
            Document(page_content="y", metadata={}),
        ]
        strategy = ChunkIdStrategy()
        with pytest.raises(ValueError):
            strategy.assign_ids(chunks)

    def test_indexing_service_should_add_new_chunks_when_repository_missing_ids(
        self, soft_assert
    ):
        """
        Verifica que el servicio de indexación agregue solo los chunks
        que no existen en el repositorio.
        """
        # Preparo el mock del chunker
        chunker = MagicMock()
        docs = [Document(page_content="x", metadata={"source": "s", "page": 1}), Document(page_content="y", metadata={"source": "s2", "page": 2})]
        chunker.split.return_value = docs

        # Se utiliza la estrategia real de asignación de IDs
        strategy = ChunkIdStrategy(version="v1")

        # El repositorio simula tener IDs diferentes a los que se asignarán
        repository = MagicMock()
        repository.get_existing_ids.return_value = {f"s1:None:0:v1"}

        service = IndexingService(
            chunker=chunker, chunk_id_strategy=strategy, repository=repository
        )

        # Llamo a index; debería llamar a repository.add con los nuevos chunks (los que no están en existing_ids)
        service.index(
            [Document(page_content="doc", metadata={"source": "s1", "page": 1})]
        )

        soft_assert(repository.get_existing_ids.called)
        soft_assert(repository.add.called)

    def test_indexing_service_should_not_add_when_all_chunks_exist(self, soft_assert):
        """
        Verifica que el servicio de indexación no agregue chunks
        si todos ya existen en el repositorio.
        """
        chunker = MagicMock()
        docs = [Document(page_content="x", metadata={"source": "s", "page": 1})]
        chunker.split.return_value = docs

        strategy = ChunkIdStrategy(version="v1")

        repository = MagicMock()
        assigned = strategy.assign_ids(
            [Document(page_content="x", metadata={"source": "s", "page": 1})]
        )
        existing_id = assigned[0].metadata["id"]
        repository.get_existing_ids.return_value = {existing_id}

        service = IndexingService(
            chunker=chunker, chunk_id_strategy=strategy, repository=repository
        )
        service.index(
            [Document(page_content="doc", metadata={"source": "s", "page": 1})]
        )

        soft_assert(repository.get_existing_ids.called)
        soft_assert(not repository.add.called)

    def test_indexing_service_should_propagate_error_when_assign_ids_raises(self):
        """
        Verifica que el servicio de indexación propague errores
        lanzados por la estrategia de asignación de IDs.
        """
        chunker = MagicMock()
        docs = [Document(page_content="x", metadata={"source": "s", "page": 1})]
        chunker.split.return_value = docs

        strategy = MagicMock()
        strategy.assign_ids.side_effect = ValueError("bad metadata")

        repository = MagicMock()
        service = IndexingService(
            chunker=chunker, chunk_id_strategy=strategy, repository=repository
        )

        with pytest.raises(ValueError):
            service.index(docs)
