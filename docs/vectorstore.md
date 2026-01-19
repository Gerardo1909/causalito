
# Vector Store (Chroma)

## Propósito

El módulo de Vector Store tiene como propósito proveer una abstracción y una implementación concreta para almacenar y recuperar representaciones vectoriales (embeddings) de fragmentos de documento dentro del pipeline RAG.

## Responsabilidades

- Definir una interfaz limpia para operaciones vectoriales: `add`, `get_existing_ids`, `similarity_search`, `delete`.
- Proveer una implementación concreta que use ChromaDB como backend y permita persistencia en disco.

## ¿Cómo encaja en el pipeline RAG?

EMBEDDINGS → VECTOR STORE (Chroma)
	↓
RETRIEVER (`Retriever`) → PROMPT → LLM

`IndexingService` escribe datos en el `VectorRepository`, y `Retriever` consulta el repositorio para obtener contexto.

## Componentes principales

- `src/vectorstore/base.py` — `VectorRepository`:
	- Interfaz abstracta que define:
		- `add(documents: List[Document]) -> None`
		- `get_existing_ids() -> Set[str]`
		- `similarity_search(query: str, k: int) -> List[tuple[Document, float]]`
		- `delete(ids: List[str]) -> None`

- `src/vectorstore/chroma_repository.py` — `ChromaRepository`:
	- Implementa `VectorRepository` usando `langchain_chroma.Chroma`.
	- Constructor: `ChromaRepository(persist_dir: str, embedder: Embedder)` — crea y configura la instancia de Chroma con la función de embedding proporcionada.
	- `add`: extrae `metadata['id']` de cada `Document` y llama a `self._db.add_documents`.
	- `get_existing_ids`: consulta `self._db.get(include=[])` y devuelve el conjunto de `ids` existentes.
	- `similarity_search`: delega en `self._db.similarity_search_with_score(query, k=k)`.
	- `delete`: delega en `self._db.delete(ids=ids)`.

## Configuración

- `persist_dir` debe apuntar a un directorio que la aplicación pueda leer/escribir. En este repositorio se usa `data/indexes/chroma` por defecto.
- Recomendado:
	- Mantener copias periódicas del directorio `persist_dir` (backup).
	- Monitorizar el tamaño de la base y el número de vectores.
	- Si se cambia el modelo de embeddings o se actualiza `ChunkIdStrategy.version`, considerar recrear el índice desde cero.

## Limitaciones

- `ChromaRepository` delega mucho comportamiento a la librería `langchain_chroma`; cambios en su API requieren actualizar esta capa.
- No existe versionado del índice dentro del `persist_dir` — la versión se gestiona indirectamente mediante `ChunkIdStrategy.version`.
