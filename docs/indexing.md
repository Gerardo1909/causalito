
# Indexación

## Propósito

El módulo de indexación transforma documentos limpios en vectores indexables y los almacena en el vector store. Incluye la lógica de chunking, asignación de IDs y la coordinación con el repositorio vectorial.

## Responsabilidades

- Fragmentar (`chunking`) documentos grandes en fragmentos manejables.
- Generar embeddings para los fragments (vía `Embedder`).
- Asignar identificadores estables a los chunks (`ChunkIdStrategy`).
- Filtrar chunks ya indexados y añadir solo los nuevos al `VectorRepository`.

Queda fuera el proceso de recuperación en tiempo real y la generación por LLM.

## ¿Cómo encaja en el pipeline RAG?

CLEANED DOCUMENTS (`clean_documents.jsonl`)
  ↓
CHUNKER (`Chunker` / `RecursiveChunker`)
  ↓
ASIGNAR IDS (`ChunkIdStrategy`)
  ↓
EMBEDDINGS (`Embedder` / `SentenceTransformersEmbedder`)
  ↓
VECTOR STORE (`VectorRepository` / `ChromaRepository`)

## Componentes principales

- `src/indexing/indexing_service.py` — `IndexingService`:
  - Orquesta: split → assign_ids → filter existing → add new.
  - Método clave: `index(documents: List[Document])`.

- `src/indexing/chunking/`:
  - `Chunker` (abstracto) define `split(documents) -> List[Document]`.
  - `RecursiveChunker` usa `RecursiveCharacterTextSplitter` con parámetros `chunk_size` y `chunk_overlap`.
  - `ChunkIdStrategy` genera IDs estables por `source:page:chunk_index:version` y exige que cada chunk tenga `metadata['source']` y `metadata['page']`.

- `src/indexing/embedders/`:
  - `Embedder` (abstracto): `embed_documents(texts)` y `embed_query(text)`.
  - `SentenceTransformersEmbedder` usa `sentence-transformers` (`all-MiniLM-L6-v2` por defecto).


## Limitaciones

- La estrategia de IDs asume metadatos completos (`source` y `page`). Documents sin estos campos producirán excepciones.
- No hay lógica incorporada para normalizar metadatos antes de la generación de IDs.
- La elección del modelo de embeddings afecta directamente la calidad de la recuperación; el sistema depende de que el embedder esté correctamente configurado.