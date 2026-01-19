
# Ingestión

## Propósito

El módulo de ingestión transforma fuentes crudas (principalmente PDFs académicos) en documentos limpios y persistentes listos para el pipeline RAG: limpieza, chunking, embeddings e indexación.

## Responsabilidades

- Cargar documentos desde disco (`PDFLoader`).
- Limpiar y filtrar contenido irrelevante o ruidoso (`TextCleaner`).
- Persistir documentos limpios en formato JSONL en el directorio de datos procesados.
- Orquestar la indexación de los documentos limpios en el vector store (funciones de `run.py`).

## ¿Cómo encaja en el pipeline RAG?

RAW DATA (PDFs)
	↓
LOADERS (`PDFLoader`)
	↓
CLEANING (`TextCleaner`)
	↓
PROCESSED DATA (`clean_documents.jsonl`)
	↓
CHUNKING → EMBEDDINGS → VECTOR STORE (indexación realizada desde `run.py`)

## Componentes principales

- `src/ingestion/loaders.py` — `PDFLoader`:
	- Carga PDFs en bruto desde un directorio (`data/raw/` por defecto, configurable).
	- Devuelve una lista de objetos `Document` compatibles con el resto del pipeline.

- `src/ingestion/text_cleaner.py` — `TextCleaner`:
	- Normaliza texto (unicode), corrige palabras partidas, elimina headers/footers y páginas tipo índice.
	- Reglas importantes:
		- Descarta documentos con menos de `MIN_CHAR_LENGTH = 200` caracteres.
		- Filtra páginas numeradas en números romanos y textos que contienen patrones de copyright.

- `src/ingestion/run.py`:
	- Funciones principales:
		- `load_and_clean_documents()` — carga y guarda `clean_documents.jsonl` en `PROCESSED_DATA_DIR`.
		- `iter_clean_documents()` — iterador sobre el JSONL previamente generado.
		- `index_documents()` — inicializa embedder (`SentenceTransformersEmbedder`), `ChromaRepository` y `IndexingService`, luego indexa en lotes.
		- `run_ingestion()` — orquesta todo el pipeline (carga → limpieza → indexación).

## Configuración

Los parámetros relevantes se cargan desde `src/config/settings.py`:

- `RAW_DATA_DIR` — directorio de PDFs de entrada.
- `PROCESSED_DATA_DIR` — directorio donde se escribe `clean_documents.jsonl`.
- `CHROMA_PERSIST_DIR` — directorio de persistencia para ChromaDB.

Comprueba y modifica esas variables en `src/config/settings.py` antes de ejecutar el pipeline.

## Limitaciones

- La limpieza basada en heurísticas puede fallar en PDFs muy heterogéneos (p. ej. artículos con figuras densas o formatos no estándar).
- No hay control de versión ni checksum de los documentos originales en el pipeline actual — considerar añadir hashes para trazabilidad.
- No se implementa deduplicación sofisticada de chunks — podría añadirse en `IndexingService` o `ChromaRepository`.