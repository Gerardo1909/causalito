"""
Módulo principal para la ingesta y procesamiento de documentos.
"""

import json
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document

from config.settings import CHROMA_PERSIST_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from indexing.chunking.chunking_id_strategy import ChunkIdStrategy
from indexing.chunking.recursive_chunker import RecursiveChunker
from indexing.embedders.sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
)
from indexing.indexing_service import IndexingService
from ingestion.loaders import PDFLoader
from ingestion.text_cleaner import TextCleaner
from vectorstore.chroma_repository import ChromaRepository


def load_and_clean_documents() -> None:
    """
    Carga documentos PDF, los limpia y los persiste en PROCESSED_DATA_DIR como JSONL.
    """
    print("[INFO] Cargando documentos PDF desde:", RAW_DATA_DIR)
    loader = PDFLoader(data_dir=Path(RAW_DATA_DIR))
    raw_documents = loader.load()
    print(f"[INFO] {len(raw_documents)} documentos cargados. Limpiando...")
    cleaner = TextCleaner()
    processed_path = PROCESSED_DATA_DIR / "clean_documents.jsonl"
    with open(processed_path, "w", encoding="utf-8") as f:
        for doc in raw_documents:
            print(f"[DEBUG] Limpiando documento: {doc.metadata.get('source')}")
            clean_doc = cleaner.clean_document(doc)
            if not clean_doc:
                continue
            # Guardar como JSONL: page_content y metadata
            json.dump(
                {
                    "page_content": clean_doc.page_content,
                    "metadata": clean_doc.metadata,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    print(f"[INFO] Documentos limpios guardados en {processed_path}")


def iter_clean_documents() -> Generator[dict, None, None]:
    """
    Itera sobre los documentos limpios guardados en PROCESSED_DATA_DIR.
    """
    processed_path = PROCESSED_DATA_DIR / "clean_documents.jsonl"
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            yield doc


def index_documents() -> None:
    """
    Inicializa servicios y realiza la indexación de los documentos limpios uno a uno.
    """
    print("[INFO] Inicializando servicios de embeddings y vectorstore...")
    embedder = SentenceTransformersEmbedder()
    repository = ChromaRepository(
        persist_dir=str(CHROMA_PERSIST_DIR),
        embedder=embedder,
    )
    chunker = RecursiveChunker(
        chunk_size=1500,
        chunk_overlap=250,
    )
    indexing_service = IndexingService(
        chunker=chunker,
        chunk_id_strategy=ChunkIdStrategy(),
        repository=repository,
    )
    print("[INFO] Indexando documentos limpios uno a uno...")

    batch = []
    for i, doc_dict in enumerate(iter_clean_documents()):
        doc = Document(
            page_content=doc_dict["page_content"], metadata=doc_dict["metadata"]
        )
        batch.append(doc)
        # Indexar en lotes pequeños para eficiencia y evitar OOM
        if len(batch) >= 8:
            indexing_service.index(batch)
            print(f"[INFO] Indexados {i + 1} documentos...")
            batch = []
    if batch:
        indexing_service.index(batch)
        print(f"[INFO] Indexados {i + 1} documentos (final).")
    print("[INFO] Proceso de indexación finalizado.")


def run_ingestion() -> None:
    """
    Ejecuta el pipeline completo de ingesta y procesamiento de documentos.
    """
    print("[INFO] Iniciando pipeline de ingesta y procesamiento...")
    load_and_clean_documents()
    index_documents()
    print("[INFO] Pipeline de ingesta completado.")


if __name__ == "__main__":
    run_ingestion()
