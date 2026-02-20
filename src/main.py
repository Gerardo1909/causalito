"""
Punto de entrada para inicializar a causalito.
"""

from agent.prompt_builder import PromptBuilder
from agent.rag_agent import RAGAgent
from confidence.feature_extractor import FeatureLogger
from config.settings import CHROMA_PERSIST_DIR, DATA_DIR
from indexing.embedders.sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
)
from llm.qroq_llm import GroqLLM
from retrieval.retriever import Retriever
from vectorstore.chroma_repository import ChromaRepository


def build_agent() -> RAGAgent:
    """
    Se encarga de construir y configurar el agente RAG completo.

    :return: Agente inicializado
    :rtype: RAGAgent
    """
    # Embeddings
    embedder = SentenceTransformersEmbedder()
    feature_logger = FeatureLogger(DATA_DIR / "eval_log.csv")

    # Vector store (repository)
    repository = ChromaRepository(
        persist_dir=str(CHROMA_PERSIST_DIR),
        embedder=embedder,
    )

    # Retriever
    retriever = Retriever(
        repository=repository,
        k=15,
        min_score=0.1,
        require_multiple_sources=False,
        feature_logger=feature_logger,
    )

    # LLM
    llm = GroqLLM()

    # Prompt builder
    prompt_builder = PromptBuilder()

    # Agent
    return RAGAgent(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder,
    )
