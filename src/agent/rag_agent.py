"""
Módulo que contiene la lógica de orquestación del RAG.
"""

from typing import List, Optional, Tuple

from agent.prompt_builder import PromptBuilder
from confidence.confidence_engine import ConfidenceEngine, ConfidenceResult
from confidence.feature_extractor import RetrievalFeatureExtractor
from llm.base import LLM
from retrieval.retriever import Retriever


class RAGAgent:
    """
    Agente de Recuperación-Augmentada por Generación (RAG).

    Usa capa que estima confianza de la respuesta usando un modelo bayesiano.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        prompt_builder: PromptBuilder,
        confidence_engine: Optional[ConfidenceEngine],
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.confidence_engine = confidence_engine
        self._feature_extractor = RetrievalFeatureExtractor()

    def answer(
        self, question: str, use_confidence_engine: bool = True
    ) -> Tuple[str, List[str], Optional[ConfidenceResult]]:
        """
        Genera una respuesta a una pregunta utilizando documentos recuperados por el retriever.

        De forma opcional ofrece una estimación de confianza sobre la respuesta.

        :param question: Consulta realizada por el usuario.
        :type question: str
        :param use_confidence_engine: Flag que indica si se desea utilizar el motor de estimación de confianza.
        :type use_confidence_engine: bool
        :return: Respuesta generada, lista de fuentes utilizadas y resultado del motor de confianza.
        :rtype: Tuple[str, List[str], Optional[ConfidenceResult]]
        """
        docs, sources, scores = self.retriever.retrieve(question)

        confidence_result = None

        if not docs:
            return (
                "No hay suficiente información en los documentos para responder.",
                [],
                None,
            )

        # Extraer features del retrieval usando los scores reales del retriever
        features = self._feature_extractor.extract(question, docs, scores)

        # Estimar confianza
        if self.confidence_engine and use_confidence_engine:
            confidence_result = self.confidence_engine.predict(
                mean_similarity=features.mean_similarity,
                max_similarity=features.max_similarity,
                n_sources=features.n_sources,
            )

            # Si baja confianza, abstener
            if confidence_result.decision.value == "abstain":
                return confidence_result.reasoning, [], confidence_result

        # Generar respuesta normal
        prompt = self.prompt_builder.build(question, docs)
        answer = self.llm.generate(prompt)

        return answer, sources, confidence_result
