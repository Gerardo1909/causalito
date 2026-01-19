"""
Módulo que contiene la lógica de orquestación del RAG.
"""

from typing import List, Tuple

from agent.prompt_builder import PromptBuilder
from llm.base import LLM
from retrieval.retriever import Retriever


class RAGAgent:
    """
    Clase que implementa un agente de Recuperación-Augmentada por Generación (RAG).
    """

    def __init__(self, retriever: Retriever, llm: LLM, prompt_builder: PromptBuilder):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder

    def answer(self, question: str) -> Tuple[str, List[str]]:
        """
        Genera una respuesta a una pregunta utilizando documentos recuperados por el retriever.

        :param question: Consulta realizada por el usuario.
        :type question: str
        :return: Respuesta generada y lista de fuentes utilizadas.
        :rtype: Tuple[str, list[str]]
        """
        docs, sources = self.retriever.retrieve(question)

        if not docs:
            return "No hay suficiente información en los documentos para responder.", []

        prompt = self.prompt_builder.build(question, docs)
        return self.llm.generate(prompt), sources
