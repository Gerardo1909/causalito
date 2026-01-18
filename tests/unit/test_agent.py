"""
Módulo que contiene pruebas funcionales sobre el agente RAG.
"""

from unittest.mock import MagicMock

import pytest
import pytest_check as check
from helpers import make_document, mock_llm_response

from agent.prompt_builder import PromptBuilder
from agent.rag_agent import RAGAgent


@pytest.mark.unit
class TestPromptBuilderBehavior:
    """
    Clase que define casos de prueba para el comportamiento de `PromptBuilder`.
    """

    def test_prompt_builder_should_include_context_and_question_when_documents_provided(
        self, soft_assert
    ):
        """
        Verifica que el prompt incluye el contenido de los documentos y la pregunta.
        """
        docs = [
            make_document("El concepto A es importante.", source="s1"),
            make_document("Definición de B.", source="s2"),
        ]
        builder = PromptBuilder()
        prompt = builder.build("¿Qué es A?", docs)

        soft_assert("El concepto A es importante." in prompt)
        soft_assert("Definición de B." in prompt)
        soft_assert("¿Qué es A?" in prompt)
        soft_assert(
            "Utiliza solo la información del contexto" in prompt
            or "Utiliza solo la información" in prompt
        )


@pytest.mark.unit
class TestRagAgentOrchestration:
    """
    Pruebas que verifican la orquestación interna del agente RAG sin llamar a LLM reales.
    """

    def test_rag_agent_should_return_fallback_when_no_documents(self, soft_assert):
        """
        Verifica si el retriever no devuelve documentos, el agente debe devolver mensaje de fallback y lista vacía.
        """
        retriever = MagicMock()
        retriever.retrieve.return_value = ([], [])
        llm = MagicMock()
        builder = PromptBuilder()

        agent = RAGAgent(retriever=retriever, llm=llm, prompt_builder=builder)
        resp, sources = agent.answer("Una pregunta cualquiera")

        soft_assert(isinstance(resp, str))
        soft_assert(resp.startswith("No hay suficiente información"))
        soft_assert(sources == [])

    def test_rag_agent_should_build_prompt_and_call_llm_when_docs_present(self):
        """
        Verifica que el agente construye el prompt con los docs y llama a `llm.generate` con él.
        """
        docs = [make_document("texto útil", source="s")]
        retriever = MagicMock()
        retriever.retrieve.return_value = (docs, ["s"])

        expected_text = "LLM answer"
        llm = mock_llm_response(expected_text)

        # spy builder to capture the prompt passed
        builder = PromptBuilder()

        agent = RAGAgent(retriever=retriever, llm=llm, prompt_builder=builder)
        result_text, result_sources = agent.answer("Pregunta")

        check.is_instance(result_text, str)
        check.equal(result_text, expected_text)
        check.equal(result_sources, ["s"])

    def test_rag_agent_should_propagate_llm_errors(self):
        """
        Si `llm.generate` lanza una excepción, debe propagarse al llamador.
        """
        docs = [make_document("texto", source="s")]
        retriever = MagicMock()
        retriever.retrieve.return_value = (docs, ["s"])

        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM failed")

        agent = RAGAgent(retriever=retriever, llm=llm, prompt_builder=PromptBuilder())

        with pytest.raises(RuntimeError):
            agent.answer("Pregunta que falla")
