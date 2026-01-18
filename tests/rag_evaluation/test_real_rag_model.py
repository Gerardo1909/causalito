"""
Módulo que contiene pruebas de desempeño/conocimiento sobre el agente RAG.
"""

import time
from pathlib import Path

import pytest
import pytest_check as check

from helpers import (
    load_test_questions,
    response_contains_any
)

from main import build_agent


QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"


@pytest.mark.slow
class TestAgentRealModelEvaluation:
    """
    Clase que contiene pruebas de desempeño/conocimiento sobre el agente RAG.
    """
    def test_agent_should_answer_critical_bayesian_questions_when_using_real_model(
        self,
    ):
        """
        Verifica que el agente RAG responda correctamente preguntas críticas sobre inferencia bayesiana
        utilizando un modelo real.
        """
        agent = build_agent()
        questions = load_test_questions(QUESTIONS_PATH)[:1]

        for q in questions:
            response = agent.answer(q["question"])[0]

            check.is_not_none(response, msg="El modelo no devolvió ninguna respuesta.")

            check.is_true(
                response_contains_any(response, q["expected_answers"]),
                msg=f"[{q['id']}] Los conceptos esperados no se encontraron en la respuesta.\nRespuesta: {response}",
            )
            time.sleep(5)
