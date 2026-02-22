"""
Módulo que contiene el motor de confianza que estima la probabilidad que
una respuesta del agente RAG sea correcta.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from confidence.bayesian_model import BayesianInferenceResult


class Decision(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass
class ConfidenceResult:
    """Resultado de la estimación de confianza."""

    decision: Decision
    prob_correct: float  # P(respuesta es correcta)
    credible_interval_low: float
    credible_interval_high: float
    reasoning: str  # Por qué tomó esta decisión

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "prob_correct": float(self.prob_correct),
            "ci_low": float(self.credible_interval_low),
            "ci_high": float(self.credible_interval_high),
            "reasoning": self.reasoning,
        }


class ConfidenceEngine:
    """
    Estima confianza en respuestas RAG usando modelo bayesiano calibrado.

    Diseño:
        - Lee posteriors pre-entrenadas (de analyze_confidence.py)
        - Clasifica features en high/low confidence
        - Retorna decision + intervalo creíble
    """

    def __init__(
        self,
        high_confidence_posterior: Optional[BayesianInferenceResult] = None,
        low_confidence_posterior: Optional[BayesianInferenceResult] = None,
        abstention_threshold: float = 0.75,
    ):
        """
        Args:
            high_confidence_posterior: Resultado bayesiano para high-conf cases
            low_confidence_posterior: Resultado bayesiano para low-conf cases
            abstention_threshold: Si prob_correct < esto, abstiene
        """
        self.high_conf_posterior = high_confidence_posterior
        self.low_conf_posterior = low_confidence_posterior
        self.abstention_threshold = abstention_threshold

        # Default: neutral
        if not self.high_conf_posterior:
            self.high_conf_posterior = self._default_posterior()
        if not self.low_conf_posterior:
            self.low_conf_posterior = self._default_posterior()

    def predict(
        self,
        mean_similarity: float,
        max_similarity: float,
        n_sources: int,
        high_conf_threshold: float = 0.75,
        min_sources_threshold: int = 2,
    ) -> ConfidenceResult:
        """
        Estima confianza en la respuesta.

        Args:
            mean_similarity: Similitud promedio del retrieval
            max_similarity: Similitud máxima
            n_sources: Número de fuentes
            high_conf_threshold: Umbral para clasificar como high-conf
            min_sources_threshold: Mínimo de fuentes para high-conf

        Returns:
            ConfidenceResult con decision y reasoning
        """

        # Clasificar en grupo
        is_high_conf = (
            mean_similarity >= high_conf_threshold
            and n_sources >= min_sources_threshold
        )

        posterior = (
            self.high_conf_posterior if is_high_conf else self.low_conf_posterior
        )

        prob_correct = posterior.posterior_mean
        ci_low = posterior.credible_interval_low
        ci_high = posterior.credible_interval_high

        # Decidir
        if prob_correct >= self.abstention_threshold:
            decision = Decision.ANSWER
            reasoning = (
                f"Alta confianza ({prob_correct:.1%}). "
                f"Grupo: {'high' if is_high_conf else 'low'}. "
                f"IC 90%: [{ci_low:.2f}, {ci_high:.2f}]"
            )
        else:
            decision = Decision.ABSTAIN
            reasoning = (
                f"Confianza insuficiente ({prob_correct:.1%} < {self.abstention_threshold:.1%}). "
                f"No tengo certeza suficiente para responder."
            )

        return ConfidenceResult(
            decision=decision,
            prob_correct=prob_correct,
            credible_interval_low=ci_low,
            credible_interval_high=ci_high,
            reasoning=reasoning,
        )

    @staticmethod
    def _default_posterior() -> BayesianInferenceResult:
        """Posterior neutral (Beta(1,1))."""
        from confidence.bayesian_model import BayesianInferenceResult

        return BayesianInferenceResult(
            alpha_posterior=1.0,
            beta_posterior=1.0,
            posterior_mean=0.5,
            posterior_std=np.sqrt(1.0 / 12),
            credible_interval_low=0.025,
            credible_interval_high=0.975,
            n_successes=0,
            n_failures=0,
            n_total=0,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        high_conf_threshold: float = 0.75,
        min_sources: int = 2,
        abstention_threshold: float = 0.75,
    ) -> "ConfidenceEngine":
        """
        Factory: crea engine entrenado desde CSV etiquetado.

        Args:
            csv_path: Ruta a data/eval_log.csv
            high_conf_threshold: Umbral de similitud para high-conf
            min_sources: Mínimo de fuentes para high-conf
            abstention_threshold: Umbral de decision
        """
        from pathlib import Path

        from confidence.data_analyzer import DataAnalyzer

        analyzer = DataAnalyzer(Path(csv_path))
        high_conf_analysis, low_conf_analysis = analyzer.analyze_confidence_split(
            high_conf_threshold=high_conf_threshold, min_sources=min_sources
        )

        return cls(
            high_confidence_posterior=high_conf_analysis.inference_result,
            low_confidence_posterior=low_conf_analysis.inference_result,
            abstention_threshold=abstention_threshold,
        )
