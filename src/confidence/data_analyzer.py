"""
Módulo encargado de
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from confidence.bayesian_model import BayesianInferenceResult, BetaBinomialModel


@dataclass
class ConfidenceGroupAnalysis:
    """
    Análisis de un grupo (high/low confidence).
    """

    group_name: str
    n_examples: int
    confidence_threshold: str

    # Bayesian results
    inference_result: BayesianInferenceResult

    # Estadísticas descriptivas
    accuracy: float
    precision: float

    def summary(self) -> str:
        """Retorna resumen legible."""
        result = self.inference_result
        return f"""
        {self.group_name}
        ────────────────────────
        Ejemplos: {self.n_examples}
        Accuracy: {self.accuracy:.1%}

        P(correcta | {self.confidence_threshold}):
          Media Posterior: {result.posterior_mean:.3f}
          Std: {result.posterior_std:.4f}
          IC 90%: [{result.credible_interval_low:.3f}, {result.credible_interval_high:.3f}]

        Datos: {result.n_successes} correctas, {result.n_failures} incorrectas
        """


class DataAnalyzer:
    """
    Analiza datos etiquetados y produce insights bayesianos.
    """

    def __init__(self, eval_csv_path: Path):
        self.csv_path = eval_csv_path
        self.df = pd.read_csv(eval_csv_path)

        # Filtrar solo filas etiquetadas
        self.df_labeled = self.df[self.df["label"].notna() & (self.df["label"] != "")]

    def analyze_confidence_split(
        self, high_conf_threshold: float = 0.75, min_sources: int = 2
    ) -> tuple[ConfidenceGroupAnalysis, ConfidenceGroupAnalysis]:
        """
        Divide dataset en high/low confidence y analiza cada grupo.

        Args:
            high_conf_threshold: min_similarity >= this value
            min_sources: n_sources >= this value

        Returns:
            (high_confidence_analysis, low_confidence_analysis)
        """

        # Definir grupos
        high_conf = (self.df_labeled["mean_similarity"] >= high_conf_threshold) & (
            self.df_labeled["n_sources"] >= min_sources
        )

        df_high = self.df_labeled[high_conf]
        df_low = self.df_labeled[~high_conf]

        # Entrenar modelo bayesiano para cada grupo
        model = BetaBinomialModel()

        # Procesar labels
        labels_high = (
            df_high["label"].values.astype(int)
            if len(df_high) > 0
            else np.array([], dtype=int)
        )
        labels_low = (
            df_low["label"].values.astype(int)
            if len(df_low) > 0
            else np.array([], dtype=int)
        )

        result_high = (
            model.fit(labels_high) if len(labels_high) > 0 else self._default_result()
        )
        result_low = (
            model.fit(labels_low) if len(labels_low) > 0 else self._default_result()
        )

        # Calcular accuracies
        acc_high = (labels_high == 1).mean() if len(labels_high) > 0 else 0.0
        acc_low = (labels_low == 1).mean() if len(labels_low) > 0 else 0.0

        return (
            ConfidenceGroupAnalysis(
                group_name="HIGH CONFIDENCE",
                n_examples=len(df_high),
                confidence_threshold=f"mean_sim >= {high_conf_threshold}, sources >= {min_sources}",
                inference_result=result_high,
                accuracy=float(acc_high),
                precision=float(acc_high),
            ),
            ConfidenceGroupAnalysis(
                group_name="LOW CONFIDENCE",
                n_examples=len(df_low),
                confidence_threshold=f"mean_sim < {high_conf_threshold} OR sources < {min_sources}",
                inference_result=result_low,
                accuracy=float(acc_low),
                precision=float(acc_low),
            ),
        )

    def _default_result(self) -> BayesianInferenceResult:
        """
        Retorna resultado neutro cuando no hay datos.
        """
        from confidence.bayesian_model import BayesianInferenceResult

        return BayesianInferenceResult(
            alpha_posterior=1.0,
            beta_posterior=1.0,
            posterior_mean=0.5,
            posterior_std=0.0,
            credible_interval_low=0.0,
            credible_interval_high=1.0,
            n_successes=0,
            n_failures=0,
            n_total=0,
        )
