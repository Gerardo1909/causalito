"""
Módulo que define el modelo bayesiano beta-binomial encargado
de estimar la probabilidad de que una respuesta del agente sea
correcta.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass
class BayesianInferenceResult:
    """Resultado de la inferencia bayesiana."""

    # Parámetros posteriores
    alpha_posterior: float
    beta_posterior: float

    # Estadísticas
    posterior_mean: float  # E[θ | data]
    posterior_std: float  # σ[θ | data]
    credible_interval_low: float  # 5% quantile
    credible_interval_high: float  # 95% quantile

    # Metadatos
    n_successes: int
    n_failures: int
    n_total: int


class BetaBinomialModel:
    """
    Modelo Beta-Binomial simple para estimación de probabilidad.

    Usa prior no informativo Beta(1, 1) para modelar:
        θ = P(respuesta_correcta | features)

    Dado un dataset de etiquetas (1=correct, 0=incorrect),
    estima la distribución posterior usando conjugación beta.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Args:
            alpha_prior: Parámetro alpha del prior Beta
            beta_prior: Parámetro beta del prior Beta
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def fit(self, labels: np.ndarray) -> BayesianInferenceResult:
        """
        Entrena el modelo en un dataset de etiquetas binarias.

        Args:
            labels: Array binario [0, 1, 1, 0, ...] indicando correcta/incorrecta

        Returns:
            Distribución posterior
        """
        n_successes = np.sum(labels == 1)
        n_failures = np.sum(labels == 0)

        # Actualizar parámetros (conjugación)
        alpha_post = self.alpha_prior + n_successes
        beta_post = self.beta_prior + n_failures

        # Calcular estadísticas
        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_variance = (alpha_post * beta_post) / (
            (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
        )
        posterior_std = np.sqrt(posterior_variance)

        # Intervalo creíble 90%
        ci_low = beta.ppf(0.05, alpha_post, beta_post)
        ci_high = beta.ppf(0.95, alpha_post, beta_post)

        return BayesianInferenceResult(
            alpha_posterior=alpha_post,
            beta_posterior=beta_post,
            posterior_mean=float(posterior_mean),
            posterior_std=float(posterior_std),
            credible_interval_low=float(ci_low),
            credible_interval_high=float(ci_high),
            n_successes=int(n_successes),
            n_failures=int(n_failures),
            n_total=int(len(labels)),
        )
