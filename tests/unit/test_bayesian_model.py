import numpy as np
import pytest

from confidence.bayesian_model import BetaBinomialModel


@pytest.mark.unit
class TestBetaBinomialModel:
    """
    Clase que define casos de prueba para el modelo bayesiano beta-binomial.
    """

    def test_fit_should_compute_posterior_parameters(self):
        """Verifica que la posterioridad se calcula correctamente."""
        model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0)

        # 8 correctas, 2 incorrectas
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        result = model.fit(labels)

        # Posterior debe ser Beta(1+8, 1+2) = Beta(9, 3)
        assert result.alpha_posterior == 9.0
        assert result.beta_posterior == 3.0

        # Media debe ser 9/12 = 0.75
        assert abs(result.posterior_mean - 0.75) < 1e-6

        # n_successes, n_failures
        assert result.n_successes == 8
        assert result.n_failures == 2
        assert result.n_total == 10

    def test_credible_interval_should_be_valid(self):
        """Intervalo creíble debe contener la media."""
        model = BetaBinomialModel()
        labels = np.array([1, 1, 1, 0, 0])
        result = model.fit(labels)

        assert (
            result.credible_interval_low
            < result.posterior_mean
            < result.credible_interval_high
        )

        assert (
            result.credible_interval_low
            < result.posterior_mean
            < result.credible_interval_high
        )
