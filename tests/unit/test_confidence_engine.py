import pytest

from confidence.bayesian_model import BayesianInferenceResult
from confidence.confidence_engine import ConfidenceEngine, Decision


@pytest.mark.unit
class TestConfidenceEngine:
    """
    Clase que define casos de prueba para probar el motor
    de confianza.
    """

    def test_predict_should_answer_when_high_confidence(self):
        """Retorna ANSWER si prob > threshold."""
        high_posterior = BayesianInferenceResult(
            alpha_posterior=9,
            beta_posterior=1,
            posterior_mean=0.9,
            posterior_std=0.1,
            credible_interval_low=0.7,
            credible_interval_high=0.95,
            n_successes=8,
            n_failures=2,
            n_total=10,
        )

        engine = ConfidenceEngine(
            high_confidence_posterior=high_posterior, abstention_threshold=0.75
        )

        result = engine.predict(mean_similarity=0.8, max_similarity=0.9, n_sources=3)

        assert result.decision == Decision.ANSWER
        assert result.prob_correct == 0.9

    def test_predict_should_abstain_when_low_confidence(self):
        """Retorna ABSTAIN si prob < threshold."""
        low_posterior = BayesianInferenceResult(
            alpha_posterior=2,
            beta_posterior=8,
            posterior_mean=0.2,
            posterior_std=0.1,
            credible_interval_low=0.05,
            credible_interval_high=0.4,
            n_successes=1,
            n_failures=9,
            n_total=10,
        )

        engine = ConfidenceEngine(
            low_confidence_posterior=low_posterior, abstention_threshold=0.75
        )

        result = engine.predict(mean_similarity=0.3, max_similarity=0.4, n_sources=1)

        assert result.decision == Decision.ABSTAIN

    def test_classify_should_use_high_confidence_group_when_appropriate(self):
        """Usa posterior de high-conf cuando features lo justifican."""
        high_posterior = BayesianInferenceResult(
            alpha_posterior=90,
            beta_posterior=10,
            posterior_mean=0.9,
            posterior_std=0.03,
            credible_interval_low=0.85,
            credible_interval_high=0.94,
            n_successes=89,
            n_failures=11,
            n_total=100,
        )

        low_posterior = BayesianInferenceResult(
            alpha_posterior=5,
            beta_posterior=15,
            posterior_mean=0.25,
            posterior_std=0.1,
            credible_interval_low=0.1,
            credible_interval_high=0.45,
            n_successes=4,
            n_failures=16,
            n_total=20,
        )

        engine = ConfidenceEngine(
            high_confidence_posterior=high_posterior,
            low_confidence_posterior=low_posterior,
            abstention_threshold=0.8,
        )

        result = engine.predict(
            mean_similarity=0.8,
            max_similarity=0.9,
            n_sources=3,
            high_conf_threshold=0.75,
        )

        # Debe usar high-conf posterior (0.9)
        assert result.prob_correct == 0.9
