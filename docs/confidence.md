
# Motor de confianza bayesiano

## Propósito

El motor de confianza estima la probabilidad de que una respuesta del agente RAG sea correcta antes de entregarla al usuario. Cuando la confianza es insuficiente, el agente se abstiene de responder en lugar de generar una respuesta potencialmente incorrecta. Esta capa opera entre el retrieval y la generación, actuando como un mecanismo de control de calidad basado en inferencia bayesiana.

## Responsabilidades

- Clasificar cada consulta en un grupo de confianza (high/low) a partir de las métricas del retrieval.
- Estimar P(respuesta correcta | grupo) usando distribuciones posteriores pre-entrenadas.
- Decidir si el agente debe responder (`ANSWER`) o abstenerse (`ABSTAIN`) según un umbral configurable.
- Proveer intervalos creíbles al 90% que cuantifican la incertidumbre de la estimación.

## ¿Cómo encaja en el pipeline RAG?

RETRIEVER (docs + scores)
  ↓
FEATURE EXTRACTOR (mean_similarity, max_similarity, n_sources)
  ↓
CONFIDENCE ENGINE
  ├─ Clasifica en grupo HIGH o LOW
  ├─ Consulta posterior Beta pre-entrenada del grupo
  ├─ Evalúa P(correcta) vs. abstention_threshold
  ↓
  ├─ ANSWER  → continúa a PROMPT BUILDER → LLM
  └─ ABSTAIN → retorna reasoning, no invoca LLM

## Fundamento estadístico

### Modelo Beta-Binomial

El motor usa un modelo conjugado Beta-Binomial para estimar la probabilidad de respuesta correcta. La elección se justifica por:

- **Conjugación exacta:** la distribución Beta es el prior conjugado de la verosimilitud Binomial. Esto permite actualización analítica directa sin necesidad de MCMC ni aproximaciones.
- **Prior no informativo:** Beta(1, 1) equivale a una distribución uniforme en [0, 1]. Máxima entropía; deja que los datos determinen la posterior.
- **Comportamiento con pocos datos:** con pocas observaciones, la posterior se mantiene conservadora (cercana a 0.5). Con muchas observaciones, converge a la frecuencia observada. Esto es exactamente el comportamiento deseado para un sistema que debe ser cauteloso ante la falta de evidencia.

### Actualización de la posterior

Dado un prior Beta(α₀, β₀) y un dataset con `s` respuestas correctas y `f` incorrectas:

    Posterior = Beta(α₀ + s, β₀ + f)

    E[θ | data] = (α₀ + s) / (α₀ + s + β₀ + f)

El intervalo creíble al 90% se obtiene de los cuantiles 5% y 95% de la distribución posterior.

### Clasificación en grupos

El motor divide las consultas en dos grupos usando heurísticas sobre las features del retrieval:

- **HIGH confidence:** `mean_similarity >= threshold` AND `n_sources >= min_sources`. Indica que el retriever encontró múltiples fuentes con alta similitud semántica.
- **LOW confidence:** cualquier consulta que no cumpla ambas condiciones.

Cada grupo tiene su propia distribución posterior, entrenada independientemente sobre datos etiquetados.

## Componentes principales

- `BetaBinomialModel` (`src/confidence/bayesian_model.py`): implementa el modelo conjugado. Recibe un array de etiquetas binarias y retorna `BayesianInferenceResult` con parámetros posteriores, media, desviación estándar e intervalo creíble.
- `ConfidenceEngine` (`src/confidence/confidence_engine.py`): interfaz de inferencia. Expone `.predict(mean_similarity, max_similarity, n_sources)` que retorna `ConfidenceResult` con decisión `ANSWER`/`ABSTAIN`, probabilidad estimada e intervalo creíble.
- `DataAnalyzer` (`src/confidence/data_analyzer.py`): analiza un CSV etiquetado, divide en grupos high/low y entrena un modelo Beta-Binomial por grupo. Produce `ConfidenceGroupAnalysis` con estadísticas descriptivas y resultado bayesiano.
- `RetrievalFeatureExtractor` (`src/confidence/feature_extractor.py`): extrae features del retrieval (similitudes, número de fuentes, longitud de contexto).
- `FeatureLogger` (`src/confidence/feature_extractor.py`): persiste features en CSV para etiquetado posterior y re-entrenamiento.
- `analyze_confidence.py` (`src/confidence/analyze_confidence.py`): script CLI para ejecutar el análisis offline sobre datos etiquetados.

## Flujo de entrenamiento y uso

### 1. Recolección de datos

El `FeatureLogger` registra automáticamente las features de cada consulta en `data/eval_log.csv`. La columna `label` queda vacía para etiquetado manual posterior (1=correcta, 0=incorrecta).

### 2. Entrenamiento offline

```bash
python -m confidence.analyze_confidence --csv-path data/eval_log.csv
```

Este script lee el CSV etiquetado, divide en grupos y muestra las posteriors resultantes con estadísticas comparativas.

### 3. Carga en producción

El `ConfidenceEngine` se inicializa con el factory method `from_csv`:

```python
from confidence.confidence_engine import ConfidenceEngine

engine = ConfidenceEngine.from_csv(
    csv_path="data/eval_log.csv",
    high_conf_threshold=0.75,
    min_sources=2,
    abstention_threshold=0.8,
)
```

Si no hay datos disponibles, el engine usa posteriors neutrales Beta(1, 1), lo que produce P(correcta) = 0.5 y el agente se abstendrá por defecto (0.5 < threshold).

### 4. Inferencia

```python
result = engine.predict(
    mean_similarity=0.82,
    max_similarity=0.91,
    n_sources=3,
)

result.decision        # Decision.ANSWER
result.prob_correct     # 0.87
result.credible_interval_low   # 0.78
result.credible_interval_high  # 0.94
result.reasoning       # "Alta confianza (87.0%). Grupo: high. IC 90%: [0.78, 0.94]"
```

## Configuración

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `alpha_prior` | 1.0 | Parámetro alpha del prior Beta. |
| `beta_prior` | 1.0 | Parámetro beta del prior Beta. |
| `abstention_threshold` | 0.75 | Si P(correcta) < este valor, el agente se abstiene. |
| `high_conf_threshold` | 0.75 | Umbral de `mean_similarity` para clasificar como high-conf. |
| `min_sources` | 2 | Número mínimo de fuentes para clasificar como high-conf. |

## Limitaciones

- La clasificación binaria en dos grupos (high/low) es gruesa. Una extensión natural sería usar regresión logística bayesiana con features continuas como covariables.
- Requiere etiquetado manual de respuestas para calibrar las posteriors. Sin datos etiquetados, el motor opera con posteriors neutrales y se abstiene sistemáticamente.
- No hay mecanismo de re-entrenamiento automático; el análisis se ejecuta offline bajo demanda.
- La calidad de la estimación depende de la representatividad de los datos etiquetados respecto al tráfico real de consultas.