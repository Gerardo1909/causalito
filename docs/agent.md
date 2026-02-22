
# Agente RAG

## Propósito

El agente coordina recuperación y generación para responder consultas del usuario sobre inferencia causal. Su objetivo es producir respuestas fundamentadas en fragmentos recuperados del corpus, manteniendo trazabilidad hacia las fuentes. Adicionalmente, integra un motor de confianza bayesiano que estima la probabilidad de que la respuesta sea correcta y decide si responder o abstenerse.

## Responsabilidades

- Orquestar la búsqueda de contexto relevante (`Retriever`).
- Construir prompts a partir de los documentos recuperados (`PromptBuilder`).
- Invocar el LLM para generar la respuesta final (`LLM`).
- Estimar la confianza de la respuesta mediante el `ConfidenceEngine` y decidir si responder o abstenerse.
- Devolver el texto generado junto con las fuentes asociadas y el resultado de confianza.

## ¿Cómo encaja en el pipeline RAG?

USER QUERY
  ↓
RETRIEVER (recupera documentos relevantes + scores de similitud)
  ↓
FEATURE EXTRACTOR (extrae métricas del retrieval)
  ↓
CONFIDENCE ENGINE (estima P(correcta) y decide answer/abstain)
  ↓
  ├─ ABSTAIN → retorna reasoning sin invocar LLM
  └─ ANSWER  → PROMPT BUILDER → LLM → RESPUESTA + FUENTES + CONFIANZA

## Componentes principales

- `RAGAgent` (`src/agent/rag_agent.py`): clase principal que recibe `Retriever`, `LLM`, `PromptBuilder` y `ConfidenceEngine` en su constructor y expone `answer(question) -> (text, sources, confidence_result)`.
- `PromptBuilder` (`src/agent/prompt_builder.py`): genera el prompt instructivo que obliga al LLM a utilizar únicamente el contexto recuperado y a responder con un estilo académico y riguroso.
- `LLM` (abstracción en `src/llm/base.py`, implementación `qroq_llm.py`): componente responsable de la generación textual.
- `Retriever` (`src/retrieval/retriever.py`): devuelve los documentos, fuentes y scores de similitud necesarios para construir el prompt y alimentar el motor de confianza.
- `ConfidenceEngine` (`src/confidence/confidence_engine.py`): motor bayesiano que clasifica las features del retrieval y retorna una decisión `ANSWER` o `ABSTAIN` con intervalo creíble. Ver [docs/confidence.md](confidence.md) para detalle completo.

## Configuración

- Instanciación: construir `RAGAgent` con instancias concretas de `Retriever`, `LLM`, `PromptBuilder` y opcionalmente `ConfidenceEngine`.
- Parámetros relevantes provienen de cada componente:
  - `Retriever`: `k`, `min_score`, `require_multiple_sources`.
  - `LLM`: modelo y parámetros de generación (temperatura, top_k, etc.) según la implementación.
  - `PromptBuilder`: plantilla y reglas de estilo (actualmente integradas en `PromptBuilder.build`).
  - `ConfidenceEngine`: `abstention_threshold`, posteriors pre-entrenadas, umbrales de clasificación high/low.

Ejemplo de creación:

```python
from agent.rag_agent import RAGAgent
from agent.prompt_builder import PromptBuilder
from confidence.confidence_engine import ConfidenceEngine

engine = ConfidenceEngine.from_csv("data/eval_log.csv", abstention_threshold=0.8)

agent = RAGAgent(
    retriever=repo_retriever,
    llm=llm_impl,
    prompt_builder=PromptBuilder(),
    confidence_engine=engine,
)
text, sources, confidence = agent.answer("¿Qué es inferencia causal bayesiana?")
```

Si no se dispone de datos etiquetados, se puede pasar `confidence_engine=None` para desactivar la estimación de confianza. En ese caso, `confidence_result` será `None` en la respuesta.

## Limitaciones

- El agente confía en la calidad del retriever: si los embeddings o el ranking son pobres, la respuesta puede carecer de cobertura.
- El control de veracidad depende de la presencia del contenido en los documentos; no hay verificación externa adicional.
- El prompt obliga al modelo a no "inventar", pero el LLM puede todavía generar salidas indeseadas; se recomienda evaluación humana para respuestas críticas.
- La calidad de la estimación de confianza depende del volumen y representatividad de los datos etiquetados disponibles.
- No hay actualmente manejo de conversaciones multi-turno ni historial de contexto.