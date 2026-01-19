
# Agente RAG

## Propósito

El agente coordina recuperación y generación para responder consultas del usuario sobre inferencia causal. Su objetivo es producir respuestas fundamentadas en fragmentos recuperados del corpus, manteniendo trazabilidad hacia las fuentes.

## Responsabilidades

- Orquestar la búsqueda de contexto relevante (`Retriever`).
- Construir prompts a partir de los documentos recuperados (`PromptBuilder`).
- Invocar el LLM para generar la respuesta final (`LLM`).
- Devolver el texto generado junto con las fuentes asociadas para permitir citación.

## ¿Cómo encaja en el pipeline RAG?

USER QUERY
  ↓
RETRIEVER (recupera documentos relevantes)
  ↓
PROMPT BUILDER (construye prompt con contexto)
  ↓
LLM (genera respuesta basada en el prompt)
  ↓
RESPUESTA + FUENTES

## Componentes principales

- `RAGAgent` (`src/agent/rag_agent.py`): clase principal que recibe `Retriever`, `LLM` y `PromptBuilder` en su constructor y expone `answer(question) -> (text, sources)`.
- `PromptBuilder` (`src/agent/prompt_builder.py`): genera el prompt instructivo que obliga al LLM a utilizar únicamente el contexto recuperado y a responder con un estilo académico y riguroso.
- `LLM` (abstracción en `src/llm/base.py`, implementación `qroq_llm.py`): componente responsable de la generación textual.
- `Retriever` (`src/retrieval/retriever.py`): devuelve los documentos y metadatos necesarios para construir el prompt.

## Configuración

- Instanciación: construir `RAGAgent` con instancias concretas de `Retriever`, `LLM` y `PromptBuilder`.
- Parámetros relevantes provienen de cada componente:
  - `Retriever`: `k`, `min_score`, `require_multiple_sources`.
  - `LLM`: modelo y parámetros de generación (temperatura, top_k, etc.) según la implementación.
  - `PromptBuilder`: plantilla y reglas de estilo (actualmente integradas en `PromptBuilder.build`).

Ejemplo de creación:

```python
from agent.rag_agent import RAGAgent
from agent.prompt_builder import PromptBuilder

agent = RAGAgent(retriever=repo_retriever, llm=llm_impl, prompt_builder=PromptBuilder())
text, sources = agent.answer('¿Qué es inferencia causal bayesiana?')
```

## Limitaciones

- El agente confía en la calidad del retriever: si los embeddings o el ranking son pobres, la respuesta puede carecer de cobertura.
- El control de veracidad depende de la presencia del contenido en los documentos; no hay verificación externa adicional.
- El prompt obliga al modelo a no “inventar”, pero el LLM puede todavía generar salidas indeseadas; se recomienda evaluación humana para respuestas críticas.
- No hay actualmente manejo de conversaciones multi-turno ni historial de contexto.
