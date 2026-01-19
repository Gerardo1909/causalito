
# Testing

## Propósito

El testing del proyecto tiene como propósito proveer un conjunto reproducible de pruebas, fixtures y procesos que validen la corrección, robustez y ausencia de regresiones en el pipeline RAG.

## Responsabilidades

- Definir y mantener pruebas unitarias y de integración que verifiquen el comportamiento esperado de los componentes (ingestión, chunking, embedders, vectorstore, retriever, agente).
- Proveer utilidades y fixtures reutilizables en `tests/` para facilitar pruebas repetibles.
- Mantener criterios mínimos de calidad para cambios en el código, y ejemplos de pruebas de regresión para funciones críticas.

## ¿Cómo encaja en el pipeline RAG?

Las pruebas cubren cada fase del pipeline:

- Ingestión: validación de `TextCleaner`, loaders y persistencia de `clean_documents.jsonl`.
- Indexación: verificación de `ChunkIdStrategy`, comportamiento de `RecursiveChunker` y `IndexingService` (filtrado de IDs existentes).
- Vector store: pruebas mockeadas de `ChromaRepository` para asegurar llamadas correctas a la API subyacente.
- Retrieval: tests de filtrado (`min_score`), `k` y `require_multiple_sources` en `Retriever`.
- Agent/LLM: tests de integración que comprueban la orquestación (`RAGAgent.answer`) — en unit tests se mockea el LLM para controlar salidas. También se escribieron pruebas que verifican su conocimiento y veracidad de las respuestas.

## Componentes principales

- `tests/unit/` — pruebas unitarias de funcionalidad para `TextCleaner`, `ChunkIdStrategy`, `Retriever`, etc.
- `tests/rag_evaluation/` — pruebas de evaluación end-to-end o semi-integradas que pueden usar modelos reales o mocks controlados.
- `tests/conftest.py` y `tests/helpers.py` — fixtures y utilidades compartidas (p. ej. generar `Document` de prueba, directorios temporales).

## Configuración

- Ejecutar la suite de tests localmente (desde la raíz del proyecto):

```bash
pytest
```

- Entorno recomendado para desarrollo:
  - Python >= 3.13 (coincidente con `pyproject.toml`).
  - Dependencias de `dev` en `pyproject.toml` (pytest, pytest-cov, pytest-html).

## Limitaciones

- Los tests end-to-end que usan modelos reales o una instancia de Chroma pueden ser costosos y no apropiados para ejecución en cada commit; se mantienen separados y etiquetados (p. ej. `@pytest.mark.slow`).
- Algunas pruebas dependen de heurísticas (p. ej. reglas de `TextCleaner`) que pueden necesitar ajustes conforme cambien las fuentes de documentos; los falsos positivos/negativos son posibles.
- No hay cobertura automática de rendimiento o escalado (latencia/throughput) — considerar añadir benchmarks si se planifica manejar colecciones grandes.
