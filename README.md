# Causalito - Agente RAG orientado a inferencia causal 

## Overview

**Causalito** es un agente inteligente diseñado para **asistir a estudiantes, investigadores y profesionales** en el campo de la inferencia bayesiana y causal. Su objetivo es proporcionar respuestas fundamentadas y precisas a consultas sobre inferencia causal, apoyándose en bibliografía académica de referencia en formato PDF.

A través de la integración de tecnologías de recuperación aumentada por generación (RAG), Causalito permite acceder de manera eficiente y transparente al conocimiento contenido en textos clave, facilitando el aprendizaje, la consulta y la aplicación rigurosa de conceptos complejos en estadística y causalidad.

Este proyecto **busca ser una herramienta confiable** y accesible para quienes requieren información verificada y contextualizada en el ámbito de la inferencia bayesiana causal.

![causalito_overview](resources/causalito_overview.png)

## Beneficios clave

- **Respuestas fundamentadas:** el agente combina recuperación de texto con generación (RAG) para proporcionar respuestas ancladas en documentos académicos.
- **Trazabilidad:** cada fragmento recuperado se asocia a su fuente, facilitando la verificación y citación de las respuestas.
- **Reproducibilidad:** indexado persistente con Chroma y IDs estables permiten reproducir búsquedas y evitar reindexados innecesarios.
- **Extensibilidad:** arquitectura por módulos (loaders, chunkers, embedders, vectorstore, retriever, llm, agent) facilita sustituir componentes o adaptar el sistema a nuevas colecciones y modelos.

## A quién ayuda

- Estudiantes que necesitan explicaciones pedagógicas y referencias bibliográficas sobre inferencia bayesiana y causalidad.
- Investigadores que quieren recuperar rápidamente fragmentos relevantes de bibliografía y obtener resúmenes contextualizados.
- Ingenieros y desarrolladores que desean una base reproducible para construir agentes RAG orientados a dominios científicos.

## Documentación detallada

A continuación se presenta documentación detallada del proyecto presente en el directorio `docs/` y su propósito:

- [docs/execution.md](docs/execution.md): Instrucciones de ejecución (configuración de entorno virtual, ejecución de agente interactivo).
- [docs/ingestion.md](docs/ingestion.md): Descripción del pipeline de ingestión — loaders, `TextCleaner` y persistencia de `clean_documents.jsonl`.
- [docs/indexing.md](docs/indexing.md): Chunking, estrategia de IDs (`ChunkIdStrategy`) y el `IndexingService` que orquesta la creación de embeddings e inserción en el vectorstore.
- [docs/retrieval.md](docs/retrieval.md): Lógica de recuperación desde el vector store, filtrado por `min_score`, control de `k` y manejo de fuentes.
- [docs/vectorstore.md](docs/vectorstore.md): Interfaz `VectorRepository` e implementación `ChromaRepository` — persistencia, gestión de IDs y operaciones CRUD sobre vectores.
- [docs/agent.md](docs/agent.md): Funcionamiento del `RAGAgent`, diseño de prompts y cómo se coordinan retrieval + LLM para generar respuestas trazables.
- [docs/testing.md](docs/testing.md): Estrategia de pruebas, ubicación de tests unitarios/integación y cómo ejecutar la suite (`pytest`).


## Ejemplo interactivo

Abajo se muestra un ejemplo de interacción típica en terminal: se hace una pregunta y el agente responde utilizando únicamente el contexto indexado:

![Ejemplo de interacción del agente](resources/pregunta_principal.png)

*Descripción:* la captura muestra la pregunta en el prompt, la respuesta generada (resumen con énfasis en conceptos clave) y la lista de fuentes citadas.

## Limitaciones

- La calidad de las respuestas depende directamente de la cobertura y la calidad de los documentos indexados y del modelo de embeddings/LLM elegido.
- Las heurísticas de limpieza (`TextCleaner`) pueden necesitar ajustes para colecciones con formatos muy heterogéneos.
- No existe verificación externa automática de veracidad — el agente se limita a sintetizar lo presente en los documentos. Recomendamos validación humana para uso crítico.

## Posibles extensiones y mejoras a futuro

- Soporte para múltiples modelos de embeddings y selección automática según dominio.
- Reranking por segundo paso (BM25 o LLM-based) para mejorar precisión en las primeras posiciones.
- Deduplicación semántica avanzada y políticas de retención/compactación del índice.
- Soporte multi-turno y manejo de historial conversacional en el agente.
- Integración de trazabilidad extendida (hashes de documentos, control de versiones del índice).

---

**Stack:** Python 3.13 · Langchain · ChromaDB · Docker 

**Autor:** Gerardo Toboso · [gerardotoboso1909@gmail.com](mailto:gerardotoboso1909@gmail.com)

**Licencia:** MIT