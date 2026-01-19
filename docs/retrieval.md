
# Recuperación (Retrieval)

## Propósito

El módulo de retrieval se encarga de recuperar fragments de texto relevantes desde el vector store dada una consulta textual. Está pensado para obtener contexto que luego será pasado al agente RAG para generación de respuestas fundamentadas.

## Responsabilidades

- Ejecutar búsquedas por similitud semántica contra el `VectorRepository`.
- Filtrar resultados por puntuación mínima y por número de fuentes distintas (opcional).
- Devolver los fragmentos y la lista de fuentes asociadas para permitir citación y trazabilidad.

## ¿Cómo encaja en el pipeline RAG?

EMBEDDINGS → VECTOR STORE
	↓
RETRIEVER (`Retriever`)  ← consulta del usuario
	↓
DOCUMENTOS RECUPERADOS → PROMPT → LLM

## Componentes principales

- `src/retrieval/retriever.py` — `Retriever`:
	- Constructor:
		- `repository: VectorRepository` — adaptador al vector store (p. ej. `ChromaRepository`).
		- `k: int` — número máximo de vecinos a recuperar.
		- `min_score: float` — umbral mínimo de similitud para aceptar un resultado.
		- `require_multiple_sources: bool` — si True, fuerza que los resultados provengan de al menos dos fuentes distintas.
	- Método principal:
		- `retrieve(query: str) -> Tuple[List[str], List[str]]` — devuelve `(documents, sources)` donde `documents` son los objetos recuperados que cumplen `score >= min_score`, y `sources` es la lista única de nombres de fichero origen.


## Configuración

- Ajusta `k` y `min_score` según la colección y la calidad de los embeddings. Valores por defecto no están fijados en el módulo; se definen al instanciar `Retriever` desde el agente o la configuración de ejecución.
- `require_multiple_sources` es una medida de seguridad para evitar respuestas basadas en una única fuente; actívala si necesitas mayor robustez en citación.

## Limitaciones

- El filtrado actual es sencillo (score + conteo de fuentes). No hay lógica de deduplicación de contenido ni del ranking por relevancia contextual.
- Comportamiento dependiente de la API del `VectorRepository`; cualquier cambio en la forma de retorno de `similarity_search` requiere actualizar `Retriever`.
