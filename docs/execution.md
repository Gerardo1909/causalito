
# Ejecución

## Propósito

Actualmente el agente se ejecuta mediante un comando de ejecución de python en la terminal, en interaciones futuras se evaluará la inclusión
de una mejor UI. Por otro lado hay una ejecución que se hace aparte del flujo principal, la cual lleva a cabo el proceso de ingesta de información, es el paso previo que prepara la información que el agente consume. A continuación se explica la ejecución de los componentes 
principales del proyecto. 

## Requisitos

- Python 3.13+ instalado.
- Dependencias listadas en `requirements.txt` o gestionadas vía `pyproject.toml`.

## Entorno virtual (`venv`)

1. Crear y activar entorno virtual (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

O instalar el paquete local en editable mode (opcional):

```bash
pip install -e .
```

## Ejecutar la ingestión e indexación (paso previo al modelo)

- Ejecutar solo la fase de ingestión (limpieza y persistencia `clean_documents.jsonl`):

```bash
python src/ingestion/run.py
```

Esto normalmente deberá ejecutarse una sola vez, que es cuando se agregaron inicialmente las fuentes o 
en su defecto se puede volver a ejecutar si se agrega una fuente nueva. Se espera formato PDF para las fuentes.

**Nota**: los directorios de entrada (`data/raw/`) y persistencia de Chroma están configurados en `src/config/settings.py`.

## Ejecutar el agente interactivo (terminal)

- Construir e iniciar el agente desde `src/main.py`:

```bash
python src/main.py
```

- Uso:
  - Escribe una pregunta y presiona Enter.
  - Escribe `exit` o `salir` para terminar la sesión.
  - El agente imprime la respuesta y la lista de fuentes (si las hay).

Ejemplo de interacción:

```
Causalito listo. Escribe tu pregunta o 'exit' para salir.
>> ¿Qué es inferencia causal bayesiana?
[Respuesta]
...texto generado...

Fuentes: ['paper1.pdf', 'book_chapter.pdf']
```

## Consideraciones y troubleshooting

- Si Chroma no encuentra el directorio de persistencia, confirma `CHROMA_PERSIST_DIR` en `src/config/settings.py` y permisos de escritura.
- Si faltan dependencias, instala desde `requirements.txt` o revisa `pyproject.toml`.
- Para problemas de rendimiento con embeddings, considera usar una máquina con más memoria o reducir el `batch` de indexación en `ingestion.run.index_documents()`.
- Si el agente responde "No hay suficiente información...", indexa más documentos o ajusta los parámetros del `Retriever` (`k`, `min_score`).
