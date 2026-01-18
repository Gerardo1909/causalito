"""
Directorio que contiene la configuración de rutas para el proyecto.
"""

from pathlib import Path

# Directorio raíz del proyecto (ajusta si es necesario)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorio de datos
DATA_DIR = BASE_DIR.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

# Persistencia de Chroma
CHROMA_PERSIST_DIR = INDEXES_DIR / "chroma"

# Crear los directorios si no existen
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEXES_DIR, CHROMA_PERSIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)
