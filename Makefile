# Variables
COMPOSE = docker compose -f docker/docker-compose.yml
IMAGE_NAME = causalito

.PHONY: help build-prod build-dev run stop dev shell-prod logs clean

help:
	@echo "Uso: make <comando>"
	@echo ""
	@echo "CONSTRUCCIÓN (BUILD):"
	@echo "  build-prod   Construye la imagen de PRODUCCIÓN (optimizada)"
	@echo "  build-dev    Construye la imagen de DESARROLLO (con tests y torch)"
	@echo ""
	@echo "EJECUCIÓN:"
	@echo "  run          Levanta la app de producción (background)"
	@echo "  dev          Entra al entorno de desarrollo (bash + hot-reload)"
	@echo "  stop         Detiene todos los servicios"
	@echo ""
	@echo "DEBUG & UTILIDADES:"
	@echo "  shell-prod   Entra a la imagen de producción para inspección"
	@echo "  logs         Ver logs en tiempo real"
	@echo "  clean        Remueve contenedores e imágenes del proyecto"

# Construcción de Producción
build-prod:
	@echo "Construyendo imagen de PRODUCCIÓN..."
	$(COMPOSE) build prod

# Construcción de Desarrollo
build-dev:
	@echo "Construyendo imagen de DESARROLLO..."
	$(COMPOSE) build dev

# Ejecución
run:
	$(COMPOSE) up prod -d
	@echo "✓ App disponible en http://localhost:8501"

stop:
	$(COMPOSE) down

# Desarrollo interactivo
dev:
	$(COMPOSE) run --rm -it dev /bin/bash

shell-prod:
	$(COMPOSE) run --rm -it prod /bin/bash

logs:
	$(COMPOSE) logs -f

clean:
	$(COMPOSE) down -v --remove-orphans
	docker rmi $(IMAGE_NAME):prod $(IMAGE_NAME):dev 2>/dev/null || true
