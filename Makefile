# ============================================================
# PredictIQ — Makefile
# ============================================================
# Shorthand for all common Docker Compose operations.
# Run from the project root (where docker-compose.yml lives).
# ============================================================

.PHONY: help up down restart logs status build clean \
        trigger-pipeline ps api-health shell-api shell-scheduler

# ── Default: show help ─────────────────────────────────────────
help:
	@echo ""
	@echo "PredictIQ — Docker Compose Commands"
	@echo "────────────────────────────────────"
	@echo "  make up                  Start all services (detached)"
	@echo "  make down                Stop all services"
	@echo "  make restart             Full stop + start"
	@echo "  make build               Rebuild all images (no cache)"
	@echo "  make logs                Tail logs for all services"
	@echo "  make logs s=api          Tail logs for one service"
	@echo "  make status              Show health of all containers"
	@echo "  make ps                  docker compose ps"
	@echo "  make api-health          Check API health endpoint"
	@echo "  make trigger-pipeline    Trigger Airflow DAG manually"
	@echo "  make shell-api           Open shell in API container"
	@echo "  make shell-scheduler     Open shell in scheduler container"
	@echo "  make clean               Remove all containers + volumes (DESTRUCTIVE)"
	@echo ""

# ── Start everything ──────────────────────────────────────────
up:
	@echo "Starting PredictIQ services..."
	docker compose up -d
	@echo ""
	@echo "Services:"
	@echo "  FastAPI     →  http://localhost:8000/docs"
	@echo "  Airflow UI  →  http://localhost:8080  (admin / admin)"
	@echo "  Dashboard   →  http://localhost:8501"
	@echo ""
	@echo "Waiting for API health check..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
	    sleep 10; \
	    curl -sf http://localhost:8000/api/v1/health \
	      | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data',{}).get('models_loaded') else 1)" \
	      2>/dev/null && echo "✓ API is healthy" && break; \
	    echo "  Still waiting... ($$i/10)"; \
	done

# ── Stop ──────────────────────────────────────────────────────
down:
	docker compose down

# ── Restart all ───────────────────────────────────────────────
restart: down up

# ── Build images (no cache) ───────────────────────────────────
build:
	docker compose build --no-cache

# ── Logs ──────────────────────────────────────────────────────
logs:
ifdef s
	docker compose logs -f $(s)
else
	docker compose logs -f
endif

# ── Status / ps ───────────────────────────────────────────────
status:
	@echo "── Container Status ──────────────────────────────────────────"
	docker compose ps
	@echo ""
	@echo "── API Health ────────────────────────────────────────────────"
	@curl -sf http://localhost:8000/api/v1/health | python3 -m json.tool 2>/dev/null || echo "API not reachable"

ps:
	docker compose ps

# ── API quick health ──────────────────────────────────────────
api-health:
	@curl -sf http://localhost:8000/api/v1/health | python3 -m json.tool

# ── Trigger the Airflow DAG with simulation date ──────────────
trigger-pipeline:
	docker compose exec scheduler airflow dags trigger predictiq_pipeline \
	    --conf '{"as_of": "2015-08-01 06:00:00"}'
	@echo "Pipeline triggered. Watch at http://localhost:8080"

# ── Interactive shells ────────────────────────────────────────
shell-api:
	docker compose exec -it api bash

shell-scheduler:
	docker compose exec -it scheduler bash

# ── DESTRUCTIVE: remove everything ────────────────────────────
clean:
	@echo "WARNING: This removes all containers and volumes (data will be lost)."
	@read -p "Are you sure? [y/N] " confirm; \
	    [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ] || exit 1
	docker compose down -v --remove-orphans
	docker image rm predictiq-api predictiq-scheduler predictiq-webserver predictiq-dashboard 2>/dev/null || true
	@echo "Clean complete."