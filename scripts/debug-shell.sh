#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# debug-shell.sh — Exec into the app container for live debugging.
#
# Usage:
#   bash scripts/debug-shell.sh [command...]
#   bash scripts/debug-shell.sh python -m pytest tests/test_rbac_redteam.py -vv -s
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"

# Load .env if present
ENV_FILE="${DOCKER_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

CONTAINER="tax-rag-app"

# Check if the container is running
if ! docker inspect "${CONTAINER}" > /dev/null 2>&1; then
  echo "[debug-shell] Container '${CONTAINER}' not found. Starting stack first..."
  docker compose -f "${DOCKER_DIR}/docker-compose.yml" \
                 -f "${DOCKER_DIR}/compose.override.yml" \
    up -d opensearch redis-stack jaeger app
fi

STATUS=$(docker inspect "${CONTAINER}" --format '{{.State.Status}}' 2>/dev/null || echo "not_found")
if [[ "${STATUS}" != "running" ]]; then
  echo "[debug-shell] Container is '${STATUS}', not 'running'."
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "[debug-shell] Dropping into bash in ${CONTAINER}..."
  docker exec -it \
    -e PYTHONPATH=/app \
    "${CONTAINER}" \
    bash
else
  echo "[debug-shell] Running: $* in ${CONTAINER}"
  docker exec -it \
    -e PYTHONPATH=/app \
    "${CONTAINER}" \
    "$@"
fi
