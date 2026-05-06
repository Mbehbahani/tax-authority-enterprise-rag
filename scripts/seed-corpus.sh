#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# seed-corpus.sh — Synthesize ~250 tax-law fixtures and push into OpenSearch.
#
# Creates index 'tax-docs-test' with Lucene k-NN HNSW settings per MASTER-PLAN §C.
# Uses a 1024-dim random vector for each chunk (real embeddings require Bedrock).
# In CI with real Bedrock: replace random vectors with Cohere embed calls.
#
# Role-tagged metadata enables RBAC tests:
#   - classification = public | internal | fiod
#   - doc_type = legislation | case_law | policy | elearning
#   - tax_year = 2021 | 2022 | 2023 | 2024
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

OPENSEARCH_URL="${OPENSEARCH_URL:-https://opensearch:9200}"
OPENSEARCH_USER="${OPENSEARCH_USER:-admin}"
OPENSEARCH_PASS="${OPENSEARCH_PASS:-admin}"
INDEX="tax-docs-test"

echo "[seed-corpus] Seeding index '${INDEX}' into ${OPENSEARCH_URL}..."

# ─── Run the Python seeder inside the app container if available, else locally ─
if docker inspect tax-rag-app > /dev/null 2>&1; then
  echo "[seed-corpus] Running seeder inside tax-rag-app container..."
  docker exec \
    -e OPENSEARCH_URL="${OPENSEARCH_URL}" \
    -e OPENSEARCH_USER="${OPENSEARCH_USER}" \
    -e OPENSEARCH_PASS="${OPENSEARCH_PASS}" \
    tax-rag-app \
    python /app/tests/seed_opensearch.py
else
  echo "[seed-corpus] Container not running; attempting local seeder..."
  OPENSEARCH_URL="${OPENSEARCH_URL}" \
  OPENSEARCH_USER="${OPENSEARCH_USER}" \
  OPENSEARCH_PASS="${OPENSEARCH_PASS}" \
  python3 "${REPO_ROOT}/tests/seed_opensearch.py" || {
    echo "[seed-corpus] WARNING: local seeder failed. Tests will use MOCK_CORPUS only."
  }
fi

echo "[seed-corpus] Done."
