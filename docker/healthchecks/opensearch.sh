#!/usr/bin/env sh
# Healthcheck for OpenSearch — waits for yellow cluster status.
# The /_cluster/health?wait_for_status=yellow endpoint blocks until healthy
# or the timeout expires. This handles the demo-cert first-boot race condition
# (MASTER-PLAN §G Risk 1).
set -e

OPENSEARCH_URL="${OPENSEARCH_URL:-https://localhost:9200}"
OPENSEARCH_USER="${OPENSEARCH_USER:-admin}"
OPENSEARCH_PASS="${OPENSEARCH_PASS:-admin}"

curl -fsk \
  -u "${OPENSEARCH_USER}:${OPENSEARCH_PASS}" \
  "${OPENSEARCH_URL}/_cluster/health?wait_for_status=yellow&timeout=60s" \
  | grep -q '"status":"yellow"\|"status":"green"'
