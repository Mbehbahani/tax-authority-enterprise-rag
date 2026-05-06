#!/usr/bin/env sh
# Healthcheck for Redis Stack — PING must return PONG.
set -e

REDIS_PORT="${REDIS_PORT:-6379}"

redis-cli -p "${REDIS_PORT}" ping | grep -q "PONG"
