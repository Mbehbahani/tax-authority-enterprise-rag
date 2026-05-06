#!/usr/bin/env sh
# Healthcheck for Jaeger all-in-one — admin endpoint must return 200.
set -e

wget -qO- http://localhost:14269/ > /dev/null 2>&1
