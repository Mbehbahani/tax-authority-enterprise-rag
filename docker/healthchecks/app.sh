#!/usr/bin/env sh
# Healthcheck for app container — verifies AWS credentials are accessible.
# Startup probe: pings sts:GetCallerIdentity and logs the ARN.
set -e

python - <<'EOF'
import boto3
import os
import sys

region = os.environ.get("AWS_REGION", "us-east-1")
try:
    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    arn = identity.get("Arn", "UNKNOWN")
    print(f"[STARTUP PROBE] AWS identity: {arn}", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"[STARTUP PROBE FAILED] Cannot get AWS identity: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
EOF
