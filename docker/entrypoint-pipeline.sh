#!/usr/bin/env bash
set -euo pipefail

export DAGSTER_HOME="${DAGSTER_HOME:-/app/.dagster_home}"
mkdir -p "${DAGSTER_HOME}" /app/.cache/borealis

echo "==> [Step 1-5] Running & caching Borealis Graph Compiler pipeline (Document AST -> DB Init -> Asset Materialization -> Hybrid CTE)..."
uv run --no-sync python -m borealis_dagster.run_pipeline

if [ "${RUN_DAGSTER_SERVER:-true}" = "true" ]; then
    echo "==> Starting Dagster UI on http://0.0.0.0:3000 (DAGSTER_HOME=${DAGSTER_HOME})..."
    exec uv run --no-sync dagster dev -h 0.0.0.0 -p 3000 -m borealis_dagster
fi
