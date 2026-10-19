#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export DAGSTER_HOME="${SCRIPT_DIR}/.dagster_home"
mkdir -p "${DAGSTER_HOME}" "${SCRIPT_DIR}/.cache/borealis"

# Clean up any leftover ephemeral .tmp_dagster_home_* directories
rm -rf "${SCRIPT_DIR}"/.tmp_dagster_home_*

echo "==> [1/3] Ensuring PostgreSQL + pgvector container (borealis-specs-db) is up and initialized..."
docker compose up -d postgres
docker start borealis-specs-db >/dev/null 2>&1 || true

echo "==> [2/3] Materializing & caching pipeline steps (AST -> dLLM Graph -> Embeddings -> Postgres -> Hybrid CTE)..."
uv run python -m borealis_dagster.run_pipeline

if [ "${NO_SERVER:-0}" != "1" ]; then
    echo "==> [3/3] Launching Dagster UI with persistent DAGSTER_HOME=${DAGSTER_HOME}..."
    exec uv run dagster dev -m borealis_dagster
fi