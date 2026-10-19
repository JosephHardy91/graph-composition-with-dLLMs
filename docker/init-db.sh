#!/usr/bin/env bash
set -euo pipefail

SCHEMA_FILE="/opt/sql/schema_and_cte.sql"

echo "==> Initializing PostgreSQL schema and pgvector extension from ${SCHEMA_FILE}..."

# 1. Extract and execute the DDL statements (everything before the standalone WITH RECURSIVE query)
DDL_SQL="$(sed '/^WITH RECURSIVE/,$d' "${SCHEMA_FILE}")"

# 2. Extract the standalone parameterized CTE query (from WITH RECURSIVE to the end of file)
CTE_SQL="$(sed -n '/^WITH RECURSIVE/,$p' "${SCHEMA_FILE}")"

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<-EOSQL
    ${DDL_SQL}

    -- Register the parameterized CTE from schema_and_cte.sql as a callable SQL function
    -- (\$1 = query_embedding, \$2 = max_hops)
    CREATE OR REPLACE FUNCTION resolve_spec_context(
        query_embedding vector(1536),
        max_hops INT DEFAULT 4
    )
    RETURNS TABLE (
        node_id TEXT,
        doc_id TEXT,
        heading TEXT,
        precedence_tier INT,
        hop_depth INT,
        edge_via TEXT,
        condition_predicate TEXT,
        traversal_path TEXT[],
        content TEXT
    )
    LANGUAGE sql
    STABLE
    AS \$func\$
    ${CTE_SQL}
    \$func\$;
EOSQL

echo "==> Database '${POSTGRES_DB}' initialized with spec_nodes, spec_edges, HNSW index, and resolve_spec_context()."
