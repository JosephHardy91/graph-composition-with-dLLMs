-- ============================================================================
-- PostgreSQL Graph-Compiled Specification Schema (pgvector + Recursive CTEs)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Structural & Semantic Nodes Table
CREATE TABLE spec_nodes (
    node_id TEXT PRIMARY KEY,              -- e.g., 'SEC-4.5' or 'PART:SEAL-FFKM-ULT90'
    doc_id TEXT NOT NULL,                  -- e.g., 'DOC-04-OVERRIDES'
    node_type TEXT NOT NULL                -- 'CLAUSE', 'PART', 'CONDITION', 'STANDARD'
        CHECK (node_type IN ('CLAUSE', 'PART', 'CONDITION', 'STANDARD')),
    heading TEXT,
    content TEXT NOT NULL,
    precedence_tier INT DEFAULT 1,         -- 1=Baseline (Sec 2), 2=Excursion (Sec 4), 3=Safety Interlock (Sec 5)
    embedding vector(1536)                 -- Compatible with text-embedding-3-small or Gemini 1536-d
);

-- 2. Typed Directed Dependency & Override Edges
CREATE TABLE spec_edges (
    edge_id SERIAL PRIMARY KEY,
    source_node_id TEXT NOT NULL REFERENCES spec_nodes(node_id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES spec_nodes(node_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL
        CHECK (relation_type IN (
            'OVERRIDES',        -- Target supersedes Source under specific conditions
            'PROHIBITS',        -- Source invalidates Target
            'REQUIRES',         -- Source mandates Target
            'TRIGGERS',         -- Condition/Clause activates another Clause
            'REFERENCES',       -- Explicit structural cross-reference
            'UPGRADES_TO'       -- Part A must be replaced by Part B
        )),
    condition_predicate TEXT,              -- e.g., 'ambient_temp_c < -30 AND methanol_present == true'
    confidence FLOAT DEFAULT 1.0,
    UNIQUE(source_node_id, target_node_id, relation_type)
);

CREATE INDEX idx_spec_edges_source ON spec_edges(source_node_id);
CREATE INDEX idx_spec_edges_target ON spec_edges(target_node_id);
CREATE INDEX idx_spec_nodes_embedding ON spec_nodes USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 3. Hybrid Vector-Anchor + Multi-Hop Override Resolution Query
-- ============================================================================
-- Given a query embedding ($1) and max hops ($2, default 4), this query:
--   1. Finds the Top-3 semantic anchor clauses (even if they are naive baseline specs).
--   2. Walks BOTH outbound ('TRIGGERS', 'REQUIRES', 'REFERENCES') AND inbound ('OVERRIDES')
--      edges up to N hops without cycles.
--   3. Orders the final compiled context by precedence_tier DESC so Safety Interlocks
--      and Environmental Overrides appear at the top of the prompt with explicit provenance.

WITH RECURSIVE
anchor_nodes AS (
    SELECT
        node_id,
        doc_id,
        heading,
        content,
        precedence_tier,
        1 - (embedding <=> $1::vector) AS similarity
    FROM spec_nodes
    WHERE node_type = 'CLAUSE'
    ORDER BY embedding <=> $1::vector
    LIMIT 3
),
graph_walk AS (
    -- Base case: Anchor nodes discovered via vector search
    SELECT
        a.node_id,
        a.doc_id,
        a.heading,
        a.content,
        a.precedence_tier,
        0 AS hop_depth,
        'VECTOR_ANCHOR'::TEXT AS edge_via,
        NULL::TEXT AS condition_predicate,
        ARRAY[a.node_id] AS traversal_path
    FROM anchor_nodes a

    UNION ALL

    -- Recursive step: Walk forward along dependencies/overrides/triggers
    SELECT
        n.node_id,
        n.doc_id,
        n.heading,
        n.content,
        n.precedence_tier,
        gw.hop_depth + 1 AS hop_depth,
        e.relation_type AS edge_via,
        e.condition_predicate,
        gw.traversal_path || n.node_id
    FROM graph_walk gw
    JOIN spec_edges e
        ON (e.source_node_id = gw.node_id OR (e.target_node_id = gw.node_id AND e.relation_type = 'OVERRIDES'))
    JOIN spec_nodes n
        ON n.node_id = CASE
            WHEN e.source_node_id = gw.node_id THEN e.target_node_id
            ELSE e.source_node_id
        END
    WHERE gw.hop_depth < COALESCE($2, 4)
      AND NOT (n.node_id = ANY(gw.traversal_path)) -- Cycle prevention
)
SELECT DISTINCT ON (node_id)
    node_id,
    doc_id,
    heading,
    precedence_tier,
    hop_depth,
    edge_via,
    condition_predicate,
    traversal_path,
    content
FROM graph_walk
ORDER BY node_id, hop_depth ASC;
