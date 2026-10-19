"""
Dagster Asset Pipeline for the Borealis Specification Graph Compiler
with Deterministic Step-by-Step Content-Addressed Caching:
  Step 1: raw_document_ast          (PDF/DOCX -> Hierarchical AST + Deterministic Anchors, cached by raw_docs SHA-256)
  Step 2: compiled_clause_graph     (AST -> Pinned-Boundary dLLM Mercury-2 Graph, cached by AST SHA-256)
  Step 3: embedded_spec_nodes       (1536-d Vector Embeddings via OpenAI, cached by node content SHA-256)
  Step 4: postgres_spec_graph       (Idempotent DB Schema Init + Upsert spec_nodes & spec_edges, cached in Postgres)
  Step 5: hybrid_cte_verification   (Execute Hybrid Vector-Anchor + Multi-Hop Override CTE & Display Results)
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import psycopg
from dagster import AssetExecutionContext, MetadataValue, Output, asset
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from starter.stage1_doc_ast_extractor import compile_raw_directory_to_ast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

STARTER_DIR = PROJECT_ROOT / "starter"
RAW_DOCS_DIR = STARTER_DIR / "raw_docs"
SCHEMA_SQL_PATH = STARTER_DIR / "schema_and_cte.sql"
AST_JSON_PATH = STARTER_DIR / "extracted_document_ast.json"
COMPILED_GRAPH_JSON_PATH = STARTER_DIR / "compiled_clause_graph.json"

CACHE_DIR = PROJECT_ROOT / ".cache" / "borealis"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_META_PATH = CACHE_DIR / "stage1_ast_meta.json"
STAGE2_META_PATH = CACHE_DIR / "stage2_graph_meta.json"
STAGE3_NODES_CACHE_PATH = CACHE_DIR / "stage3_embedded_nodes.json"
STAGE5_QUERY_CACHE_PATH = CACHE_DIR / "stage5_query_embeddings.json"
STAGE5_RESULTS_PATH = CACHE_DIR / "stage5_cte_results.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def _hash_raw_docs_dir(raw_dir: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(raw_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".docx"}:
            hasher.update(file_path.name.encode("utf-8"))
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def ensure_postgres_schema_initialized(conn: psycopg.Connection) -> None:
    """
    Idempotently initializes pgvector, spec_nodes, spec_edges, resolve_spec_context(),
    and pipeline_step_cache in PostgreSQL using starter/schema_and_cte.sql.
    """
    raw_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    marker = "WITH RECURSIVE"
    idx = raw_sql.find(marker)
    if idx == -1:
        ddl_sql = raw_sql
        cte_sql = ""
    else:
        ddl_sql = raw_sql[:idx]
        cte_sql = raw_sql[idx:]

    # Make DDL statements idempotent
    ddl_sql_idempotent = (
        ddl_sql.replace("CREATE TABLE spec_nodes", "CREATE TABLE IF NOT EXISTS spec_nodes")
        .replace("CREATE TABLE spec_edges", "CREATE TABLE IF NOT EXISTS spec_edges")
        .replace("CREATE INDEX idx_", "CREATE INDEX IF NOT EXISTS idx_")
    )

    with conn.cursor() as cur:
        cur.execute(ddl_sql_idempotent)
        if cte_sql.strip():
            cur.execute(
                f"""
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
                AS $func$
                {cte_sql}
                $func$;
                """
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_step_cache (
                step_name TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    conn.commit()


def build_pinned_clause_schema(valid_node_universe: List[str]) -> Dict[str, Any]:
    """
    Builds a pinned JSON Schema per AST clause so `source_node_id` and `target_node_id`
    are strictly constrained to the valid universe of extracted AST node IDs and Part SKUs.
    """
    return {
        "name": "pinned_ast_edge_resolver",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "enum": valid_node_universe},
                "semantic_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_id": {"type": "string", "enum": valid_node_universe},
                            "target_node_id": {"type": "string", "enum": valid_node_universe},
                            "relation_type": {
                                "type": "string",
                                "enum": [
                                    "OVERRIDES",
                                    "PROHIBITS",
                                    "REQUIRES",
                                    "TRIGGERS",
                                    "REFERENCES",
                                    "UPGRADES_TO",
                                ],
                            },
                            "condition_predicate": {
                                "type": ["string", "null"],
                                "description": (
                                    "Formal boolean expression over site variables "
                                    "(ambient_temp_c, pressure_psi, loop_id, methanol_present)"
                                ),
                            },
                        },
                        "required": [
                            "source_node_id",
                            "target_node_id",
                            "relation_type",
                            "condition_predicate",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["clause_id", "semantic_edges"],
            "additionalProperties": False,
        },
    }


@asset(
    group_name="borealis_graph_compiler",
    description="Step 1: Extracts hierarchical Document AST and deterministic regex anchors from raw PDF/DOCX specs (cached by raw_docs SHA-256).",
)
def raw_document_ast(context: AssetExecutionContext) -> Output[Dict[str, Any]]:
    raw_docs_hash = _hash_raw_docs_dir(RAW_DOCS_DIR)
    cache_hit = False

    if AST_JSON_PATH.exists() and STAGE1_META_PATH.exists():
        try:
            meta = json.loads(STAGE1_META_PATH.read_text(encoding="utf-8"))
            if meta.get("raw_docs_hash") == raw_docs_hash:
                ast_data = json.loads(AST_JSON_PATH.read_text(encoding="utf-8"))
                cache_hit = True
                context.log.info(
                    f"[CACHE HIT] Reusing cached Document AST from {AST_JSON_PATH} (hash={raw_docs_hash[:12]})"
                )
        except Exception as exc:
            context.log.warning(f"Stage 1 cache read failed, re-parsing raw docs: {exc}")

    if not cache_hit:
        context.log.info(f"[CACHE MISS] Parsing raw documents in {RAW_DOCS_DIR}...")
        ast_data = compile_raw_directory_to_ast(
            input_dir=str(RAW_DOCS_DIR),
            output_json_path=str(AST_JSON_PATH),
        )
        STAGE1_META_PATH.write_text(
            json.dumps({"raw_docs_hash": raw_docs_hash}, indent=2), encoding="utf-8"
        )

    docs = ast_data.get("documents", [])
    sections = [s for d in docs for s in d.get("sections", [])]
    clauses = [c for s in sections for c in s.get("clauses", [])]
    all_skus = sorted({sku for c in clauses for sku in c.get("deterministic_part_skus", [])})
    all_refs = sorted({ref for c in clauses for ref in c.get("deterministic_cross_refs", [])})

    return Output(
        value=ast_data,
        metadata={
            "cache_hit": cache_hit,
            "content_hash": raw_docs_hash[:16],
            "document_count": len(docs),
            "section_count": len(sections),
            "clause_count": len(clauses),
            "part_skus": MetadataValue.json(all_skus),
            "cross_refs": MetadataValue.json(all_refs),
            "ast_json_path": MetadataValue.path(str(AST_JSON_PATH)),
        },
    )


async def _compile_clauses_with_dllm(
    all_clauses: List[Dict[str, Any]],
    valid_node_universe: List[str],
    logger: Any,
) -> List[Dict[str, Any]]:
    api_key = os.environ.get("INCEPTION_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("DLLM_BASE_URL", "https://api.inceptionlabs.ai/v1")
    model = os.environ.get("DLLM_MODEL", "mercury-2")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def compile_single_clause(clause: Dict[str, Any]) -> Dict[str, Any]:
        local_candidate_nodes = sorted(
            {
                clause["clause_id"],
                clause["parent_section_id"],
                *clause.get("deterministic_cross_refs", []),
                *clause.get("deterministic_part_skus", []),
            }
        )
        pinned_universe = (
            local_candidate_nodes if len(local_candidate_nodes) > 1 else valid_node_universe
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Stage 2 of a deterministic specification compiler. Stage 1 AST parsing has already "
                        "identified the exact clause ID, cross-referenced clauses, and part SKUs present in this block. "
                        "Resolve the directed edges connecting these pinned nodes and write formal boolean condition_predicates."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "clause_id": clause["clause_id"],
                            "precedence_tier": clause["precedence_tier"],
                            "heading": clause["heading"],
                            "pinned_cross_refs": clause.get("deterministic_cross_refs", []),
                            "pinned_part_skus": clause.get("deterministic_part_skus", []),
                            "raw_text_and_tables": clause["text"],
                        }
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": build_pinned_clause_schema(pinned_universe),
            },
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        edges = parsed.get("semantic_edges", [])
        logger.info(
            f"Compiled {clause['clause_id']} ({clause['heading']}) -> {len(edges)} semantic edges"
        )

        return {
            "clause_id": clause["clause_id"],
            "parent_section_id": clause["parent_section_id"],
            "precedence_tier": clause["precedence_tier"],
            "semantic_edges": edges,
        }

    return list(await asyncio.gather(*(compile_single_clause(cl) for cl in all_clauses)))


@asset(
    group_name="borealis_graph_compiler",
    description="Step 2: Compiles the Document AST into a validated property graph using Mercury-2 dLLM (cached by AST SHA-256).",
)
def compiled_clause_graph(
    context: AssetExecutionContext, raw_document_ast: Dict[str, Any]
) -> Output[List[Dict[str, Any]]]:
    ast_hash = _sha256_json(raw_document_ast)
    cache_hit = False

    if COMPILED_GRAPH_JSON_PATH.exists():
        try:
            meta = (
                json.loads(STAGE2_META_PATH.read_text(encoding="utf-8"))
                if STAGE2_META_PATH.exists()
                else {}
            )
            compiled_graphs = json.loads(COMPILED_GRAPH_JSON_PATH.read_text(encoding="utf-8"))
            if meta.get("ast_hash") == ast_hash or (
                not STAGE2_META_PATH.exists() and len(compiled_graphs) > 0
            ):
                if not STAGE2_META_PATH.exists():
                    STAGE2_META_PATH.write_text(
                        json.dumps({"ast_hash": ast_hash}, indent=2), encoding="utf-8"
                    )
                cache_hit = True
                context.log.info(
                    f"[CACHE HIT] Reusing cached dLLM Compiled Clause Graph from {COMPILED_GRAPH_JSON_PATH} (hash={ast_hash[:12]})"
                )
        except Exception as exc:
            context.log.warning(f"Stage 2 cache read failed, re-running dLLM compiler: {exc}")

    if not cache_hit:
        all_clauses: List[Dict[str, Any]] = []
        valid_node_set: set[str] = set()

        for doc in raw_document_ast.get("documents", []):
            for sec in doc.get("sections", []):
                valid_node_set.add(sec["section_id"])
                for cl in sec.get("clauses", []):
                    all_clauses.append(cl)
                    valid_node_set.add(cl["clause_id"])
                    for ref in cl.get("deterministic_cross_refs", []):
                        valid_node_set.add(ref)
                    for sku in cl.get("deterministic_part_skus", []):
                        valid_node_set.add(sku)

        valid_node_universe = sorted(valid_node_set)
        compiled_graphs = asyncio.run(
            _compile_clauses_with_dllm(all_clauses, valid_node_universe, context.log)
        )

        COMPILED_GRAPH_JSON_PATH.write_text(
            json.dumps(compiled_graphs, indent=2), encoding="utf-8"
        )
        STAGE2_META_PATH.write_text(
            json.dumps({"ast_hash": ast_hash}, indent=2), encoding="utf-8"
        )

    all_edges = [edge for cg in compiled_graphs for edge in cg.get("semantic_edges", [])]
    md_rows = [
        "| Source | Relation | Target | Condition Predicate |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for e in all_edges:
        md_rows.append(
            f"| `{e['source_node_id']}` | **{e['relation_type']}** | `{e['target_node_id']}` | `{e.get('condition_predicate')}` |"
        )

    return Output(
        value=compiled_graphs,
        metadata={
            "cache_hit": cache_hit,
            "ast_hash": ast_hash[:16],
            "clauses_compiled": len(compiled_graphs),
            "total_semantic_edges": len(all_edges),
            "compiled_graph_path": MetadataValue.path(str(COMPILED_GRAPH_JSON_PATH)),
            "edges_preview": MetadataValue.md("\n".join(md_rows)),
        },
    )


@asset(
    group_name="borealis_graph_compiler",
    description="Step 3: Builds spec_nodes and computes 1536-d embeddings via OpenAI text-embedding-3-small (cached by node content SHA-256).",
)
def embedded_spec_nodes(
    context: AssetExecutionContext,
    raw_document_ast: Dict[str, Any],
    compiled_clause_graph: List[Dict[str, Any]],
) -> Output[List[Dict[str, Any]]]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    for doc in raw_document_ast.get("documents", []):
        doc_id = doc["doc_id"]
        for sec in doc.get("sections", []):
            sec_id = sec["section_id"]
            nodes_by_id[sec_id] = {
                "node_id": sec_id,
                "doc_id": doc_id,
                "node_type": "CLAUSE",
                "heading": sec["title"],
                "content": f"Section {sec_id}: {sec['title']}",
                "precedence_tier": sec["default_precedence_tier"],
                "embedding": None,
            }
            for cl in sec.get("clauses", []):
                cl_id = cl["clause_id"]
                nodes_by_id[cl_id] = {
                    "node_id": cl_id,
                    "doc_id": doc_id,
                    "node_type": "CLAUSE",
                    "heading": cl["heading"],
                    "content": cl["text"],
                    "precedence_tier": cl["precedence_tier"],
                    "embedding": None,
                }
                for sku in cl.get("deterministic_part_skus", []):
                    if sku not in nodes_by_id:
                        nodes_by_id[sku] = {
                            "node_id": sku,
                            "doc_id": doc_id,
                            "node_type": "PART",
                            "heading": f"Component SKU {sku.replace('PART:', '')}",
                            "content": f"Hardware / Elastomer / Piping Component {sku} referenced in {cl_id} ({cl['heading']}).",
                            "precedence_tier": cl["precedence_tier"],
                            "embedding": None,
                        }

    for cg in compiled_clause_graph:
        for edge in cg.get("semantic_edges", []):
            for nid in (edge["source_node_id"], edge["target_node_id"]):
                if nid not in nodes_by_id:
                    ntype = "PART" if nid.startswith("PART:") else "CLAUSE"
                    nodes_by_id[nid] = {
                        "node_id": nid,
                        "doc_id": "DOC-SYNTHETIC",
                        "node_type": ntype,
                        "heading": nid,
                        "content": f"Referenced specification entity {nid}",
                        "precedence_tier": cg.get("precedence_tier", 1),
                        "embedding": None,
                    }

    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    ordered_nodes = list(nodes_by_id.values())
    nodes_hash = _sha256_json(
        {"model": embedding_model, "nodes": ordered_nodes}
    )

    cache_hit = False
    if STAGE3_NODES_CACHE_PATH.exists():
        try:
            cached_payload = json.loads(STAGE3_NODES_CACHE_PATH.read_text(encoding="utf-8"))
            if cached_payload.get("nodes_hash") == nodes_hash:
                ordered_nodes = cached_payload["nodes"]
                cache_hit = True
                context.log.info(
                    f"[CACHE HIT] Reusing {len(ordered_nodes)} cached 1536-d node embeddings from {STAGE3_NODES_CACHE_PATH}"
                )
        except Exception as exc:
            context.log.warning(f"Stage 3 embedding cache read failed, recomputing: {exc}")

    if not cache_hit:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts_to_embed = [
            f"{n['node_id']} | {n['heading'] or ''}\n{n['content']}" for n in ordered_nodes
        ]
        context.log.info(
            f"[CACHE MISS] Requesting 1536-d embeddings for {len(ordered_nodes)} specification nodes using {embedding_model}..."
        )
        emb_response = openai_client.embeddings.create(
            model=embedding_model,
            input=texts_to_embed,
        )
        for node, emb_item in zip(ordered_nodes, emb_response.data):
            node["embedding"] = emb_item.embedding

        STAGE3_NODES_CACHE_PATH.write_text(
            json.dumps({"nodes_hash": nodes_hash, "nodes": ordered_nodes}),
            encoding="utf-8",
        )

    clause_count = sum(1 for n in ordered_nodes if n["node_type"] == "CLAUSE")
    part_count = sum(1 for n in ordered_nodes if n["node_type"] == "PART")

    return Output(
        value=ordered_nodes,
        metadata={
            "cache_hit": cache_hit,
            "nodes_hash": nodes_hash[:16],
            "total_nodes": len(ordered_nodes),
            "clause_nodes": clause_count,
            "part_nodes": part_count,
            "embedding_model": embedding_model,
            "embedding_dimensions": len(ordered_nodes[0]["embedding"]) if ordered_nodes else 0,
        },
    )


@asset(
    group_name="borealis_graph_compiler",
    description="Step 4: Ensures PostgreSQL schema is initialized and upserts spec_nodes and spec_edges (cached in Postgres by payload hash).",
)
def postgres_spec_graph(
    context: AssetExecutionContext,
    raw_document_ast: Dict[str, Any],
    compiled_clause_graph: List[Dict[str, Any]],
    embedded_spec_nodes: List[Dict[str, Any]],
) -> Output[Dict[str, Any]]:
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/borealis_specs"
    )

    edges_to_upsert: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for cg in compiled_clause_graph:
        for edge in cg.get("semantic_edges", []):
            key = (edge["source_node_id"], edge["target_node_id"], edge["relation_type"])
            edges_to_upsert[key] = {
                "source_node_id": edge["source_node_id"],
                "target_node_id": edge["target_node_id"],
                "relation_type": edge["relation_type"],
                "condition_predicate": edge.get("condition_predicate"),
                "confidence": 1.0,
            }

    existing_pairs = {(s, t) for (s, t, _) in edges_to_upsert}
    for doc in raw_document_ast.get("documents", []):
        for sec in doc.get("sections", []):
            for cl in sec.get("clauses", []):
                src = cl["clause_id"]
                for ref in cl.get("deterministic_cross_refs", []):
                    if (src, ref) not in existing_pairs and (ref, src) not in existing_pairs:
                        key = (src, ref, "REFERENCES")
                        edges_to_upsert[key] = {
                            "source_node_id": src,
                            "target_node_id": ref,
                            "relation_type": "REFERENCES",
                            "condition_predicate": None,
                            "confidence": 1.0,
                        }

    ordered_edges = list(edges_to_upsert.values())
    graph_state_hash = _sha256_json(
        {
            "nodes": [
                {k: v for k, v in n.items() if k != "embedding"}
                | {"emb_head": n["embedding"][:4] if n.get("embedding") else []}
                for n in embedded_spec_nodes
            ],
            "edges": ordered_edges,
        }
    )

    cache_hit = False
    with psycopg.connect(db_url) as conn:
        # 1. Ensure schema, indexes, and resolve_spec_context() are initialized
        ensure_postgres_schema_initialized(conn)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM pipeline_step_cache WHERE step_name = 'postgres_spec_graph';"
            )
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM spec_nodes;")
            db_node_count = cur.fetchone()[0]

            if row and row[0] == graph_state_hash and db_node_count == len(embedded_spec_nodes):
                cache_hit = True
                context.log.info(
                    f"[CACHE HIT] PostgreSQL already contains up-to-date spec_nodes ({db_node_count}) and spec_edges ({len(ordered_edges)}) (hash={graph_state_hash[:12]})"
                )
            else:
                context.log.info(
                    f"[CACHE MISS] Loading {len(embedded_spec_nodes)} nodes and {len(ordered_edges)} edges into PostgreSQL..."
                )
                cur.execute("TRUNCATE TABLE spec_edges, spec_nodes RESTART IDENTITY CASCADE;")

                for node in embedded_spec_nodes:
                    vec_literal = "[" + ",".join(f"{x:.8f}" for x in node["embedding"]) + "]"
                    cur.execute(
                        """
                        INSERT INTO spec_nodes (
                            node_id, doc_id, node_type, heading, content, precedence_tier, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (node_id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            node_type = EXCLUDED.node_type,
                            heading = EXCLUDED.heading,
                            content = EXCLUDED.content,
                            precedence_tier = EXCLUDED.precedence_tier,
                            embedding = EXCLUDED.embedding;
                        """,
                        (
                            node["node_id"],
                            node["doc_id"],
                            node["node_type"],
                            node["heading"],
                            node["content"],
                            node["precedence_tier"],
                            vec_literal,
                        ),
                    )

                for edge in ordered_edges:
                    cur.execute(
                        """
                        INSERT INTO spec_edges (
                            source_node_id, target_node_id, relation_type, condition_predicate, confidence
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (source_node_id, target_node_id, relation_type) DO UPDATE SET
                            condition_predicate = EXCLUDED.condition_predicate,
                            confidence = EXCLUDED.confidence;
                        """,
                        (
                            edge["source_node_id"],
                            edge["target_node_id"],
                            edge["relation_type"],
                            edge["condition_predicate"],
                            edge["confidence"],
                        ),
                    )

                cur.execute(
                    """
                    INSERT INTO pipeline_step_cache (step_name, content_hash, updated_at)
                    VALUES ('postgres_spec_graph', %s, NOW())
                    ON CONFLICT (step_name) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        updated_at = NOW();
                    """,
                    (graph_state_hash,),
                )
        conn.commit()

    relation_counts: Dict[str, int] = {}
    for e in ordered_edges:
        rel = e["relation_type"]
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    summary = {
        "cache_hit": cache_hit,
        "graph_state_hash": graph_state_hash[:16],
        "nodes_loaded": len(embedded_spec_nodes),
        "edges_loaded": len(ordered_edges),
        "relation_breakdown": relation_counts,
    }

    return Output(
        value=summary,
        metadata={
            "cache_hit": cache_hit,
            "graph_state_hash": graph_state_hash[:16],
            "nodes_loaded": len(embedded_spec_nodes),
            "edges_loaded": len(ordered_edges),
            "relation_breakdown": MetadataValue.json(relation_counts),
        },
    )


def display_cte_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Renders the Hybrid Vector-Anchor + Multi-Hop Override CTE resolution table to stdout."""
    console = Console(width=140)
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Hybrid Vector-Anchor + Multi-Hop Override Resolution CTE[/bold cyan]\n"
            f"[yellow]Query:[/yellow] {query}\n"
            f"[green]Resolved Nodes:[/green] {len(results)} across Tiers 1 -> 2 -> 3",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Tier", justify="center", style="bold")
    table.add_column("Hop", justify="center")
    table.add_column("Node ID", style="cyan", no_wrap=True)
    table.add_column("Via Edge", style="yellow", no_wrap=True)
    table.add_column("Condition Predicate", style="green")
    table.add_column("Heading / Component", style="white")
    table.add_column("Traversal Path", style="dim")

    for r in results:
        tier_val = r["precedence_tier"]
        tier_label = (
            f"[red]T{tier_val} (SAFETY)[/red]"
            if tier_val == 3
            else (f"[yellow]T{tier_val} (OVERRIDE)[/yellow]" if tier_val == 2 else f"T{tier_val} (BASE)")
        )
        table.add_row(
            tier_label,
            str(r["hop_depth"]),
            r["node_id"],
            r["edge_via"],
            str(r["condition_predicate"] or "-"),
            r["heading"] or "",
            " -> ".join(r["traversal_path"]),
        )

    console.print(table)
    console.print()


@asset(
    group_name="borealis_graph_compiler",
    description="Step 5: Executes the Hybrid Vector-Anchor + Multi-Hop Override Resolution CTE in PostgreSQL and displays results.",
)
def hybrid_cte_verification(
    context: AssetExecutionContext,
    postgres_spec_graph: Dict[str, Any],
) -> Output[List[Dict[str, Any]]]:
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/borealis_specs"
    )
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    sample_query = (
        "What isolation valve, elastomer seal, and actuator SKU are required for "
        "Manifold Loop B during winter operations at -40C?"
    )
    query_cache_key = _sha256_json({"model": embedding_model, "query": sample_query})

    query_cache: Dict[str, List[float]] = {}
    if STAGE5_QUERY_CACHE_PATH.exists():
        try:
            query_cache = json.loads(STAGE5_QUERY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            query_cache = {}

    if query_cache_key in query_cache:
        q_emb = query_cache[query_cache_key]
        query_emb_cached = True
        context.log.info("[CACHE HIT] Reusing cached query embedding for CTE resolution.")
    else:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        q_emb = (
            openai_client.embeddings.create(model=embedding_model, input=[sample_query])
            .data[0]
            .embedding
        )
        query_cache[query_cache_key] = q_emb
        STAGE5_QUERY_CACHE_PATH.write_text(json.dumps(query_cache), encoding="utf-8")
        query_emb_cached = False

    vec_literal = "[" + ",".join(f"{x:.8f}" for x in q_emb) + "]"

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    node_id,
                    doc_id,
                    heading,
                    precedence_tier,
                    hop_depth,
                    edge_via,
                    condition_predicate,
                    traversal_path,
                    content
                FROM resolve_spec_context(%s::vector, 4)
                ORDER BY precedence_tier DESC, hop_depth ASC, node_id ASC;
                """,
                (vec_literal,),
            )
            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    md_table = [
        f"### Query: *{sample_query}*\n",
        "| Tier | Node ID | Heading | Hop | Via Edge | Condition Predicate | Traversal Path |",
        "| :---: | :--- | :--- | :---: | :--- | :--- | :--- |",
    ]

    for row in rows:
        item = {
            "node_id": row[0],
            "doc_id": row[1],
            "heading": row[2],
            "precedence_tier": row[3],
            "hop_depth": row[4],
            "edge_via": row[5],
            "condition_predicate": row[6],
            "traversal_path": row[7],
            "content": row[8],
        }
        results.append(item)
        path_str = " → ".join(item["traversal_path"])
        md_table.append(
            f"| **Tier {item['precedence_tier']}** | `{item['node_id']}` | {item['heading']} | "
            f"{item['hop_depth']} | `{item['edge_via']}` | `{item['condition_predicate']}` | `{path_str}` |"
        )

    STAGE5_RESULTS_PATH.write_text(
        json.dumps({"query": sample_query, "results": results}, indent=2),
        encoding="utf-8",
    )

    display_cte_results(sample_query, results)

    return Output(
        value=results,
        metadata={
            "query": sample_query,
            "query_embedding_cached": query_emb_cached,
            "resolved_node_count": len(results),
            "highest_precedence_tier": max((r["precedence_tier"] for r in results), default=0),
            "results_cache_path": MetadataValue.path(str(STAGE5_RESULTS_PATH)),
            "resolution_trace": MetadataValue.md("\n".join(md_table)),
        },
    )


@asset(
    group_name="borealis_graph_compiler",
    description="Step 6: Runs the SOTA Graph-Compiled RAG Agent (CTE + Symbolic Predicate Compiler + Pinned Synthesis + Safety Interlock Audit).",
)
def sota_rag_agent_evaluation(
    context: AssetExecutionContext,
    hybrid_cte_verification: List[Dict[str, Any]],
) -> Output[Dict[str, Any]]:
    from borealis_dagster.rag_agent import BorealisGraphRAGAgent, display_rag_agent_report

    agent = BorealisGraphRAGAgent()
    benchmark_query = (
        "Specify the complete isolation valve, elastomer seal, actuator, piping, and flange bolting "
        "SKUs for Manifold Loop B at -40C and 120 PSI."
    )
    result = agent.answer_query(benchmark_query, max_hops=4, use_cache=True)

    if result.get("cache_hit"):
        context.log.info("[CACHE HIT] Reusing cached SOTA Graph-Compiled RAG Agent evaluation.")
    else:
        context.log.info("[CACHE MISS] Executed SOTA Graph-Compiled RAG Agent synthesis and safety audit.")

    display_rag_agent_report(result)

    synthesis = result["synthesis"]
    audit = result["safety_audit"]

    md_lines = [
        f"### Executive Summary\n{synthesis.get('executive_summary', '')}\n",
        "### Audited Bill of Materials (BOM)",
        "| Subsystem | Selected SKU | Governing Clause | Precedence Tier | Rationale |",
        "| :--- | :--- | :--- | :---: | :--- |",
    ]
    for item in synthesis.get("recommended_bom", []):
        md_lines.append(
            f"| **{item['subsystem']}** | `{item['selected_part_sku']}` | `{item['governing_clause_id']}` | "
            f"Tier {item['precedence_tier']} | {item['engineering_rationale']} |"
        )

    md_lines.extend(
        [
            "\n### Rejected Baseline & Intermediate SKUs",
            "| Prohibited SKU | Baseline Clause | Overriding Clause | Failure Mode Prevented |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for rej in synthesis.get("rejected_baseline_parts", []):
        md_lines.append(
            f"| `{rej['prohibited_part_sku']}` | `{rej['baseline_clause_id']}` | "
            f"`{rej['overriding_clause_id']}` | {rej['failure_mode_prevented']} |"
        )

    return Output(
        value=result,
        metadata={
            "cache_hit": result.get("cache_hit", False),
            "safety_audit_passed": audit["passed"],
            "selected_skus": MetadataValue.json(audit["selected_skus"]),
            "prohibited_skus_blocked": MetadataValue.json(audit["prohibited_skus_blocked"]),
            "report": MetadataValue.md("\n".join(md_lines)),
        },
    )

