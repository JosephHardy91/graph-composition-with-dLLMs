"""
Bootstrap runner that:
1. Waits for PostgreSQL + pgvector to accept connections
2. Materializes all Borealis Dagster assets into the persistent DAGSTER_HOME instance
   (reusing content-addressed step caches in .cache/borealis/ when unchanged)
3. Displays the Hybrid Vector-Anchor + Multi-Hop Override CTE results table
   and the SOTA Graph-Compiled RAG Agent BOM & Safety Audit report
"""

import os
import sys
import time
from pathlib import Path

import psycopg
from dagster import DagsterInstance, materialize
from dotenv import load_dotenv
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from borealis_dagster.assets import (  # noqa: E402
    compiled_clause_graph,
    embedded_spec_nodes,
    hybrid_cte_verification,
    postgres_spec_graph,
    raw_document_ast,
    sota_rag_agent_evaluation,
)


def wait_for_postgres(db_url: str, timeout_sec: int = 30) -> None:
    console = Console()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with psycopg.connect(db_url, connect_timeout=3):
                return
        except Exception:
            time.sleep(1)
    console.print(f"[bold red]Timed out waiting for PostgreSQL at {db_url}[/bold red]")
    sys.exit(1)


def main() -> None:
    console = Console()
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/borealis_specs"
    )

    dagster_home = os.environ.get("DAGSTER_HOME", str(PROJECT_ROOT / ".dagster_home"))
    Path(dagster_home).mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = dagster_home

    console.print("[bold blue]==> Waiting for PostgreSQL + pgvector database...[/bold blue]")
    wait_for_postgres(db_url)

    console.print(
        f"[bold blue]==> Materializing Borealis Graph Compiler & SOTA RAG Agent assets (DAGSTER_HOME={dagster_home})...[/bold blue]"
    )
    instance = DagsterInstance.get()
    result = materialize(
        assets=[
            raw_document_ast,
            compiled_clause_graph,
            embedded_spec_nodes,
            postgres_spec_graph,
            hybrid_cte_verification,
            sota_rag_agent_evaluation,
        ],
        instance=instance,
    )

    if not result.success:
        console.print("[bold red]==> Asset materialization failed![/bold red]")
        sys.exit(1)

    console.print(
        "[bold green]==> All 6 stages (including SOTA Graph-Compiled RAG Agent) materialized & cached successfully![/bold green]"
    )


if __name__ == "__main__":
    main()
