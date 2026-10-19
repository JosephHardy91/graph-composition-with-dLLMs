from dagster import AssetSelection, Definitions, define_asset_job

from borealis_dagster.assets import (
    compiled_clause_graph,
    embedded_spec_nodes,
    hybrid_cte_verification,
    postgres_spec_graph,
    raw_document_ast,
    sota_rag_agent_evaluation,
)

compile_and_load_spec_graph_job = define_asset_job(
    name="compile_and_load_spec_graph_job",
    selection=AssetSelection.groups("borealis_graph_compiler"),
    description="End-to-end pipeline: Raw PDF/DOCX -> AST -> Mercury-2 dLLM Graph -> OpenAI Embeddings -> Postgres pgvector + CTE + SOTA RAG Agent.",
)

defs = Definitions(
    assets=[
        raw_document_ast,
        compiled_clause_graph,
        embedded_spec_nodes,
        postgres_spec_graph,
        hybrid_cte_verification,
        sota_rag_agent_evaluation,
    ],
    jobs=[compile_and_load_spec_graph_job],
)
