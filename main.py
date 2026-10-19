"""
CLI entrypoint for the Borealis SOTA Graph-Compiled RAG Agent.
Streams the Layer 4/5 tables at ~0.42s (`jev-latest`) and streams Layer 6 prose (`gpt-6.1-sol` low)
token-by-token so you can observe real-time TTFT (~1.83s E2E).

Usage:
  uv run python main.py
  uv run python main.py "What valve and seal should be used for Manifold Loop A at -38C?"
"""

import sys

from borealis_dagster.rag_agent import BorealisGraphRAGAgent


def main() -> None:
    agent = BorealisGraphRAGAgent()
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "Specify the complete isolation valve, elastomer seal, actuator, piping, and flange bolting SKUs for Manifold Loop B at -40C and 120 PSI.",
            "Specify the isolation valve, elastomer seal, and actuator SKUs for Manifold Loop A (Glycol Return) at -38C and 110 PSI.",
        ]

    for q in queries:
        agent.answer_query(q, max_hops=4, use_cache=True, stream_output=True)


if __name__ == "__main__":
    main()
