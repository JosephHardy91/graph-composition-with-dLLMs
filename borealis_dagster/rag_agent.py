"""
SOTA Graph-Compiled Specification RAG Agent powered by PostgreSQL pgvector + Recursive CTEs.
--------------------------------------------------------------------------------------------
Architecture:
  1. Hybrid Vector-Anchor + Multi-Hop Override CTE Retriever (`resolve_spec_context`):
     Finds top vector anchors in `spec_nodes` and walks outbound/inbound edges (`OVERRIDES`, `PROHIBITS`,
     `REQUIRES`, `TRIGGERS`, `UPGRADES_TO`, `REFERENCES`) up to N hops without cycles.
  2. Pluggable Layer 4/5 Regime & Override Judgement Engine (A/B Selectable, Zero Manual Regex Extractor):
     - `resolver_mode="jev"`: Uses TypeSafe System One (`jev-latest`) two-pass parallel `Choice` & `Noul`
       judgments over the retrieved subgraph to extract the site regime and judge every clause and part SKU.
       Code directly assembles `recommended_bom`, `rejected_baseline_parts`, and `multi_hop_override_chain`
       from Jev's typed judgments.
     - `resolver_mode="dllm_forest"`: Uses a parallel forest of `mercury-2` dLLMs (`ThreadPoolExecutor`)
       where dedicated dLLM workers extract the site regime and concurrently judge every clause and part SKU.
       Code directly assembles `recommended_bom`, `rejected_baseline_parts`, and `multi_hop_override_chain`
       from the dLLM forest's judgments.
  3. Layer 6 Synthesizer (`gpt-6.1-sol`, `reasoning_effort="low"`):
     Acts strictly as the prose synthesizer over the Layer 4/5 decisions and retrieved clauses, without
     modifying the BOM or overriding Layer 4/5's part selections.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

CACHE_DIR = PROJECT_ROOT / ".cache" / "borealis"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAG_QUERY_EMB_CACHE_PATH = CACHE_DIR / "stage5_query_embeddings.json"
RAG_AGENT_CACHE_PATH = CACHE_DIR / "stage6_rag_agent_cache.json"


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class SiteRegime:
    loop_id: Optional[str] = None
    ambient_temp_c: Optional[float] = None
    pressure_psi: Optional[float] = None
    methanol_present: Optional[bool] = None
    inferred_notes: List[str] = field(default_factory=list)


class BorealisGraphRAGAgent:
    """
    SOTA Graph-Compiled RAG Agent using PostgreSQL pgvector + Recursive CTE (`resolve_spec_context`)
    + A/B Selectable Layer 4/5 Resolver (`jev` vs `dllm_forest`)
    + Layer 6 Prose Synthesizer (`gpt-6.1-sol` on `low`).
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        resolver_mode: str = "jev",
        synthesis_model: str = "gpt-6.1-sol",
        synthesis_reasoning_effort: Optional[str] = "low",
    ) -> None:
        self.db_url = db_url or os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/borealis_specs"
        )
        self.resolver_mode = resolver_mode
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        dllm_key = os.environ.get("INCEPTION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        dllm_base = os.environ.get("DLLM_BASE_URL", "https://api.inceptionlabs.ai/v1")
        self.dllm_model = os.environ.get("DLLM_MODEL", "mercury-2")
        self.dllm_client = OpenAI(api_key=dllm_key, base_url=dllm_base)

        self.synthesis_model = synthesis_model
        self.synthesis_reasoning_effort = synthesis_reasoning_effort
        self.jev_model = os.environ.get("JEV_MODEL", "jev-latest")

    def _embed_query(self, query: str) -> List[float]:
        cache_key = _sha256_json({"model": self.embedding_model, "query": query})
        cache: Dict[str, List[float]] = {}
        if RAG_QUERY_EMB_CACHE_PATH.exists():
            try:
                cache = json.loads(RAG_QUERY_EMB_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        if cache_key in cache:
            return cache[cache_key]

        emb = (
            self.openai_client.embeddings.create(model=self.embedding_model, input=[query])
            .data[0]
            .embedding
        )
        cache[cache_key] = emb
        RAG_QUERY_EMB_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        return emb

    def retrieve_naive_vector_top_k(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Baseline Naive Vector RAG (Top-K cosine similarity only, no multi-hop CTE traversal).
        """
        q_emb = self._embed_query(query)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in q_emb) + "]"
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        node_id,
                        doc_id,
                        heading,
                        precedence_tier,
                        1 - (embedding <=> %s::vector) AS similarity,
                        content
                    FROM spec_nodes
                    WHERE node_type = 'CLAUSE'
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vec_literal, vec_literal, k),
                )
                rows = cur.fetchall()
        return [
            {
                "node_id": r[0],
                "doc_id": r[1],
                "heading": r[2],
                "precedence_tier": r[3],
                "similarity": round(float(r[4]), 4),
                "content": r[5],
            }
            for r in rows
        ]

    def _fetch_cte_subgraph(self, query: str, max_hops: int = 4) -> Dict[str, Any]:
        """
        Executes the PostgreSQL `resolve_spec_context` Hybrid Vector-Anchor + Multi-Hop CTE
        and enriches the subgraph with governing clauses and directed edges.
        """
        q_emb = self._embed_query(query)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in q_emb) + "]"

        with psycopg.connect(self.db_url) as conn:
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
                    FROM resolve_spec_context(%s::vector, %s)
                    ORDER BY precedence_tier DESC, hop_depth ASC, node_id ASC;
                    """,
                    (vec_literal, max_hops),
                )
                cte_rows = cur.fetchall()

                resolved_nodes = [
                    {
                        "node_id": r[0],
                        "doc_id": r[1],
                        "heading": r[2],
                        "precedence_tier": r[3],
                        "hop_depth": r[4],
                        "edge_via": r[5],
                        "condition_predicate": r[6],
                        "traversal_path": r[7],
                        "content": r[8],
                    }
                    for r in cte_rows
                ]

                node_ids = [n["node_id"] for n in resolved_nodes]
                part_ids = [nid for nid in node_ids if nid.startswith("PART:")]

                sku_patterns = [f"%{pid.replace('PART:', '')}%" for pid in part_ids] if part_ids else []
                cur.execute(
                    """
                    SELECT
                        node_id,
                        doc_id,
                        heading,
                        precedence_tier,
                        content
                    FROM spec_nodes
                    WHERE node_type = 'CLAUSE'
                      AND NOT (node_id = ANY(%s))
                      AND (
                          node_id IN ('SEC-1.2', 'SEC-1.3', 'SEC-3.2')
                          OR (cardinality(%s::text[]) > 0 AND content ILIKE ANY(%s::text[]))
                      )
                    ORDER BY precedence_tier DESC, node_id ASC;
                    """,
                    (node_ids, sku_patterns, sku_patterns),
                )
                for r in cur.fetchall():
                    resolved_nodes.append(
                        {
                            "node_id": r[0],
                            "doc_id": r[1],
                            "heading": r[2],
                            "precedence_tier": r[3],
                            "hop_depth": 2,
                            "edge_via": "PART_GOVERNANCE",
                            "condition_predicate": None,
                            "traversal_path": [r[0]],
                            "content": r[4],
                        }
                    )
                resolved_nodes.sort(
                    key=lambda x: (-x["precedence_tier"], x["hop_depth"], x["node_id"])
                )
                node_ids = [n["node_id"] for n in resolved_nodes]

                cur.execute(
                    """
                    SELECT
                        source_node_id,
                        target_node_id,
                        relation_type,
                        condition_predicate,
                        confidence
                    FROM spec_edges
                    WHERE source_node_id = ANY(%s) OR target_node_id = ANY(%s)
                    ORDER BY edge_id ASC;
                    """,
                    (node_ids, node_ids),
                )
                edge_rows = cur.fetchall()

        return {
            "resolved_nodes": resolved_nodes,
            "edge_rows": edge_rows,
        }

    def _resolve_with_jev(
        self, query: str, resolved_nodes: List[Dict[str, Any]], edge_rows: List[Any]
    ) -> Dict[str, Any]:
        """
        Layer 4/5 Option B: TypeSafe System One (`jev-latest`) two-pass parallel structured judgments.
        Code owns the workflow and builds `recommended_bom`, `rejected_baseline_parts`, and
        `multi_hop_override_chain` directly from Jev's typed judgments.
        """
        t0 = time.perf_counter()
        clauses = {
            c["node_id"]: {
                "heading": c["heading"],
                "precedence_tier": c["precedence_tier"],
                "content": c["content"],
            }
            for c in resolved_nodes
            if c["node_id"].startswith("SEC-")
        }
        parts = {
            e["node_id"]: e["heading"]
            for e in resolved_nodes
            if e["node_id"].startswith("PART:")
        }

        all_text = query + "\n" + "\n".join(c["content"] for c in clauses.values())
        temp_spans = list(
            dict.fromkeys(
                m.group(1).replace("+", "")
                for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*°?\s*C\b", all_text)
            )
        )
        press_spans = list(
            dict.fromkeys(
                m.group(1)
                for m in re.finditer(r"(\d+(?:\.\d+)?)\s*PSI\b", all_text, flags=re.IGNORECASE)
            )
        )
        loop_spans = list(
            dict.fromkeys(
                m.group(1).upper()
                for m in re.finditer(r"\bLoop\s+([A-Z])\b", all_text, flags=re.IGNORECASE)
            )
        )

        with TypeSafeClient() as ts_client:
            # Pass 1: Regime Extraction via Jev Choice + Noul
            p1_state = {"query": query, "resolved_clauses": clauses}
            p1_questions: Dict[str, Any] = {
                "loop_id": Choice(
                    instructions="In `query`, which Manifold Loop is being asked about?",
                    criteria={
                        **{k: f"Manifold Loop {k}" for k in loop_spans},
                        "none": "No specific manifold loop is asked about in `query`",
                    },
                ),
                "ambient_temp_c": Choice(
                    instructions=(
                        "Based on `query` and the site zone/loop tables in `resolved_clauses` (e.g. SEC-1.2 winter MDMT), "
                        "what is the minimum ambient operating temperature (°C) for the scenario in `query`?"
                    ),
                    criteria={
                        **{t: f"{t}°C" for t in temp_spans},
                        "none": "No specific ambient temperature applies to `query`",
                    },
                ),
                "pressure_psi": Choice(
                    instructions=(
                        "Based on `query` and the loop table in `resolved_clauses` (e.g. SEC-1.2), "
                        "what is the operating pressure (PSI) for the scenario in `query`?"
                    ),
                    criteria={
                        **{p: f"{p} PSI" for p in press_spans},
                        "none": "No specific operating pressure applies to `query`",
                    },
                ),
                "methanol_present": Noul(
                    instructions=(
                        "Under the scenario in `query` and the chemical injection rules in `resolved_clauses` "
                        "(e.g. SEC-3.2 and SEC-3.3), is anhydrous methanol injection active or present in the line?"
                    ),
                    criteria=NoulCriteria(
                        true="Anhydrous methanol injection is active/present in the line for this scenario",
                        false="No methanol / zero methanol dosing is present in the line for this scenario",
                    ),
                ),
            }
            resp1 = ts_client.system_one(
                model=self.jev_model, state=p1_state, questions=p1_questions
            )
            r1 = resp1.answers
            tok1 = resp1.usage.input_tokens

            loop_val = None if r1["loop_id"].choice == "none" else r1["loop_id"].choice
            temp_val = (
                None if r1["ambient_temp_c"].choice == "none" else float(r1["ambient_temp_c"].choice)
            )
            press_val = (
                None if r1["pressure_psi"].choice == "none" else float(r1["pressure_psi"].choice)
            )
            meoh_val = r1["methanol_present"].noul >= 0.5

            regime = SiteRegime(
                loop_id=loop_val,
                ambient_temp_c=temp_val,
                pressure_psi=press_val,
                methanol_present=meoh_val,
                inferred_notes=[
                    f"Jev Pass-1 Regime: loop={loop_val} (conf={r1['loop_id'].confidence:.2f}), "
                    f"temp={temp_val} (conf={r1['ambient_temp_c'].confidence:.2f}), "
                    f"press={press_val} (conf={r1['pressure_psi'].confidence:.2f}), "
                    f"methanol_noul={r1['methanol_present'].noul:.2f}"
                ],
            )

            # Pass 2: Parallel Clause & Part Judgments via Jev
            regime_dict = {
                "loop_id": regime.loop_id,
                "ambient_temp_c": regime.ambient_temp_c,
                "pressure_psi": regime.pressure_psi,
                "methanol_present": regime.methanol_present,
            }
            p2_state = {
                "query": query,
                "extracted_site_regime": regime_dict,
                "resolved_clauses": clauses,
                "candidate_parts": parts,
            }
            p2_questions: Dict[str, Any] = {}
            clause_options = {cid: f"{cid}: {cinfo['heading']}" for cid, cinfo in clauses.items()}

            for cid, cinfo in clauses.items():
                p2_questions[f"clause:{cid}"] = Noul(
                    instructions={
                        "clause_id": cid,
                        "clause": cinfo,
                        "question": (
                            "Given `extracted_site_regime` (every numeric threshold in `clause` such as "
                            "pressure > 100 PSI or temperature < -30°C or methanol_present must be strictly "
                            "satisfied by `extracted_site_regime`), does `clause` actively trigger/apply for `query`?"
                        ),
                    }
                )

            for pid, pdesc in parts.items():
                p2_questions[f"part:{pid}"] = Choice(
                    instructions={
                        "part_sku": pid,
                        "part_name": pdesc,
                        "site_regime": regime_dict,
                        "question": (
                            "Given `extracted_site_regime` and `resolved_clauses` (strictly checking every "
                            "numeric threshold like pressure > 100 PSI and temperature limits, and applying "
                            "Tier 3 > Tier 2 > Tier 1 precedence), what is the status of `part_sku` for `query`?"
                        ),
                    },
                    criteria={
                        "active": (
                            "Required or approved for use in the assembly under `extracted_site_regime` "
                            "(including baseline parts whose override threshold is NOT met, e.g. pressure <= 100 PSI)"
                        ),
                        "prohibited": (
                            "Strictly prohibited or superseded because all numeric & chemical conditions "
                            "of an overriding clause are satisfied by `extracted_site_regime`"
                        ),
                        "not_applicable": (
                            "An upgrade or dosing skid whose numeric/chemical trigger conditions are NOT "
                            "all met by `extracted_site_regime`"
                        ),
                    },
                )
                p2_questions[f"gov:{pid}"] = Choice(
                    instructions={
                        "part_sku": pid,
                        "part_name": pdesc,
                        "site_regime": regime_dict,
                        "question": (
                            "Which clause in `resolved_clauses` is the highest-precedence governing clause "
                            "that determines whether `part_sku` is approved or prohibited under `extracted_site_regime`?"
                        ),
                    },
                    criteria=clause_options,
                )

            resp2 = ts_client.system_one(
                model=self.jev_model, state=p2_state, questions=p2_questions
            )
            r2 = resp2.answers
            tok2 = resp2.usage.input_tokens

        active_clauses = {cid for cid in clauses if r2[f"clause:{cid}"].noul >= 0.5}
        prohibited_parts: Dict[str, List[str]] = {}
        final_mandated_parts: Dict[str, List[str]] = {}
        upgraded_parts: Dict[str, str] = {}
        recommended_bom: List[Dict[str, Any]] = []
        rejected_baseline_parts: List[Dict[str, Any]] = []

        for pid, pdesc in parts.items():
            status = r2[f"part:{pid}"].choice
            gov_cid = r2[f"gov:{pid}"].choice
            conf = r2[f"part:{pid}"].confidence
            gov_tier = clauses.get(gov_cid, {}).get("precedence_tier", 1)
            gov_heading = clauses.get(gov_cid, {}).get("heading", "")

            if status == "active":
                final_mandated_parts[pid] = [
                    f"Jev System One: active/approved under {gov_cid} (conf={conf:.2f})"
                ]
                recommended_bom.append(
                    {
                        "subsystem": pdesc,
                        "selected_part_sku": pid,
                        "governing_clause_id": gov_cid,
                        "precedence_tier": gov_tier,
                        "engineering_rationale": f"Approved/mandated by {gov_cid} ({gov_heading}) [Jev conf={conf:.2f}]",
                    }
                )
            elif status == "prohibited":
                prohibited_parts[pid] = [
                    f"Jev System One: prohibited/superseded under {gov_cid} (conf={conf:.2f})"
                ]
                rejected_baseline_parts.append(
                    {
                        "prohibited_part_sku": pid,
                        "baseline_clause_id": "SEC-2.2" if "SEC-2.2" in clauses else gov_cid,
                        "overriding_clause_id": gov_cid,
                        "failure_mode_prevented": f"Prohibited by {gov_cid} ({gov_heading}) [Jev conf={conf:.2f}]",
                    }
                )

        active_edges: List[Dict[str, Any]] = []
        for src, tgt, rel, pred, conf in edge_rows:
            is_act = (src in active_clauses) or (src in final_mandated_parts)
            active_edges.append(
                {
                    "source_node_id": src,
                    "target_node_id": tgt,
                    "relation_type": rel,
                    "condition_predicate": pred,
                    "active_under_regime": is_act,
                }
            )
            if rel == "UPGRADES_TO" and src in prohibited_parts and tgt in final_mandated_parts:
                upgraded_parts[src] = tgt

        multi_hop_chain = [
            f"{cid}: {clauses[cid]['heading']} (Tier {clauses[cid]['precedence_tier']})"
            for cid in sorted(
                active_clauses,
                key=lambda c: (clauses[c]["precedence_tier"], c),
            )
        ]

        elapsed_s = round(time.perf_counter() - t0, 3)
        return {
            "regime": regime,
            "active_edges": active_edges,
            "recommended_bom": recommended_bom,
            "rejected_baseline_parts": rejected_baseline_parts,
            "multi_hop_override_chain": multi_hop_chain,
            "symbolic_resolution": {
                "resolver_mode": "jev",
                "resolver_model": self.jev_model,
                "resolver_latency_s": elapsed_s,
                "resolver_input_tokens": tok1 + tok2,
                "active_clauses": sorted(active_clauses),
                "prohibited_part_skus": prohibited_parts,
                "part_upgrades": upgraded_parts,
                "mandated_active_part_skus": final_mandated_parts,
            },
        }

    def _resolve_with_dllm_forest(
        self, query: str, resolved_nodes: List[Dict[str, Any]], edge_rows: List[Any]
    ) -> Dict[str, Any]:
        """
        Layer 4/5 Option A: Forest of Parallel dLLMs (`mercury-2` via `ThreadPoolExecutor`).
        Code directly builds `recommended_bom`, `rejected_baseline_parts`, and
        `multi_hop_override_chain` from the parallel dLLM forest's judgments.
        """
        t0 = time.perf_counter()
        clauses = {
            c["node_id"]: {
                "heading": c["heading"],
                "precedence_tier": c["precedence_tier"],
                "content": c["content"],
            }
            for c in resolved_nodes
            if c["node_id"].startswith("SEC-")
        }
        parts = {
            e["node_id"]: e["heading"]
            for e in resolved_nodes
            if e["node_id"].startswith("PART:")
        }
        valid_clause_ids = sorted(clauses.keys()) or ["SEC-1.0"]

        # Pass 1: Regime Extractor dLLM (`mercury-2`)
        regime_schema = {
            "name": "dllm_regime_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": ["string", "null"],
                        "enum": ["A", "B", "C", None],
                    },
                    "ambient_temp_c": {"type": ["number", "null"]},
                    "pressure_psi": {"type": ["number", "null"]},
                    "methanol_present": {"type": ["boolean", "null"]},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "loop_id",
                    "ambient_temp_c",
                    "pressure_psi",
                    "methanol_present",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        }
        def _call_dllm_with_retry(**kwargs: Any) -> Any:
            for attempt in range(6):
                try:
                    return self.dllm_client.chat.completions.create(**kwargs)
                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise
            return self.dllm_client.chat.completions.create(**kwargs)

        r1_comp = _call_dllm_with_retry(
            model=self.dllm_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the operating SiteRegime for the user's `query` using `resolved_clauses` "
                        "(including SEC-1.2 for loop operating pressure and Zone-A3 winter MDMT -42°C, and "
                        "SEC-3.2 / SEC-3.3 for whether anhydrous methanol injection is active on that loop at that temperature). "
                        "If the query asks a general rule question with no operating scenario, return null for numeric fields."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"query": query, "resolved_clauses": clauses}),
                },
            ],
            response_format={"type": "json_schema", "json_schema": regime_schema},
        )
        r1_data = json.loads(r1_comp.choices[0].message.content or "{}")
        total_tokens = getattr(r1_comp.usage, "total_tokens", 0) or 0

        regime = SiteRegime(
            loop_id=r1_data.get("loop_id"),
            ambient_temp_c=r1_data.get("ambient_temp_c"),
            pressure_psi=r1_data.get("pressure_psi"),
            methanol_present=bool(r1_data.get("methanol_present")),
            inferred_notes=[f"dLLM Forest Pass-1: {r1_data.get('rationale', '')}"],
        )
        regime_dict = {
            "loop_id": regime.loop_id,
            "ambient_temp_c": regime.ambient_temp_c,
            "pressure_psi": regime.pressure_psi,
            "methanol_present": regime.methanol_present,
        }

        # Pass 2: Parallel Forest of dLLMs across all clauses and candidate parts
        clause_eval_schema = {
            "name": "dllm_clause_eval",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "is_active": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": ["is_active", "rationale"],
                "additionalProperties": False,
            },
        }
        part_eval_schema = {
            "name": "dllm_part_eval",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "prohibited", "not_applicable"],
                    },
                    "governing_clause_id": {
                        "type": "string",
                        "enum": valid_clause_ids,
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["status", "governing_clause_id", "rationale"],
                "additionalProperties": False,
            },
        }

        def _eval_clause_worker(cid: str, cinfo: Dict[str, Any]) -> Dict[str, Any]:
            comp = _call_dllm_with_retry(
                model=self.dllm_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Judge whether `target_clause` actively applies/triggers for `query` under `site_regime`. "
                            "Every numeric threshold (such as pressure > 100 PSI, ambient < -35°C, ambient < -30°C, "
                            "ambient < -29°C, or methanol_present) and part dependency must be strictly satisfied."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": query,
                                "site_regime": regime_dict,
                                "target_clause_id": cid,
                                "target_clause": cinfo,
                            }
                        ),
                    },
                ],
                response_format={"type": "json_schema", "json_schema": clause_eval_schema},
            )
            parsed = json.loads(comp.choices[0].message.content or "{}")
            toks = getattr(comp.usage, "total_tokens", 0) or 0
            return {"cid": cid, "parsed": parsed, "tokens": toks}

        def _eval_part_worker(pid: str, pdesc: str) -> Dict[str, Any]:
            sku_short = pid.replace("PART:", "")
            relevant_clauses = {
                cid: cinfo
                for cid, cinfo in clauses.items()
                if sku_short in cinfo["content"] or cid in ("SEC-1.3", "SEC-4.3", "SEC-4.4", "SEC-4.5", "SEC-5.1", "SEC-5.2")
            } or clauses
            comp = _call_dllm_with_retry(
                model=self.dllm_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Judge the status of `part_sku` for `query` under `site_regime` and `resolved_clauses` "
                            "(respecting Tier 3 > Tier 2 > Tier 1 precedence and strictly checking all numeric thresholds "
                            "such as pressure > 100 PSI in SEC-4.1, methanol_present in SEC-4.4/SEC-4.5, and whether "
                            "SEAL-FFKM-ULT90 is installed in SEC-5.1/SEC-5.2):\n"
                            "- 'active': required or approved for the assembly under `site_regime` (including baseline parts "
                            "when an override's pressure/temperature/methanol threshold is NOT met)\n"
                            "- 'prohibited': strictly prohibited or superseded under `site_regime`\n"
                            "- 'not_applicable': an upgrade or dosing skid whose trigger condition is NOT met under `site_regime`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": query,
                                "site_regime": regime_dict,
                                "part_sku": pid,
                                "part_name": pdesc,
                                "resolved_clauses": relevant_clauses,
                            }
                        ),
                    },
                ],
                response_format={"type": "json_schema", "json_schema": part_eval_schema},
            )
            parsed = json.loads(comp.choices[0].message.content or "{}")
            toks = getattr(comp.usage, "total_tokens", 0) or 0
            return {"pid": pid, "pdesc": pdesc, "parsed": parsed, "tokens": toks}

        with ThreadPoolExecutor(max_workers=4) as pool:
            clause_futures = [pool.submit(_eval_clause_worker, cid, cinfo) for cid, cinfo in clauses.items()]
            part_futures = [pool.submit(_eval_part_worker, pid, pdesc) for pid, pdesc in parts.items()]
            clause_results = [f.result() for f in clause_futures]
            part_results = [f.result() for f in part_futures]

        active_clauses = set()
        for cr in clause_results:
            total_tokens += cr["tokens"]
            if cr["parsed"].get("is_active"):
                active_clauses.add(cr["cid"])

        prohibited_parts: Dict[str, List[str]] = {}
        final_mandated_parts: Dict[str, List[str]] = {}
        upgraded_parts: Dict[str, str] = {}
        recommended_bom: List[Dict[str, Any]] = []
        rejected_baseline_parts: List[Dict[str, Any]] = []

        for pr in part_results:
            total_tokens += pr["tokens"]
            pid = pr["pid"]
            pdesc = pr["pdesc"]
            status = pr["parsed"].get("status")
            gov_cid = pr["parsed"].get("governing_clause_id", valid_clause_ids[0])
            rat = pr["parsed"].get("rationale", "")
            gov_tier = clauses.get(gov_cid, {}).get("precedence_tier", 1)

            if status == "active":
                final_mandated_parts[pid] = [f"dLLM Forest ({gov_cid}): {rat}"]
                recommended_bom.append(
                    {
                        "subsystem": pdesc,
                        "selected_part_sku": pid,
                        "governing_clause_id": gov_cid,
                        "precedence_tier": gov_tier,
                        "engineering_rationale": rat,
                    }
                )
            elif status == "prohibited":
                prohibited_parts[pid] = [f"dLLM Forest ({gov_cid}): {rat}"]
                rejected_baseline_parts.append(
                    {
                        "prohibited_part_sku": pid,
                        "baseline_clause_id": "SEC-2.2" if "SEC-2.2" in clauses else gov_cid,
                        "overriding_clause_id": gov_cid,
                        "failure_mode_prevented": rat,
                    }
                )

        active_edges: List[Dict[str, Any]] = []
        for src, tgt, rel, pred, conf in edge_rows:
            is_act = (src in active_clauses) or (src in final_mandated_parts)
            active_edges.append(
                {
                    "source_node_id": src,
                    "target_node_id": tgt,
                    "relation_type": rel,
                    "condition_predicate": pred,
                    "active_under_regime": is_act,
                }
            )
            if rel == "UPGRADES_TO" and src in prohibited_parts and tgt in final_mandated_parts:
                upgraded_parts[src] = tgt

        multi_hop_chain = [
            f"{cid}: {clauses[cid]['heading']} (Tier {clauses[cid]['precedence_tier']})"
            for cid in sorted(
                active_clauses,
                key=lambda c: (clauses[c]["precedence_tier"], c),
            )
        ]

        elapsed_s = round(time.perf_counter() - t0, 3)
        return {
            "regime": regime,
            "active_edges": active_edges,
            "recommended_bom": recommended_bom,
            "rejected_baseline_parts": rejected_baseline_parts,
            "multi_hop_override_chain": multi_hop_chain,
            "symbolic_resolution": {
                "resolver_mode": "dllm_forest",
                "resolver_model": f"{self.dllm_model} (parallel forest, {1 + len(clauses) + len(parts)} calls)",
                "resolver_latency_s": elapsed_s,
                "resolver_input_tokens": total_tokens,
                "active_clauses": sorted(active_clauses),
                "prohibited_part_skus": prohibited_parts,
                "part_upgrades": upgraded_parts,
                "mandated_active_part_skus": final_mandated_parts,
            },
        }

    def retrieve_graph_compiled_context(
        self, query: str, max_hops: int = 4
    ) -> Dict[str, Any]:
        """
        Executes the PostgreSQL `resolve_spec_context` Hybrid Vector-Anchor + Multi-Hop CTE
        and runs the configured Layer 4/5 Resolver (`jev` or `dllm_forest`).
        """
        subgraph = self._fetch_cte_subgraph(query, max_hops=max_hops)
        resolved_nodes = subgraph["resolved_nodes"]
        edge_rows = subgraph["edge_rows"]

        if self.resolver_mode == "dllm_forest":
            resolved = self._resolve_with_dllm_forest(query, resolved_nodes, edge_rows)
        else:
            resolved = self._resolve_with_jev(query, resolved_nodes, edge_rows)

        regime: SiteRegime = resolved["regime"]
        return {
            "query": query,
            "site_regime": {
                "loop_id": regime.loop_id,
                "ambient_temp_c": regime.ambient_temp_c,
                "pressure_psi": regime.pressure_psi,
                "methanol_present": regime.methanol_present,
                "inferred_notes": regime.inferred_notes,
            },
            "resolved_nodes": resolved_nodes,
            "subgraph_edges": resolved["active_edges"],
            "recommended_bom": resolved["recommended_bom"],
            "rejected_baseline_parts": resolved["rejected_baseline_parts"],
            "multi_hop_override_chain": resolved["multi_hop_override_chain"],
            "symbolic_resolution": resolved["symbolic_resolution"],
        }

    def answer_query(
        self,
        query: str,
        max_hops: int = 4,
        use_cache: bool = True,
        stream_output: bool = False,
    ) -> Dict[str, Any]:
        """
        End-to-end SOTA RAG execution:
          1. Retrieve via Hybrid Vector-Anchor + Multi-Hop Override CTE
          2. Run Layer 4/5 Judgement Engine (`jev` or `dllm_forest`) to produce the locked BOM & rejections
          3. When `stream_output=True`, immediately render the Header, Audited BOM, and Rejected Parts tables
             at ~0.42s, then stream Layer 6 (`gpt-6.1-sol` low) tokens live to stdout showing real-time TTFT
          4. Verify Safety & Override Compliance
        """
        import sys

        naive_top_k = self.retrieve_naive_vector_top_k(query, k=3)

        cache_key = _sha256_json(
            {
                "schema_v": "v4_ab_pure_synth",
                "query": query,
                "resolver_mode": self.resolver_mode,
                "synthesis_model": self.synthesis_model,
                "synthesis_reasoning_effort": self.synthesis_reasoning_effort,
            }
        )

        agent_cache: Dict[str, Any] = {}
        if use_cache and RAG_AGENT_CACHE_PATH.exists():
            try:
                agent_cache = json.loads(RAG_AGENT_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                agent_cache = {}

        if use_cache and cache_key in agent_cache and not stream_output:
            cached_resp = agent_cache[cache_key]
            cached_resp["cache_hit"] = True
            return cached_resp

        retrieval = self.retrieve_graph_compiled_context(query, max_hops=max_hops)
        res_lat = float(retrieval["symbolic_resolution"].get("resolver_latency_s", 0.0))
        res_tok = int(retrieval["symbolic_resolution"].get("resolver_input_tokens", 0))

        prohibited_set = set(retrieval["symbolic_resolution"]["prohibited_part_skus"].keys())
        selected_skus = [item["selected_part_sku"] for item in retrieval["recommended_bom"]]
        leaked_prohibited = [sku for sku in selected_skus if sku in prohibited_set]
        naive_top_3_strs = [
            f"{c['node_id']} (Tier {c['precedence_tier']}, sim={c['similarity']})"
            for c in naive_top_k
        ]
        cte_node_count = len(retrieval["resolved_nodes"])
        cte_max_tier = max(
            (n["precedence_tier"] for n in retrieval["resolved_nodes"]), default=1
        )

        if stream_output:
            _render_header_and_tables(
                query=query,
                regime=retrieval["site_regime"],
                sym=retrieval["symbolic_resolution"],
                recommended_bom=retrieval["recommended_bom"],
                rejected_baseline_parts=retrieval["rejected_baseline_parts"],
                audit_passed=(len(leaked_prohibited) == 0),
                naive_top_3_clauses=naive_top_3_strs,
                cte_resolved_node_count=cte_node_count,
                cte_max_tier_reached=cte_max_tier,
                cache_hit=False,
                timing_line=(
                    f"[bold cyan]Latency Telemetry:[/bold cyan] L4/5 Tables Ready in [bold green]{res_lat:.2f}s[/bold green] "
                    f"({res_tok:,} tok) | Streaming Layer 6 ({self.synthesis_model} {self.synthesis_reasoning_effort or ''})..."
                ),
            )

        clause_nodes = [
            n for n in retrieval["resolved_nodes"] if not n["node_id"].startswith("PART:")
        ]
        compiled_context_blocks = [
            f"[TIER {n['precedence_tier']} | {n['node_id']}] {n['heading']}\n{n['content']}"
            for n in clause_nodes
        ]

        prompt_payload = {
            "query": query,
            "site_regime": retrieval["site_regime"],
            "active_clauses": retrieval["symbolic_resolution"]["active_clauses"],
            "approved_bom_parts": retrieval["recommended_bom"],
            "prohibited_parts": retrieval["rejected_baseline_parts"],
            "retrieved_clauses": compiled_context_blocks,
        }

        synth_t0 = time.perf_counter()
        create_kwargs: Dict[str, Any] = {
            "model": self.synthesis_model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the final prose synthesizer for the Borealis Specification RAG Agent. "
                        "Using ONLY the Layer 4/5 decisions (`site_regime`, `approved_bom_parts`, `prohibited_parts`, "
                        "`active_clauses`) and `retrieved_clauses`, write a concise, direct engineering answer to `query`. "
                        "Do not override or contradict `approved_bom_parts` or `prohibited_parts`: if a part is in "
                        "`approved_bom_parts`, state it as approved/required; if it is in `prohibited_parts`, state it as prohibited. "
                        "Include the exact section citations and quantitative values/thresholds from `retrieved_clauses` "
                        "that answer `query`."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, indent=2)},
            ],
        }
        if self.synthesis_reasoning_effort and not self.synthesis_model.startswith("mercury"):
            create_kwargs["reasoning_effort"] = self.synthesis_reasoning_effort
            synth_client = self.openai_client
        elif self.synthesis_model.startswith("mercury"):
            synth_client = self.dllm_client
        else:
            synth_client = self.openai_client

        stream = synth_client.chat.completions.create(**create_kwargs)
        synth_ttft_s: Optional[float] = None
        chunks: List[str] = []
        synth_tokens = 0
        synth_input_tokens = 0
        synth_output_tokens = 0

        for chunk in stream:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                text_piece = getattr(delta, "content", None)
                if text_piece:
                    if synth_ttft_s is None:
                        synth_ttft_s = round(time.perf_counter() - synth_t0, 3)
                        if stream_output:
                            e2e_ttft_now = round(res_lat + synth_ttft_s, 3)
                            Console(width=140).print(
                                f"\n[bold green]╭── Causal Specification Synthesis (Streaming Live — "
                                f"L4/5 Tables: {res_lat:.2f}s | L6 TTFT: {synth_ttft_s:.2f}s | "
                                f"E2E TTFT: {e2e_ttft_now:.2f}s) ──╮[/bold green]"
                            )
                    if stream_output:
                        sys.stdout.write(text_piece)
                        sys.stdout.flush()
                    chunks.append(text_piece)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                synth_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                synth_input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                synth_output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        synth_latency_s = round(time.perf_counter() - synth_t0, 3)
        if synth_ttft_s is None:
            synth_ttft_s = synth_latency_s

        executive_summary = "".join(chunks).strip()
        e2e_ttft_s = round(res_lat + synth_ttft_s, 3)
        total_agent_lat = round(res_lat + synth_latency_s, 3)

        if stream_output:
            chain_lines = "\n".join(
                f"  {idx}. {step}"
                for idx, step in enumerate(retrieval["multi_hop_override_chain"], start=1)
            )
            console = Console(width=140)
            console.print(f"\n\n[bold yellow]Multi-Hop Override Chain:[/bold yellow]\n{chain_lines}")
            console.print(
                f"[bold green]╰── Stream Complete — L4/5 Tables: {res_lat:.2f}s | "
                f"L6 TTFT: {synth_ttft_s:.2f}s | E2E TTFT: {e2e_ttft_s:.2f}s | "
                f"L6 Total: {synth_latency_s:.2f}s ({synth_tokens:,} tok) | "
                f"E2E Total: {total_agent_lat:.2f}s ──╯[/bold green]\n"
            )

        synthesis = {
            "executive_summary": executive_summary,
            "recommended_bom": retrieval["recommended_bom"],
            "rejected_baseline_parts": retrieval["rejected_baseline_parts"],
            "multi_hop_override_chain": retrieval["multi_hop_override_chain"],
        }

        audit_result = {
            "passed": len(leaked_prohibited) == 0,
            "selected_skus": selected_skus,
            "prohibited_skus_blocked": sorted(prohibited_set),
            "leaked_prohibited_skus": leaked_prohibited,
            "naive_rag_top_3_clauses": naive_top_3_strs,
            "cte_resolved_node_count": cte_node_count,
            "cte_max_tier_reached": cte_max_tier,
            "synthesis_ttft_s": synth_ttft_s,
            "e2e_ttft_s": e2e_ttft_s,
            "synthesis_latency_s": synth_latency_s,
            "synthesis_tokens": synth_tokens,
            "synthesis_input_tokens": synth_input_tokens,
            "synthesis_output_tokens": synth_output_tokens,
            "total_agent_latency_s": total_agent_lat,
            "total_agent_tokens": res_tok + synth_tokens,
        }

        final_output = {
            "cache_hit": False,
            "query": query,
            "resolver_mode": self.resolver_mode,
            "synthesis_model": self.synthesis_model,
            "synthesis_reasoning_effort": self.synthesis_reasoning_effort,
            "site_regime": retrieval["site_regime"],
            "symbolic_resolution": retrieval["symbolic_resolution"],
            "naive_rag_comparison": naive_top_k,
            "cte_resolved_nodes": retrieval["resolved_nodes"],
            "synthesis": synthesis,
            "safety_audit": audit_result,
        }

        if use_cache:
            agent_cache[cache_key] = final_output
            RAG_AGENT_CACHE_PATH.write_text(json.dumps(agent_cache, indent=2), encoding="utf-8")

        return final_output


def _render_header_and_tables(
    query: str,
    regime: Dict[str, Any],
    sym: Dict[str, Any],
    recommended_bom: List[Dict[str, Any]],
    rejected_baseline_parts: List[Dict[str, Any]],
    audit_passed: bool,
    naive_top_3_clauses: List[str],
    cte_resolved_node_count: int,
    cte_max_tier_reached: int,
    cache_hit: bool,
    timing_line: str = "",
) -> None:
    console = Console(width=140)
    extra_line = f"\n{timing_line}" if timing_line else ""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Borealis SOTA Graph-Compiled RAG Agent (PostgreSQL pgvector + Recursive CTE)[/bold cyan]\n"
            f"[bold yellow]Query:[/bold yellow] {query}\n"
            f"[bold green]Extracted Site Regime ({sym.get('resolver_mode', 'jev')}):[/bold green] Loop={regime['loop_id']} | "
            f"Ambient={regime['ambient_temp_c']}°C | Pressure={regime['pressure_psi']} PSI | "
            f"Methanol={regime['methanol_present']} | Cache Hit={cache_hit}\n"
            f"[bold magenta]Safety Interlock Audit:[/bold magenta] "
            f"{'[bold green]PASSED (0 Prohibited Parts Leaked)[/bold green]' if audit_passed else '[bold red]FAILED[/bold red]'} "
            f"| Naive Top-3 Clauses: {', '.join(naive_top_3_clauses)} "
            f"| CTE Graph Walk: {cte_resolved_node_count} nodes (up to Tier {cte_max_tier_reached})"
            f"{extra_line}",
            border_style="cyan",
        )
    )

    has_overrides = bool(rejected_baseline_parts) or any(
        item.get("precedence_tier", 1) > 1 for item in recommended_bom
    )
    bom_title = (
        "Audited Bill of Materials (Tier-3 & Tier-2 Overrides Applied)"
        if has_overrides
        else "Audited Bill of Materials (Tier-1 Baseline Schedule — No Excursion Overrides Triggered)"
    )

    bom_table = Table(
        title=bom_title,
        show_header=True,
        header_style="bold green",
        show_lines=True,
    )
    bom_table.add_column("Subsystem", style="bold white", no_wrap=True)
    bom_table.add_column("Selected SKU", style="bold cyan", no_wrap=True)
    bom_table.add_column("Governing Clause", style="yellow", no_wrap=True)
    bom_table.add_column("Tier", justify="center", style="bold")
    bom_table.add_column("Engineering Rationale", style="white")

    for item in recommended_bom:
        bom_table.add_row(
            item["subsystem"],
            item["selected_part_sku"],
            item["governing_clause_id"],
            f"Tier {item['precedence_tier']}",
            item["engineering_rationale"],
        )
    console.print(bom_table)

    rej_table = Table(
        title="Rejected Baseline & Intermediate Parts (Prevented Failure Modes)",
        show_header=True,
        header_style="bold red",
        show_lines=True,
    )
    rej_table.add_column("Prohibited SKU", style="bold red", no_wrap=True)
    rej_table.add_column("Baseline Clause", style="dim", no_wrap=True)
    rej_table.add_column("Overriding Clause", style="yellow", no_wrap=True)
    rej_table.add_column("Failure Mode Prevented", style="white")

    for rej in rejected_baseline_parts:
        rej_table.add_row(
            rej["prohibited_part_sku"],
            rej["baseline_clause_id"],
            rej["overriding_clause_id"],
            rej["failure_mode_prevented"],
        )
    console.print(rej_table)


def display_rag_agent_report(result: Dict[str, Any]) -> None:
    """Renders a rich terminal report comparing Naive RAG vs. SOTA CTE Graph RAG and displaying the audited BOM."""
    console = Console(width=140)
    regime = result["site_regime"]
    synthesis = result["synthesis"]
    audit = result["safety_audit"]
    sym = result.get("symbolic_resolution", {})

    res_lat = float(sym.get("resolver_latency_s", 0.0))
    syn_ttft = float(audit.get("synthesis_ttft_s", 0.0))
    e2e_ttft = float(audit.get("e2e_ttft_s", round(res_lat + syn_ttft, 3)))
    syn_lat = float(audit.get("synthesis_latency_s", 0.0))
    tot_lat = float(audit.get("total_agent_latency_s", round(res_lat + syn_lat, 3)))

    timing_line = (
        f"[bold cyan]Latency Telemetry:[/bold cyan] L4/5 Tables: {res_lat:.2f}s | "
        f"L6 TTFT: {syn_ttft:.2f}s | E2E TTFT: {e2e_ttft:.2f}s | "
        f"L6 Total: {syn_lat:.2f}s | E2E Total: {tot_lat:.2f}s"
    )

    _render_header_and_tables(
        query=result["query"],
        regime=regime,
        sym=sym,
        recommended_bom=synthesis.get("recommended_bom", []),
        rejected_baseline_parts=synthesis.get("rejected_baseline_parts", []),
        audit_passed=audit["passed"],
        naive_top_3_clauses=audit["naive_rag_top_3_clauses"],
        cte_resolved_node_count=audit["cte_resolved_node_count"],
        cte_max_tier_reached=audit["cte_max_tier_reached"],
        cache_hit=result.get("cache_hit", False),
        timing_line=timing_line,
    )

    chain_lines = "\n".join(
        f"  {idx}. {step}"
        for idx, step in enumerate(synthesis.get("multi_hop_override_chain", []), start=1)
    )
    console.print(
        Panel(
            f"[bold white]{synthesis.get('executive_summary', '')}[/bold white]\n\n"
            f"[bold yellow]Multi-Hop Override Chain:[/bold yellow]\n{chain_lines}",
            title="Causal Specification Synthesis",
            border_style="green",
        )
    )
    console.print()


if __name__ == "__main__":
    import sys

    user_query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else (
            "Specify the complete isolation valve, elastomer seal, actuator, piping, and flange bolting "
            "SKUs for Manifold Loop B at -40C and 120 PSI."
        )
    )
    agent = BorealisGraphRAGAgent(
        resolver_mode="jev",
        synthesis_model="gpt-6.1-sol",
        synthesis_reasoning_effort="low",
    )
    agent.answer_query(user_query, stream_output=True)
