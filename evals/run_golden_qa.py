"""
Runs the Borealis Golden QA A/B Evaluation Set (`evals/borealis_golden_qa.json`)
comparing:
  - Arm A (`dllm_forest`): Layer 4/5 = Forest of Parallel `mercury-2` dLLMs + Layer 6 = `gpt-6.1-sol (low)`
  - Arm B (`jev`): Layer 4/5 = TypeSafe System One (`jev-latest`) + Layer 6 = `gpt-6.1-sol (low)`
Captures the rendered text from `display_rag_agent_report` and grades each answer holistically
using `judge_ground_truth` (`gpt-6.1-sol`, `reasoning_effort="high"`).
"""

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from borealis_dagster.rag_agent import (  # noqa: E402
    BorealisGraphRAGAgent,
    display_rag_agent_report,
)
from evals.evaluate_rag import (  # noqa: E402
    JUDGE_MODEL,
    JUDGE_REASONING_EFFORT,
    judge_ground_truth,
    load_golden_qa,
)

RESULTS_OUTPUT_PATH = PROJECT_ROOT / "evals" / "golden_qa_eval_results.json"


def render_agent_report_text(agent_res: Dict[str, Any]) -> str:
    """Captures the text output produced by `display_rag_agent_report(agent_res)`."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_rag_agent_report(agent_res)
    return buf.getvalue()


def run_arm(
    console: Console,
    qa_items: List[Dict[str, Any]],
    resolver_mode: str,
    synthesis_model: str = "gpt-6.1-sol",
    synthesis_reasoning_effort: str = "low",
) -> Dict[str, Any]:
    agent = BorealisGraphRAGAgent(
        resolver_mode=resolver_mode,
        synthesis_model=synthesis_model,
        synthesis_reasoning_effort=synthesis_reasoning_effort,
    )

    console.print(
        Panel.fit(
            f"[bold cyan]Running Golden QA Arm: Layer 4/5 = {resolver_mode} | "
            f"Layer 6 = {synthesis_model} ({synthesis_reasoning_effort})[/bold cyan]\n"
            f"[yellow]Judge Model:[/yellow] {JUDGE_MODEL} (reasoning_effort={JUDGE_REASONING_EFFORT})",
            border_style="cyan",
        )
    )

    eval_records: List[Dict[str, Any]] = []
    total_resolver_latency = 0.0
    total_resolver_tokens = 0
    total_synthesis_ttft = 0.0
    total_e2e_ttft = 0.0
    total_synthesis_latency = 0.0
    total_synthesis_tokens = 0

    for item in qa_items:
        qid = item["id"]
        category = item["category"]
        question = item["question"]

        console.print(
            f"[bold blue]==> [{resolver_mode} + {synthesis_model}] Running {qid} ({category})...[/bold blue]"
        )
        try:
            agent_res = agent.answer_query(question, max_hops=4, use_cache=True)
            display_rag_agent_report(agent_res)
            candidate_text = render_agent_report_text(agent_res)
            agent_error = None
            sym = agent_res.get("symbolic_resolution", {})
            aud = agent_res.get("safety_audit", {})
            res_lat = float(sym.get("resolver_latency_s", 0.0))
            res_tok = int(sym.get("resolver_input_tokens", 0))
            syn_ttft = float(aud.get("synthesis_ttft_s", aud.get("synthesis_latency_s", 0.0)))
            e2e_ttft = round(res_lat + syn_ttft, 3)
            syn_lat = float(aud.get("synthesis_latency_s", 0.0))
            syn_tok = int(aud.get("synthesis_tokens", 0))
            total_resolver_latency += res_lat
            total_resolver_tokens += res_tok
            total_synthesis_ttft += syn_ttft
            total_e2e_ttft += e2e_ttft
            total_synthesis_latency += syn_lat
            total_synthesis_tokens += syn_tok
        except Exception as exc:
            candidate_text = f"ERROR during agent execution: {exc}"
            agent_error = str(exc)
            res_lat = 0.0
            res_tok = 0
            syn_ttft = 0.0
            e2e_ttft = 0.0
            syn_lat = 0.0
            syn_tok = 0

        verdict = judge_ground_truth(item, candidate_text, use_cache=True)

        status_str = (
            "[bold green]PASS[/bold green]" if verdict["success"] else "[bold red]FAIL[/bold red]"
        )
        console.print(
            f"    {status_str} | Score: [bold]{verdict['score']:.2f}[/bold] | "
            f"L4/5 (Tables Ready): {res_lat:.2f}s ({res_tok:,} tok) | "
            f"L6 TTFT: {syn_ttft:.2f}s (E2E TTFT: {e2e_ttft:.2f}s) | "
            f"L6 Total: {syn_lat:.2f}s ({syn_tok:,} tok) | E2E Total: {res_lat + syn_lat:.2f}s"
        )

        eval_records.append(
            {
                "id": qid,
                "category": category,
                "question": question,
                "expected_answer": item["expected_answer"],
                "candidate_answer": candidate_text,
                "agent_error": agent_error,
                "resolver_latency_s": res_lat,
                "resolver_input_tokens": res_tok,
                "synthesis_ttft_s": syn_ttft,
                "e2e_ttft_s": e2e_ttft,
                "synthesis_latency_s": syn_lat,
                "synthesis_tokens": syn_tok,
                "total_agent_latency_s": round(res_lat + syn_lat, 3),
                "total_agent_tokens": res_tok + syn_tok,
                "verdict": verdict,
            }
        )

    total = len(eval_records)
    passed = sum(1 for r in eval_records if r["verdict"]["success"])
    avg_score = sum(r["verdict"]["score"] for r in eval_records) / total if total else 0.0
    total_e2e_lat = total_resolver_latency + total_synthesis_latency
    total_e2e_tok = total_resolver_tokens + total_synthesis_tokens

    return {
        "resolver_mode": resolver_mode,
        "synthesis_model": synthesis_model,
        "synthesis_reasoning_effort": synthesis_reasoning_effort,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "total_questions": total,
        "passed_questions": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": round(avg_score, 4),
        "total_resolver_latency_s": round(total_resolver_latency, 3),
        "avg_resolver_latency_s": round(total_resolver_latency / total, 3) if total else 0.0,
        "total_resolver_input_tokens": total_resolver_tokens,
        "total_synthesis_ttft_s": round(total_synthesis_ttft, 3),
        "avg_synthesis_ttft_s": round(total_synthesis_ttft / total, 3) if total else 0.0,
        "total_e2e_ttft_s": round(total_e2e_ttft, 3),
        "avg_e2e_ttft_s": round(total_e2e_ttft / total, 3) if total else 0.0,
        "total_synthesis_latency_s": round(total_synthesis_latency, 3),
        "avg_synthesis_latency_s": round(total_synthesis_latency / total, 3) if total else 0.0,
        "total_synthesis_tokens": total_synthesis_tokens,
        "total_agent_latency_s": round(total_e2e_lat, 3),
        "avg_agent_latency_s": round(total_e2e_lat / total, 3) if total else 0.0,
        "total_agent_tokens": total_e2e_tok,
        "results": eval_records,
    }


ARM_CONFIGS = {
    "dllm_forest": ("dllm_forest", "gpt-6.1-sol", "low"),
    "jev": ("jev", "gpt-6.1-sol", "low"),
    "jev_dllm": ("jev", "mercury-2", ""),
    "jev_luna": ("jev", "gpt-6-luna", "low"),
    "jev_luna_med": ("jev", "gpt-6-luna", "medium"),
    "jev_luna_high": ("jev", "gpt-6-luna", "high"),
}


def main() -> None:
    console = Console(width=140)
    qa_items = load_golden_qa()

    arms = ["jev"]
    if "--all" in sys.argv:
        arms = ["dllm_forest", "jev", "jev_dllm", "jev_luna", "jev_luna_med"]
    elif "--only" in sys.argv:
        idx = sys.argv.index("--only")
        arms = [sys.argv[idx + 1]]

    existing: Dict[str, Any] = {}
    if RESULTS_OUTPUT_PATH.exists():
        try:
            existing = json.loads(RESULTS_OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    for arm_name in arms:
        rm, sm, sre = ARM_CONFIGS[arm_name]
        existing[arm_name] = run_arm(
            console=console,
            qa_items=qa_items,
            resolver_mode=rm,
            synthesis_model=sm,
            synthesis_reasoning_effort=sre,
        )

    RESULTS_OUTPUT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
