"""
Borealis Golden QA Loader & Holistic LLM-as-a-Judge (`evaluate_rag.py`)
-----------------------------------------------------------------------
Uses `gpt-6.1-sol` with `reasoning_effort="high"` to holistically evaluate
whether the candidate's text response answers the user's question accurately
compared to the reference `expected_answer`.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-6.1-sol")
JUDGE_REASONING_EFFORT = os.environ.get("EVAL_REASONING_EFFORT", "high")

CACHE_DIR = PROJECT_ROOT / ".cache" / "borealis"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_CACHE_PATH = CACHE_DIR / "llm_judge_holistic_cache.json"


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_golden_qa(
    json_path: Union[str, Path] = "borealis_golden_qa.json",
) -> List[Dict[str, Any]]:
    """Loads the Borealis Golden QA dataset items from disk."""
    path = Path(json_path)
    if not path.exists():
        for candidate in [
            Path(__file__).parent / path.name,
            PROJECT_ROOT / "evals" / path.name,
            PROJECT_ROOT / "starter" / path.name,
        ]:
            if candidate.exists():
                path = candidate
                break
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


HOLISTIC_JUDGE_SCHEMA: Dict[str, Any] = {
    "name": "holistic_qa_judge_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the candidate answer accurately and completely answers the user's question in agreement with the reference answer.",
            },
            "score": {
                "type": "number",
                "description": "Holistic quality and accuracy score from 0.0 (completely wrong) to 1.0 (fully accurate and complete).",
            },
            "judge_rationale": {
                "type": "string",
                "description": "Explanation of why the candidate answer succeeded or failed to answer the user's question.",
            },
            "failure_reasons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Empty list if success is True; otherwise specific errors, omissions, or contradictions in the candidate answer.",
            },
        },
        "required": [
            "success",
            "score",
            "judge_rationale",
            "failure_reasons",
        ],
        "additionalProperties": False,
    },
}


def _extract_answer_text(y_hat: Union[str, Dict[str, Any]]) -> str:
    if isinstance(y_hat, str):
        return y_hat
    if isinstance(y_hat, dict):
        if "answer_text" in y_hat and isinstance(y_hat["answer_text"], str):
            return y_hat["answer_text"]
        if "synthesis" in y_hat and isinstance(y_hat["synthesis"], dict):
            syn = y_hat["synthesis"]
            if "answer_text" in syn and isinstance(syn["answer_text"], str):
                return syn["answer_text"]
            if "executive_summary" in syn and isinstance(syn["executive_summary"], str):
                return syn["executive_summary"]
        return json.dumps(y_hat)
    return str(y_hat)


def judge_ground_truth(
    y: Dict[str, Any],
    y_hat: Union[str, Dict[str, Any]],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Holistic LLM-as-a-Judge (`gpt-6.1-sol`, `reasoning_effort="high"`):
    Grades only the candidate text answer against the user's question and reference answer.
    """
    question = y.get("question", "")
    expected_answer = y.get("expected_answer", "")
    candidate_answer = _extract_answer_text(y_hat)

    cache_key = _sha256_json(
        {
            "judge_model": JUDGE_MODEL,
            "reasoning_effort": JUDGE_REASONING_EFFORT,
            "question": question,
            "expected_answer": expected_answer,
            "candidate_answer": candidate_answer,
        }
    )

    judge_cache: Dict[str, Any] = {}
    if use_cache and JUDGE_CACHE_PATH.exists():
        try:
            judge_cache = json.loads(JUDGE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            judge_cache = {}

    if use_cache and cache_key in judge_cache:
        cached = judge_cache[cache_key]
        cached["cache_hit"] = True
        return cached

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    system_prompt = (
        "You are an expert evaluator grading a question-answering system. "
        "Given the user's question, the reference answer, and the candidate text response, "
        "evaluate holistically whether the candidate response accurately and completely answers "
        "the user's question without contradicting the reference answer."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{expected_answer}\n\n"
        f"Candidate Answer:\n{candidate_answer}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": HOLISTIC_JUDGE_SCHEMA,
        },
    )

    verdict = json.loads(response.choices[0].message.content or "{}")

    result = {
        "success": verdict["success"],
        "score": round(float(verdict["score"]), 4),
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "judge_rationale": verdict["judge_rationale"],
        "failure_reasons": verdict["failure_reasons"],
        "cache_hit": False,
    }

    if use_cache:
        judge_cache[cache_key] = result
        JUDGE_CACHE_PATH.write_text(json.dumps(judge_cache, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    qa_items = load_golden_qa()
    print(
        f"Loaded {len(qa_items)} Golden QA items. Running holistic LLM Judge ({JUDGE_MODEL}, reasoning_effort={JUDGE_REASONING_EFFORT})...\n"
    )
    failures = []
    for i,q in enumerate(qa_items):
        res_graph = judge_ground_truth(q, q["expected_answer"])
        if not res_graph['success']:
            failures.append(res_graph)

        # print(f"\n=== {i+1}. Reference / Graph-Compiled CTE Verdict ===")
        # print(json.dumps(res_graph, indent=2))
    
    print(f'Got {len(qa_items)-len(failures)} of {len(qa_items)} ({100.*(len(qa_items)-len(failures))/len(qa_items)}) right.',end=' ')
    if failures:
        print('Failures: ')
    for failure in failures:
        print(json.dumps(failure,indent=2))
