from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.retrieval.retrieve import classify_query, retrieve


ROOT = Path(__file__).resolve().parents[1]
OUT_OF_SCOPE_TOPICS = {"pregnancy", "pediatric", "dialysis"}


def safety_decision(query: str, results: list[dict]) -> dict:
    query_topics = set(classify_query(query))
    result_topics = set()
    for result in results:
        result_topics.update(result["metadata"].get("topics", []))
    out_of_scope = query_topics & OUT_OF_SCOPE_TOPICS
    if out_of_scope:
        return {
            "action": "refuse_with_evidence",
            "reason": f"Query touches excluded or special-population scope: {', '.join(sorted(out_of_scope))}.",
        }
    if not results:
        return {"action": "refuse_no_evidence", "reason": "No supporting guideline chunks were retrieved."}
    if not any(result["metadata"].get("citable", True) for result in results[:5]):
        return {"action": "refuse_no_citable_evidence", "reason": "Top evidence is non-citable routing context only."}
    return {"action": "answer_with_citations", "reason": "Retrieved citable evidence is available."}


def make_generation_prompt(query: str, evidence: list[dict], decision: dict) -> str:
    evidence_lines = []
    for idx, item in enumerate(evidence, start=1):
        citation = item["metadata"]["citation"]
        evidence_lines.append(
            f"[E{idx}] {citation['document']} | {citation['section']} | pages {citation['page_start']}-{citation['page_end']}\n"
            f"{item['raw_text']}"
        )
    return (
        "You are a clinical guideline evidence assistant for CKD.\n"
        "Use only the evidence below. Do not use outside medical knowledge.\n"
        "Every clinical claim must cite at least one evidence id like [E1].\n"
        "If the safety decision is not answer_with_citations, refuse briefly and explain the scope boundary using the retrieved evidence.\n\n"
        f"User query: {query}\n"
        f"Safety decision: {decision['action']} - {decision['reason']}\n\n"
        "Evidence:\n"
        + "\n\n".join(evidence_lines)
        + "\n\nRequired answer format:\n"
        "1. Recommendation or refusal\n"
        "2. Supporting evidence\n"
        "3. Citations\n"
        "4. What is not covered by the retrieved evidence\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a citable evidence pack and strict generation prompt.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = retrieve(args.query, args.top_k, args.dense_model, args.reranker_model)
    decision = safety_decision(args.query, results)
    prompt = make_generation_prompt(args.query, results, decision)
    pack = {
        "query": args.query,
        "safety_decision": decision,
        "evidence": results,
        "generation_prompt": prompt,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(pack, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
