"""For every question that scores nDCG 0, show what the system actually returned.

A total miss is recorded as "retrieval failed". It has two causes that look identical
in the metric and need opposite fixes:

    real miss        nothing relevant was returned. Fix retrieval.
    annotation gap   something correct was returned and is not in the gold set. Fix
                     the gold set - and until it is fixed, every technique measured
                     against these questions is being scored on an answer key that
                     marks correct answers wrong.

This was not hypothetical. `def_paraphrase` ("what has to be true for someone to
actually count as having long-term kidney disease rather than just one bad blood
test?") pins a single NICE passage, while the system's #1 result is KDIGO Practice
Point 1.1.3.2, "do not assume chronicity based upon a single abnormal level for eGFR
and ACR" - a direct answer to the question, unannotated, scoring nothing. That case was
also counted as a HyDE regression, so a retrieval technique was being judged partly on
an annotation gap.

Judging is left to a human on purpose. This prints the evidence and pins nothing.

Usage:
    python scripts/audit_total_misses.py
    python scripts/audit_total_misses.py --show 5 --chars 200
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.evaluation.evaluate_gold import evaluate_case  # noqa: E402
from newbieduo.paths import GOLD_SET, REPORTS  # noqa: E402
from newbieduo.retrieval import retrieve as R  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=3, help="Results to print per miss.")
    parser.add_argument("--chars", type=int, default=170)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Audit every case at or below this nDCG, not only zeros.")
    parser.add_argument("--out", default=str(REPORTS / "total_miss_audit.json"))
    args = parser.parse_args()

    with GOLD_SET.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f
                 if line.strip() and not json.loads(line)["should_abstain"]]

    records = {r["id"]: r for r in R.load_indexes().records}
    misses, rows = [], []

    for case in cases:
        results = R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER)
        row = evaluate_case(case, results)
        if row["ndcg_at_10"] > args.threshold:
            continue
        misses.append((case, results, row))

    print(f"{len(misses)} of {len(cases)} answerable questions score nDCG <= {args.threshold}\n")

    for case, results, row in misses:
        gold = {cid: g for cid, g in case["relevance"].items() if g >= 1}
        print("=" * 78)
        print(f"{case['id']}  [{case['kind']}/{case['category']}]  nDCG {row['ndcg_at_10']:.3f}")
        print(f"Q: {case['query']}")
        print(f"\nannotated as relevant ({len(gold)}):")
        for cid, grade in sorted(gold.items(), key=lambda kv: -kv[1]):
            record = records.get(cid)
            text = " ".join(record["raw_text"].split()) if record else "(chunk not in corpus!)"
            print(f"  grade {grade}  {cid[:66]}")
            print(f"           {textwrap.shorten(text, args.chars)}")

        print(f"\nwhat the system returned (top {args.show}):")
        for i, result in enumerate(results[:args.show], 1):
            mark = f"grade {gold[result['id']]}" if result["id"] in gold else "UNANNOTATED"
            text = " ".join(result["raw_text"].split())
            print(f"  {i}. [{mark}] {result['id'][:60]}")
            print(f"     {textwrap.shorten(text, args.chars)}")
        print()

        rows.append({
            "id": case["id"], "kind": case["kind"], "category": case["category"],
            "query": case["query"], "ndcg_at_10": row["ndcg_at_10"],
            "gold": list(gold), "returned": [r["id"] for r in results[:args.show]],
        })

    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    print("\nFor each: is the top result a correct answer that is simply not annotated?\n"
          "If yes the gold set needs the chunk added - not the retriever changed.")


if __name__ == "__main__":
    main()
