"""Evaluate the clinician-FAQ set on the metric a lookup task is actually judged by.

Why not just nDCG
-----------------
The FAQ gold is pinned by sweeping the corpus for every chunk that states the answer,
which means both printings of a KDIGO recommendation and both guidelines where they
agree. That is the right answer key - those chunks really do answer the question - but
it makes nDCG@10 measure "found every printing", not "answered the question". A question
with four pinned chunks cannot score 1.0 unless all four reach the top ten, and the
second printing of the G5 row adds nothing to a clinician who has already seen the
first.

For a point-of-care lookup the question is whether a correct, citable chunk appeared
high enough to be read. That is success@k, and it is reported here as the primary
figure with nDCG@10 alongside it rather than instead of it.

Both are printed every run. Reporting only the flattering one is the failure this
project has spent its evaluation work avoiding.

Usage:
    python scripts/evaluate_faq.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case  # noqa: E402
from nephrolex.retrieval import retrieve as R  # noqa: E402

GOLD = ROOT / "eval" / "faq_clinical.jsonl"
DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "reports" / "faq_eval.json"))
    args = parser.parse_args()

    with Path(args.gold).open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    rows, first_ranks = [], {}
    for case in cases:
        results = R.retrieve(case["query"], args.top_k, DENSE_MODEL, RERANKER)
        rows.append(evaluate_case(case, results))
        gold = set(case["relevance"])
        first_ranks[case["id"]] = next(
            (i for i, r in enumerate(results, 1) if r["id"] in gold), None
        )

    n = len(cases)
    success = {
        k: sum(1 for rank in first_ranks.values() if rank and rank <= k)
        for k in (1, 3, 5, 10)
    }
    stats = aggregate(rows)

    print(f"{n} clinician-FAQ questions\n")
    print("primary - did a correct, citable chunk appear high enough to read?")
    for k in (1, 3, 5, 10):
        print(f"  success@{k:<3} {success[k]}/{n} = {success[k] / n:.0%}")
    print("\nsecondary - ranked quality against an answer key that pins every printing")
    for key in ("ndcg_at_10", "recall_at_10", "recall_at_20", "mrr_at_10"):
        print(f"  {key:<14} {stats[key]:.4f}")

    misses = [cid for cid, rank in first_ranks.items() if not rank]
    if misses:
        print(f"\nno correct chunk in the top {args.top_k} ({len(misses)}):")
        for cid in misses:
            print(f"  {cid}")

    Path(args.out).write_text(json.dumps({
        "questions": n,
        "success_at": {str(k): success[k] for k in success},
        "success_rate": {str(k): round(success[k] / n, 4) for k in success},
        "ranked": stats,
        "first_rank": first_ranks,
        "misses": misses,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
