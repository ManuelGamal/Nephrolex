"""Correct-row precision: when the answer is one row of a table, is it the right row?

Why this metric exists
----------------------
Every other measure here treats a near miss as a small loss. This one does not, because
returning the wrong row of the right table is not a ranking error - it is a wrong answer
carrying a correct-looking citation to a real KDIGO table, and it is invisible in nDCG.

The case that prompted it: asked "how much salt a day is right for a child of eleven?",
the system ranked KDIGO Table 22's *0-6 months* row - 0.110 g/day - above the 9-to-13
year row that answers it, 1.2 g/day. A tenfold paediatric dosing error, with provenance,
that scored as an ordinary partial credit everywhere else.

Sibling table rows are the hardest possible discrimination for a cross-encoder: they
share a header, a section, a caption and a sentence structure, and differ only in their
numbers. So this is where a retrieval system should be checked hardest, not least.

What it measures
----------------
Over every gold question whose grade-2 answer is a table row, across all three answer
keys: find the highest-ranked chunk in the top 10 that belongs to the same table as the
gold row, and ask whether it *is* a gold row.

    right   the top-ranked row from that table is the one that answers the question
    wrong   a sibling row outranks it - the failure this metric exists to catch
    absent  the table never surfaced, which is a retrieval miss and counted separately

Usage:
    python scripts/check_table_rows.py
    python scripts/check_table_rows.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.paths import GOLD_SET, REPORTS  # noqa: E402
from nephrolex.retrieval import retrieve as R  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

SETS = {
    "main": str(GOLD_SET),
    "heldout": str(ROOT / "eval" / "heldout_chunkfirst.jsonl"),
    "faq": str(ROOT / "eval" / "faq_clinical.jsonl"),
}

# A row id is "<table identity>_<row number>", optionally with a collision hash.
ROW_ID = re.compile(r"^(.*_table_row_.+?)_(\d+)(?:_[0-9a-f]{6})?$")


def table_of(chunk_id: str) -> str | None:
    match = ROW_ID.match(chunk_id)
    return match.group(1) if match else None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=str(REPORTS / "table_row_precision.json"))
    args = parser.parse_args()

    records = {r["id"]: r for r in R.load_indexes().records}
    right = wrong = absent = 0
    failures, rows = [], []

    for name, path in SETS.items():
        with open(path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        for case in cases:
            if case["should_abstain"]:
                continue
            gold_rows = {c for c, grade in case["relevance"].items()
                         if grade == 2 and table_of(c)}
            if not gold_rows:
                continue
            gold_tables = {table_of(c) for c in gold_rows}
            ranked = [r["id"] for r in
                      R.retrieve(case["query"], args.top_k, DENSE_MODEL, RERANKER)]
            top = next((c for c in ranked if table_of(c) in gold_tables), None)

            if top is None:
                absent += 1
                verdict = "absent"
            elif top in gold_rows:
                right += 1
                verdict = "right"
            else:
                wrong += 1
                verdict = "wrong"
                failures.append((name, case["id"], case["query"], top, sorted(gold_rows)[0]))
            rows.append({"set": name, "id": case["id"], "verdict": verdict})

    total = right + wrong
    print(f"{total + absent} questions answered by a specific table row\n")
    print(f"  correct row on top   {right}/{total} = {right / total:.0%}")
    print(f"  sibling row on top   {wrong}/{total} = {wrong / total:.0%}")
    print(f"  table never surfaced {absent}")

    if failures:
        print("\nwrong-row cases - each is a wrong answer with a real citation:")
        for name, case_id, query, got, want in failures:
            print(f"\n  [{name}] {case_id}: {query[:64]}")
            for label, chunk in (("returned", got), ("should be", want)):
                text = " ".join(records[chunk]["raw_text"].split())
                print(f"    {label:<10} {textwrap.shorten(text, 88)}")

    Path(args.out).write_text(json.dumps({
        "top_k": args.top_k,
        "right": right, "wrong": wrong, "table_absent": absent,
        "correct_row_precision": round(right / total, 4) if total else None,
        "failures": [{"set": s, "id": i, "returned": g, "expected": w}
                     for s, i, _, g, w in failures],
        "cases": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
