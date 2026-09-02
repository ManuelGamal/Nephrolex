"""Enumerate every corpus chunk matching a clinical concept, for blind re-annotation.

The gold set was built by pinning *the* answer to each question - typically one to
three chunks. Anything else that answers the question scores zero, so the metric
penalises a correct retrieval and, worse, penalises any technique that surfaces a
correct-but-unpinned chunk. Measured on the paraphrase slice: 3 of 7 total misses were
this, not retrieval failure.

The unsafe way to fix that is to look at what the system returned and annotate the
good ones. That fits the answer key to the system and inflates every number after it.

This tool exists to make the safe way practical. Candidates come from regex sweeps
over the whole corpus, expressed in terms of the *clinical concept* the question asks
about, with no reference to any ranking. Every match is printed - whether or not the
retriever ever returned it - and judged against the guideline text. Chunks the system
misses are found by the same sweep as chunks it finds, so the procedure can lower the
score as easily as raise it.

Scope must also be chosen before results are seen. Re-annotating only the questions
that scored badly is the same bias wearing a different hat; take a whole slice.

Usage:
    python scripts/annotate_candidates.py --case stag_g1_paraphrase \\
        --pattern "GFR category: G1" --pattern "normal or high"
    python scripts/annotate_candidates.py --case bp_target_paraphrase \\
        --pattern "systolic blood pressure" --pattern "SBP" --max 40
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

from nephrolex.paths import CORPUS, GOLD_SET  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--pattern", action="append", required=True,
                        help="Case-insensitive regex. Repeatable; a chunk matching ANY is shown.")
    parser.add_argument("--require", action="append", default=[],
                        help="Regex a chunk must ALSO match. Repeatable; all must match.")
    parser.add_argument("--max", type=int, default=25)
    parser.add_argument("--chars", type=int, default=260)
    args = parser.parse_args()

    case = next(
        (json.loads(line) for line in GOLD_SET.open("r", encoding="utf-8")
         if line.strip() and json.loads(line)["id"] == args.case),
        None,
    )
    if case is None:
        raise SystemExit(f"no gold case {args.case!r}")

    gold = case["relevance"]
    patterns = [re.compile(p, re.I) for p in args.pattern]
    required = [re.compile(p, re.I) for p in args.require]

    hits = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = " ".join(record["raw_text"].split())
            if not any(p.search(text) for p in patterns):
                continue
            if not all(p.search(text) for p in required):
                continue
            hits.append((record["id"], text, record.get("metadata", {})))

    print(f"{case['id']}  [{case['kind']}/{case['category']}]")
    print(f"Q: {case['query']}")
    print(f"currently pinned: {len(gold)} chunk(s)\n")
    print(f"{len(hits)} corpus chunks match; showing {min(len(hits), args.max)}\n")

    for chunk_id, text, meta in hits[: args.max]:
        grade = gold.get(chunk_id)
        mark = f"PINNED grade {grade}" if grade is not None else "unpinned"
        print(f"[{mark}] {chunk_id}")
        print(f"    type={meta.get('chunk_type')} page={meta.get('page_start')}")
        print(f"    {textwrap.shorten(text, args.chars)}\n")

    if len(hits) > args.max:
        print(f"... {len(hits) - args.max} more. Raise --max or tighten --require.")


if __name__ == "__main__":
    main()
