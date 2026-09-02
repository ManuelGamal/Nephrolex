"""Draw a blind, stratified sample of corpus chunks to write held-out questions from.

Every question in the existing gold set was written question-first: someone thought of
a clinical question and then went looking for the chunk that answers it. That direction
has a bias you cannot audit away - the questions you think of are the ones the corpus
made easy to think of - and it is how the gold set ended up pinning one chunk per
question while three correct answers scored zero.

This inverts it. A chunk is drawn at random, its text is read, and a question is written
that the chunk answers. The gold chunk is therefore fixed *before* any retrieval runs,
so the answer key cannot be fitted to the system's output even accidentally.

For the sample to mean anything it must be blind to retrievability:

  - fixed seed, so the draw is reproducible and cannot be re-rolled until it flatters
  - stratified by document and chunk type, so the set is not quietly all table rows
  - drawn and written to disk in one step, before any question is written

The one judgement allowed afterwards is dropping a chunk that contains no clinical fact
a question could have a definite answer from - a reference-list fragment, a bare figure
caption, a sentence that only points elsewhere. That criterion is about the chunk's
content and is independent of whether retrieval would find it. Drops are recorded with
a reason rather than silently skipped.

Usage:
    python scripts/sample_for_questions.py --n 30 --seed 20260819 --out reports/question_sample.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.paths import CORPUS, GOLD_SET  # noqa: E402

# Proportional to the corpus, but figure_body and figure_caption are floored so the
# hard cases are represented rather than rounded away.
STRATA_WEIGHTS = {
    "recommendation_atom": 0.25,
    "practice_point": 0.25,
    "section_passage": 0.25,
    "table_row": 0.15,
    "figure_caption": 0.05,
    "figure_body": 0.05,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--chars", type=int, default=700)
    parser.add_argument("--also-exclude", action="append", default=[],
                        help="Further gold files whose pinned chunks are already spoken for.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "question_sample.json"))
    args = parser.parse_args()

    with CORPUS.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Exclude chunks already pinned anywhere in the gold set: this is meant to be a
    # held-out set, and reusing a chunk the existing questions already target would
    # measure the same thing twice.
    pinned: set[str] = set()
    sources = [GOLD_SET] + [Path(x) for x in args.also_exclude]
    for path in sources:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pinned |= set(json.loads(line)["relevance"])

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["id"] in pinned:
            continue
        doc = "kdigo" if record["id"].startswith("kdigo") else "nice"
        buckets[(doc, record["metadata"].get("chunk_type", "?"))].append(record)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    # NICE is 20% of the corpus; hold it to roughly that so the set is not all KDIGO.
    for doc, doc_share in (("kdigo", 0.8), ("nice", 0.2)):
        for chunk_type, weight in STRATA_WEIGHTS.items():
            pool = buckets.get((doc, chunk_type), [])
            take = round(args.n * doc_share * weight)
            if not pool or not take:
                continue
            sample.extend(rng.sample(pool, min(take, len(pool))))

    rng.shuffle(sample)
    print(f"seed {args.seed}, {len(sample)} chunks drawn "
          f"({sum(1 for s in sample if s['id'].startswith('kdigo'))} KDIGO / "
          f"{sum(1 for s in sample if s['id'].startswith('nice'))} NICE)\n")

    for i, record in enumerate(sample, 1):
        meta = record["metadata"]
        text = " ".join(record["raw_text"].split())
        print(f"--- {i}. {record['id']}")
        print(f"    type={meta.get('chunk_type')} page={meta.get('page_start')}"
              f"-{meta.get('page_end')}")
        print(textwrap.fill(text[: args.chars], 96, initial_indent="    ",
                            subsequent_indent="    "))
        print()

    Path(args.out).write_text(json.dumps(
        [{"id": r["id"], "chunk_type": r["metadata"].get("chunk_type"),
          "page_start": r["metadata"].get("page_start"),
          "text": " ".join(r["raw_text"].split())} for r in sample],
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
