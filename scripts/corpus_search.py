"""Ad-hoc corpus grep used to pin gold chunk IDs for the evaluation set.

Usage:
    python scripts/corpus_search.py "regex" [--doc kdigo_2024_ckd] [--type recommendation_atom] [--limit 12]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "chunks" / "retrieval_corpus.jsonl"


def load() -> list[dict]:
    with CORPUS.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--doc", default=None)
    parser.add_argument("--type", dest="chunk_type", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--width", type=int, default=150)
    args = parser.parse_args()

    rx = re.compile(args.pattern, re.I)
    shown = 0
    for record in load():
        meta = record["metadata"]
        if args.doc and meta.get("document_id") != args.doc:
            continue
        if args.chunk_type and meta.get("chunk_type") != args.chunk_type:
            continue
        if not rx.search(record["raw_text"]):
            continue
        text = " ".join(record["raw_text"].split())
        print(
            f"[{meta.get('chunk_type')}] p{meta.get('page_start')} {record['id']}\n"
            f"    {text[: args.width]}"
        )
        shown += 1
        if shown >= args.limit:
            break
    print(f"-- {shown} shown")


if __name__ == "__main__":
    main()
