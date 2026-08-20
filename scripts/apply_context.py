"""Fold generated chunk context into the retrieval corpus.

Separate from generation so the expensive LLM pass runs once and the corpus can be
rebuilt, re-indexed and re-measured as often as needed. Writes a second corpus file
rather than overwriting, so the contextual and non-contextual variants can be
evaluated head to head - the point is to measure whether context helps, not to
assume it.

Usage:
    python scripts/apply_context.py
    python scripts/apply_context.py --out data/chunks_v2/retrieval_corpus_ctx.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "data" / "chunks_v2"
CACHE = CHUNKS_DIR / "chunk_context.jsonl"
DEFAULT_OUT = CHUNKS_DIR / "retrieval_corpus_ctx.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CHUNKS_DIR / "retrieval_corpus.jsonl"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"no generated context at {CACHE}; run contextualize_chunks.py first")

    context = {row["chunk_id"]: row["context"] for row in read_jsonl(CACHE)}
    records = read_jsonl(Path(args.corpus))

    applied = 0
    for record in records:
        situating = context.get(record["id"])
        if not situating:
            continue
        # Stored as its own field and deliberately NOT prepended to raw_text.
        #
        # An earlier version did prepend it, which would have put generated prose into
        # four places it must never reach: the text quoted to the user as verbatim
        # guideline wording, the evidence the answer verifier checks numbers against
        # (weakening the one guarantee that does not rely on the model's cooperation),
        # the text compared to the PDF for citation accuracy, and the cross-encoder's
        # limited token budget. Retrieval would have improved by making grounding less
        # trustworthy.
        #
        # build_indexes.py composes the indexed text rather than storing it, so the
        # context reaches the embedding and the lexical indexes from here, while
        # raw_text stays exactly what the PDF says.
        record["context"] = situating
        record["metadata"]["has_context"] = True
        applied += 1

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{applied}/{len(records)} chunks given context -> {out_path}")
    if applied < len(records):
        print(f"  {len(records) - applied} chunks had no cached context (run generation again to fill)")
    print("\nNext:")
    print(f"  python scripts/build_indexes.py --corpus {out_path} --dense-model BAAI/bge-m3")
    print("  python scripts/evaluate_gold.py --tag contextual --dense-model BAAI/bge-m3 \\")
    print("      --reranker-model BAAI/bge-reranker-v2-m3 --compare reports/gold_eval_rerank_bge_v2m3.json")


if __name__ == "__main__":
    main()
