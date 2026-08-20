"""Embed the corpus with MedCPT's article encoder, writing only the vector file.

Deliberately separate from `build_indexes.py`. That script also rewrites
`records.json` and `tfidf.pkl` at fixed paths, and its guard against clobbering only
fires when `--corpus` is not the shipped one - which it is here. Rebuilding the lexical
index as a side effect of adding a dense variant has already cost this project two
silently-wrong baselines, so this writes the .npy and its sidecar and touches nothing
else.

The text is produced by the same `indexable_text(record, "raw+section")` every other
embedding variant uses, so the comparison is between encoders and not between inputs.

Usage:
    python scripts/build_medcpt_vectors.py
    python scripts/build_medcpt_vectors.py --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.ingestion.build_indexes import indexable_text  # noqa: E402
from newbieduo.paths import CORPUS  # noqa: E402
from newbieduo.retrieval.medcpt import ARTICLE_MODEL, LOGICAL_NAME, MedCPTEncoder  # noqa: E402

OUT = ROOT / "data" / "indexes" / "dense" / "ncbi__MedCPT.npy"
TEXT_MODE = "raw+section"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    with CORPUS.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    texts = [indexable_text(record, TEXT_MODE) for record in records]
    print(f"{len(texts)} chunks -> MedCPT article encoder")

    encoder = MedCPTEncoder("article")
    started = time.time()
    vectors = encoder.encode(texts, normalize_embeddings=True,
                             batch_size=args.batch_size, show_progress_bar=True)
    elapsed = time.time() - started

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)
    out.with_suffix(".meta.json").write_text(json.dumps({
        # The logical pair name, because retrieve() compares this against the
        # --dense-model it was given and the query side is a different checkpoint.
        "model_name": LOGICAL_NAME,
        "article_encoder": ARTICLE_MODEL,
        "text_mode": TEXT_MODE,
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "cls",
        "device": encoder.device,
        "shape": list(vectors.shape),
        "encode_seconds": round(elapsed, 1),
        "chunks_per_second": round(len(texts) / elapsed, 1) if elapsed else None,
    }, indent=2), encoding="utf-8")

    print(f"wrote {out}  {vectors.shape}  in {elapsed:.0f}s "
          f"({len(texts) / elapsed:.1f} chunks/s)")


if __name__ == "__main__":
    main()
