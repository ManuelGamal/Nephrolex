"""Build a separate lexical index over the generated clinician-voice questions.

Separate, not concatenated. The obvious implementation is to append the generated
questions to each chunk's indexed text and rebuild one index, and that is what the
original doc2query papers do for sparse retrieval. Doc2Query++ (2025) reports that
concatenating expansions into the document harms *dense* retrieval, because the added
text drags the passage embedding away from what the passage actually says, and proposes
keeping the expansions in their own index and fusing the two.

This project can do that almost for free: `retrieve()` already fuses several
independently-normalised signals over a shared candidate pool, and `pool_from` already
exists so a source can nominate candidates. So the expansions become one more signal and
one more pool source, and the original chunk text is left exactly as it is - which also
means the citation, the quote and the verifier never see generated text. That last part
is not an optimisation, it is the safety property: a hallucinated question can cost
ranking and can never reach an answer.

Alignment is by chunk id, and every chunk gets a row - an empty string where no
expansion was generated - so the matrix lines up with `records.json` positionally and a
missing expansion contributes zero rather than shifting every index after it.

Usage:
    python scripts/build_doc2query_index.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.paths import CORPUS, DATA  # noqa: E402

CACHE = DATA / "doc2query.jsonl"
OUT = ROOT / "data" / "indexes" / "doc2query_tfidf.pkl"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"no expansions at {CACHE}. Run scripts/build_doc2query.py first.")

    expansions: dict[str, list[str]] = {}
    with CACHE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                expansions[row["id"]] = row["questions"]

    with CORPUS.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    texts = [" ".join(expansions.get(record["id"], [])) for record in records]
    covered = sum(1 for t in texts if t.strip())
    print(f"{len(records)} chunks | {covered} with expansions "
          f"({covered / len(records):.0%}) | {sum(len(v) for v in expansions.values())} questions")

    if not covered:
        raise SystemExit("no chunk has an expansion; nothing to index.")

    # Same vectoriser settings as the main lexical index, so the two signals are
    # comparable and a difference between them is about the text, not the tokeniser.
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                                 token_pattern=r"(?u)\b[\w./<>-]+\b")
    matrix = vectorizer.fit_transform(texts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        # `texts` is stored alongside the matrix so the cross-encoder can be shown the
        # expansions too. Without it the reranker - half the final score - judges every
        # candidate on its raw text alone and cannot see the signal that surfaced it.
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix,
                     "ids": [r["id"] for r in records], "texts": texts,
                     "covered": covered}, f)
    print(f"wrote {out}  matrix {matrix.shape}  vocab {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
