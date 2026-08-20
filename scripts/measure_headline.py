"""One authoritative run of every headline number, at whatever the code currently is.

Why this exists
---------------
The poster and the reports drifted: they carried a held-out figure of 0.8516 against a
measured 0.8411, a recall of 0.8163 against 0.8312, and an FAQ nDCG of 0.7013 against
0.7225. Numbers copied by hand from separate runs of a moving pipeline cannot be
reconciled after the fact - there is no way to tell which run each came from.

So every headline figure is produced here, in one process, from one loaded index, and
written with the provenance of what actually loaded. If a number is on the poster it
came out of this file's output.

Usage:
    python scripts/measure_headline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.evaluation.evaluate_gold import aggregate, evaluate_case  # noqa: E402
from newbieduo.paths import CORPUS, GOLD_SET, REPORTS  # noqa: E402
from newbieduo.retrieval import retrieve as R  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

SETS = {
    "main": GOLD_SET,
    "heldout": ROOT / "eval" / "heldout_chunkfirst.jsonl",
    "faq": ROOT / "eval" / "faq_clinical.jsonl",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    indexes = R.load_indexes()
    if indexes.dense is None:
        raise SystemExit("dense index did not load - refusing to report lexical-only numbers")

    provenance = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dense_model": DENSE_MODEL,
        "reranker": RERANKER,
        "dense_vectors": list(indexes.dense.shape),
        "corpus_chunks": len(indexes.records),
        "fusion_weights": {"tfidf": R.W_TFIDF, "dense": R.W_DENSE,
                           "meta": R.W_META, "doc2query": R.W_DOC2QUERY},
        "numeric_band_prior": R.NUMERIC_BAND_PRIOR,
        "band_selectivity": R.BAND_SELECTIVITY,
        "candidate_pool": R.CANDIDATE_POOL,
    }
    print("provenance")
    for key, value in provenance.items():
        print(f"  {key:<20} {value}")
    print()

    report = {"provenance": provenance, "sets": {}}
    for name, path in SETS.items():
        cases = [c for c in (json.loads(line) for line in
                             Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
                 if not c["should_abstain"]]
        rows, ranks = [], []
        for case in cases:
            results = R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER)
            rows.append(evaluate_case(case, results))
            gold = set(case["relevance"])
            ids = [r["id"] for r in results]
            ranks.append(next((i for i, x in enumerate(ids, 1) if x in gold), None))

        stats = aggregate(rows)
        stats["success_at_5"] = sum(1 for r in ranks if r and r <= 5)
        stats["success_at_5_pct"] = round(100 * stats["success_at_5"] / len(cases), 1)
        report["sets"][name] = stats
        print(f"{name}  ({len(cases)} questions)")
        for key in ("ndcg_at_10", "recall_at_10", "recall_at_20", "mrr_at_10", "p_at_5"):
            print(f"    {key:<16} {stats[key]:.4f}")
        print(f"    {'success@5':<16} {stats['success_at_5']}/{len(cases)}"
              f" = {stats['success_at_5_pct']}%")
        print(f"    {'total misses':<16} {stats['total_miss']}")
        print()

    out = REPORTS / "headline_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
