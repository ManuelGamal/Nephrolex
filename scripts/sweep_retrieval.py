"""Tune retrieval knobs on the dev slice, then confirm the winner on test.

Sweeps in-process so BM25 parameters can be varied without rewriting pickles.
Tuning happens on dev only; the test slice is scored once at the end so the
reported number is not the product of selection on the same data.

Usage:
    python scripts/sweep_retrieval.py --what bm25
    python scripts/sweep_retrieval.py --what weights
    python scripts/sweep_retrieval.py --what pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# Tools live outside the package; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nephrolex.retrieval import retrieve as R
from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "eval" / "ckd_gold_eval.jsonl"


def load_gold(slice_name: str) -> list[dict]:
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    return [case for case in cases if slice_name == "all" or case["slice"] == slice_name]


def score(cases: list[dict], top_k: int = 20, **kwargs) -> dict:
    rows = [evaluate_case(case, R.retrieve(case["query"], top_k, **kwargs)) for case in cases]
    return aggregate(rows)


def set_bm25(k1: float, b: float) -> None:
    """Rebuild the in-memory BM25 model with new parameters."""
    indexes = R.load_indexes()
    with (R.INDEX_DIR / "bm25.pkl").open("rb") as f:
        tokenized = __import__("pickle").load(f)["tokenized_corpus"]
    indexes.bm25 = BM25Okapi(tokenized, k1=k1, b=b)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", default="bm25", choices=["bm25", "weights", "pool"])
    args = parser.parse_args()

    dev = load_gold("dev")
    test = load_gold("test")
    results: list[tuple[str, dict, dict]] = []

    if args.what == "bm25":
        print("Sweeping BM25 k1/b on dev (default k1=1.5 b=0.75)\n")
        for k1 in (0.9, 1.2, 1.5, 2.0):
            for b in (0.2, 0.4, 0.6, 0.75, 0.9):
                set_bm25(k1, b)
                stats = score(dev)
                label = f"k1={k1} b={b}"
                results.append((label, {"k1": k1, "b": b}, stats))
                print(f"  {label:<16} nDCG@10={stats['ndcg_at_10']:.4f}  recall@20={stats['recall_at_20']:.4f}")

    elif args.what == "weights":
        print("Sweeping fusion weights on dev (lexical only; dense weight unused)\n")
        grid = [
            (0.35, 0.20, 0.15), (0.50, 0.20, 0.15), (0.50, 0.10, 0.25),
            (0.40, 0.30, 0.15), (0.30, 0.30, 0.25), (0.60, 0.20, 0.05),
            (0.45, 0.25, 0.30), (0.40, 0.20, 0.40),
        ]
        for w_bm25, w_tfidf, w_meta in grid:
            kwargs = {"w_bm25": w_bm25, "w_tfidf": w_tfidf, "w_meta": w_meta}
            stats = score(dev, **kwargs)
            label = f"bm25={w_bm25} tfidf={w_tfidf} meta={w_meta}"
            results.append((label, kwargs, stats))
            print(f"  {label:<38} nDCG@10={stats['ndcg_at_10']:.4f}  recall@20={stats['recall_at_20']:.4f}")

    else:
        print("Sweeping candidate pool depth on dev\n")
        original = R.CANDIDATE_POOL
        for depth in (60, 120, 200, 300, 500):
            R.CANDIDATE_POOL = depth
            stats = score(dev)
            results.append((f"pool={depth}", {"pool": depth}, stats))
            print(f"  pool={depth:<5} nDCG@10={stats['ndcg_at_10']:.4f}  recall@20={stats['recall_at_20']:.4f}")
        R.CANDIDATE_POOL = original

    label, config, dev_stats = max(results, key=lambda item: item[2]["ndcg_at_10"])
    print(f"\nbest on dev: {label}  nDCG@10={dev_stats['ndcg_at_10']:.4f}")

    # Confirm the dev winner on the held-out slice.
    if args.what == "bm25":
        set_bm25(config["k1"], config["b"])
        test_stats = score(test)
    elif args.what == "pool":
        original = R.CANDIDATE_POOL
        R.CANDIDATE_POOL = config["pool"]
        test_stats = score(test)
        R.CANDIDATE_POOL = original
    else:
        test_stats = score(test, **config)

    print(f"held-out test: nDCG@10={test_stats['ndcg_at_10']:.4f}  recall@20={test_stats['recall_at_20']:.4f}")


if __name__ == "__main__":
    main()
