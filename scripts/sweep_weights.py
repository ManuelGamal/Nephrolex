"""Sweep the fusion weights on the current stack, with a noise floor to judge against.

A sweep without an error bar is not evidence. With 34 dev queries the standard error
on mean nDCG@10 is large enough that most weight configurations are
indistinguishable, and picking the argmax anyway is how the original pipeline ended
up scoring 0.343 on dev and 0.134 on test.

So this reports, for every configuration:
  - dev nDCG@10
  - a bootstrap 95% interval on that estimate
  - whether it beats the baseline by more than the paired bootstrap interval on the
    *difference*, which is the only comparison that accounts for the two runs sharing
    the same queries

Then it scores the best configuration once on the held-out test slice.

Usage:
    python scripts/sweep_weights.py
    python scripts/sweep_weights.py --no-rerank --resamples 5000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

# Tools live outside the package; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nephrolex.retrieval import retrieve as R
from nephrolex.evaluation.evaluate_gold import evaluate_case


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "ckd_gold_eval.jsonl"
DENSE_FILE = ROOT / "data" / "indexes" / "dense" / "abhinand__MedEmbed-large-v0_1.npy"
DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

# (bm25, tfidf, dense, metadata). The baseline is first.
GRID = [
    (0.35, 0.20, 0.30, 0.15),
    (0.25, 0.15, 0.45, 0.15),
    (0.45, 0.20, 0.20, 0.15),
    (0.30, 0.10, 0.45, 0.15),
    (0.20, 0.20, 0.40, 0.20),
    (0.40, 0.25, 0.25, 0.10),
    (0.30, 0.20, 0.30, 0.20),
    (0.35, 0.05, 0.45, 0.15),
    (0.50, 0.15, 0.25, 0.10),
    (0.25, 0.25, 0.25, 0.25),
]


def load(slice_name: str) -> list[dict]:
    with GOLD.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    return [c for c in cases if c["slice"] == slice_name and not c["should_abstain"]]


def per_query_ndcg(cases: list[dict], weights: tuple, reranker: str | None, top_k: int) -> list[float]:
    scores = []
    for case in cases:
        result = evaluate_case(
            case,
            R.retrieve(
                case["query"], top_k, DENSE_MODEL, reranker,
                w_bm25=weights[0], w_tfidf=weights[1], w_dense=weights[2], w_meta=weights[3],
                rerank_depth=30,
            ),
        )
        scores.append(result.get("ndcg_at_10", 0.0))
    return scores


def bootstrap_ci(values: list[float], resamples: int, rng: random.Random) -> tuple[float, float]:
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def paired_ci(a: list[float], b: list[float], resamples: int, rng: random.Random) -> tuple[float, float]:
    """Interval on the mean per-query difference, which shares the query sample."""
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, resamples, rng)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    rng = random.Random(17)
    reranker = None if args.no_rerank else RERANKER
    R.set_dense_path(DENSE_FILE)

    dev, test = load("dev"), load("test")
    print(f"dev {len(dev)} queries | test {len(test)} queries | rerank={reranker or 'off'}\n")

    baseline_scores = per_query_ndcg(dev, GRID[0], reranker, args.top_k)
    base_mean = statistics.fmean(baseline_scores)
    low, high = bootstrap_ci(baseline_scores, args.resamples, rng)
    print(f"baseline  bm25=.35 tfidf=.20 dense=.30 meta=.15")
    print(f"          dev nDCG@10 {base_mean:.4f}   95% CI [{low:.4f}, {high:.4f}]")
    print(f"          the interval is +/-{(high - low) / 2:.4f} - any gain smaller than that is not measurable here\n")

    rows = [(GRID[0], base_mean, baseline_scores)]
    print(f"{'weights (bm25/tfidf/dense/meta)':<34} {'dev nDCG':>9} {'vs base':>9} {'95% CI on difference':>24}")
    for weights in GRID[1:]:
        scores = per_query_ndcg(dev, weights, reranker, args.top_k)
        mean = statistics.fmean(scores)
        d_low, d_high = paired_ci(scores, baseline_scores, args.resamples, rng)
        significant = "  SIGNIFICANT" if (d_low > 0 or d_high < 0) else ""
        label = "/".join(f"{w:.2f}" for w in weights)
        print(f"{label:<34} {mean:>9.4f} {mean - base_mean:>+9.4f}   [{d_low:+.4f}, {d_high:+.4f}]{significant}")
        rows.append((weights, mean, scores))

    best_weights, best_mean, best_scores = max(rows, key=lambda r: r[1])
    d_low, d_high = paired_ci(best_scores, baseline_scores, args.resamples, rng)
    print(f"\nbest on dev: {'/'.join(f'{w:.2f}' for w in best_weights)}  {best_mean:.4f}")
    print(f"  difference from baseline: {best_mean - base_mean:+.4f}, 95% CI [{d_low:+.4f}, {d_high:+.4f}]")
    if d_low <= 0 <= d_high:
        print("  -> the interval spans zero: this is not a real improvement, it is sampling noise")

    # Held-out confirmation, scored once.
    base_test = statistics.fmean(per_query_ndcg(test, GRID[0], reranker, args.top_k))
    best_test = statistics.fmean(per_query_ndcg(test, best_weights, reranker, args.top_k))
    print(f"\nheld-out test:  baseline {base_test:.4f}   dev-best {best_test:.4f}   ({best_test - base_test:+.4f})")
    if best_test < base_test:
        print("  -> the dev winner is worse on test, which is the signature of fitting noise")

    out = ROOT / "reports" / "weight_sweep.json"
    out.write_text(json.dumps({
        "dev_queries": len(dev), "test_queries": len(test), "reranker": reranker,
        "baseline_weights": GRID[0], "baseline_dev": base_mean,
        "baseline_dev_ci": [low, high],
        "results": [{"weights": w, "dev_ndcg": m} for w, m, _ in rows],
        "best_weights": best_weights, "best_dev": best_mean,
        "baseline_test": base_test, "best_test": best_test,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
