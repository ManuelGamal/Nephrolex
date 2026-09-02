"""Leave-one-out ablation: does every component of the retriever earn its place?

Each row removes exactly one piece and re-measures. A component that costs nothing
to remove is complexity without benefit, and should be defended or deleted rather
than left in because it sounded reasonable.

Two comparisons come from the literature rather than from us:

  - Reciprocal Rank Fusion instead of the convex score combination. RRF is the more
    common choice; the benchmark in arXiv:2604.01733 notes fusion choice materially
    affects results, so it is measured rather than assumed.
  - Reranking, which that same benchmark found to be the single largest contributor
    (+17.2pp MRR@3, +12.1pp Recall@5 over unreranked hybrid). Our own number should
    be checked against that direction of effect.

Differences are reported against the bootstrap noise floor, because at 51 queries
most single-component effects are smaller than the error bar and an argmax over
them would be fitting noise.

Usage:
    python scripts/ablate_components.py
    python scripts/ablate_components.py --no-rerank      # faster, lexical+dense only
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
from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "ckd_gold_eval.jsonl"
DENSE_FILE = ROOT / "data" / "indexes" / "dense" / "abhinand__MedEmbed-large-v0_1.npy"
DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

# Weight used to switch ON a signal that ships at 0.0, so the row tests something.
# Arbitrary but comparable to the other weights; the point is the direction of the
# effect, not this exact value.
ABLATION_ON_WEIGHT = 0.30


def load_gold() -> list[dict]:
    with GOLD.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not json.loads(line)["should_abstain"]]


def bootstrap_paired(a: list[float], b: list[float], resamples: int = 3000) -> tuple[float, float]:
    """95% interval on the mean per-query difference."""
    rng = random.Random(17)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def run(cases: list[dict], reranker: str | None, **kwargs) -> tuple[dict, list[float]]:
    rows = [
        evaluate_case(case, R.retrieve(case["query"], 20, DENSE_MODEL, reranker, **kwargs))
        for case in cases
    ]
    return aggregate(rows), [r["ndcg_at_10"] for r in rows]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "reports" / "component_ablation.json"))
    args = parser.parse_args()

    R.set_dense_path(DENSE_FILE)
    reranker = None if args.no_rerank else RERANKER
    cases = load_gold()

    baseline_stats, baseline_scores = run(cases, reranker)
    print(f"{len(cases)} answerable questions | reranker={reranker or 'off'}\n")
    print(f"{'configuration':<44} {'nDCG@10':>8} {'recall@20':>10} {'MRR@10':>8} {'vs base':>9}  95% CI")
    print(f"{'full pipeline':<44} {baseline_stats['ndcg_at_10']:>8.4f} "
          f"{baseline_stats['recall_at_20']:>10.4f} {baseline_stats['mrr_at_10']:>8.4f} {'—':>9}")

    # --- signal rows, derived from retrieve.SIGNALS ---------------------------
    # Enumerated, never restated. The previous hand-written list drifted from the
    # retriever and produced a row that tested nothing (see the null-comparison guard
    # below). A signal already at weight 0 is tested by switching it ON, since turning
    # off something already off compares the baseline against itself.
    import inspect

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(R.retrieve).parameters.items()
    }
    variants: list[tuple[str, dict, dict]] = []
    for signal in R.SIGNALS:
        if signal.additive:
            current = defaults.get(signal.weight_param, 0.0)
            if current:
                variants.append((f"without {signal.name}", {signal.weight_param: 0.0}, {}))
            else:
                variants.append((
                    f"with {signal.name} (w={ABLATION_ON_WEIGHT})",
                    {signal.weight_param: ABLATION_ON_WEIGHT}, {},
                ))
        else:
            variants.append((f"without {signal.name}", {}, dict(signal.disable)))

    # --- pipeline-stage rows, which are not scoring signals --------------------
    variants += [
        ("without query/expansion blend", {}, {"W_ORIGINAL_QUERY": 1.0}),
        ("without length damping", {}, {"LENGTH_NORM_CHARS": 10**9}),
        ("RRF instead of convex fusion", {"fusion": "rrf"}, {}),
    ]
    if reranker:
        variants.insert(0, ("without cross-encoder rerank", {}, {"__reranker__": None}))

    results = []
    null_comparisons: list[str] = []
    for label, kwargs, constants in variants:
        saved = {name: getattr(R, name) for name in constants if not name.startswith("__")}
        for name, value in constants.items():
            if not name.startswith("__"):
                setattr(R, name, value)
        this_reranker = None if constants.get("__reranker__", "keep") is None else reranker
        try:
            stats, scores = run(cases, this_reranker, **kwargs)
        finally:
            for name, value in saved.items():
                setattr(R, name, value)

        delta = stats["ndcg_at_10"] - baseline_stats["ndcg_at_10"]
        low, high = bootstrap_paired(scores, baseline_scores)

        # Identical per-query scores mean one of two very different things, and calling
        # them both "no effect" is how the fake 0.0000 survived:
        #
        #   - the variant was configured to the value the baseline already uses, so
        #     nothing was tested and the row is a bug; or
        #   - the configuration genuinely differs and changes no ranking, which is a
        #     real (and useful) finding that the component is inert here.
        #
        # Only the first is a defect, so they are separated by comparing configuration,
        # not just output.
        untested = all(
            value == defaults.get(name) for name, value in kwargs.items()
        ) and all(
            value == saved.get(name) for name, value in constants.items()
            if not name.startswith("__")
        )
        if scores == baseline_scores and untested:
            flag = "  NOT TESTED - variant configured identically to the baseline"
            null_comparisons.append(label)
        elif scores == baseline_scores:
            flag = "  inert - config differs, no ranking changes"
        else:
            flag = "  MATTERS" if (low > 0 or high < 0) else ""
        print(f"{label:<44} {stats['ndcg_at_10']:>8.4f} {stats['recall_at_20']:>10.4f} "
              f"{stats['mrr_at_10']:>8.4f} {delta:>+9.4f}  [{low:+.4f}, {high:+.4f}]{flag}")
        results.append({
            "variant": label, "ndcg_at_10": stats["ndcg_at_10"],
            "recall_at_20": stats["recall_at_20"], "mrr_at_10": stats["mrr_at_10"],
            "delta_ndcg": round(delta, 4), "ci": [round(low, 4), round(high, 4)],
            "significant": bool(low > 0 or high < 0),
        })

    print("\n'MATTERS' = removing it changes nDCG@10 by more than the paired bootstrap interval.")
    print("Everything else is inside the noise floor at this sample size.")

    print("'inert' = the configuration really changed and no ranking moved.")

    if null_comparisons:
        print(f"\n{len(null_comparisons)} ROW(S) TESTED NOTHING - configured like the baseline:")
        for label in null_comparisons:
            print(f"  - {label}")
        print("Fix the variant's configuration; do not read these as 'no effect'.")
        raise SystemExit(1)

    Path(args.out).write_text(json.dumps({
        "queries": len(cases), "reranker": reranker,
        "baseline": baseline_stats, "variants": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
