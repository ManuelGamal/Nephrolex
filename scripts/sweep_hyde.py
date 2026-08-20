"""Measure HyDE: does rewriting the query into guideline register earn its place?

HyDE is the third attempt at the same weakness. The paraphrase slice scores well
below the direct slice because the question and the corpus use different words, and
the two techniques tried before this one both missed where the mismatch lives:
Contextual Retrieval enriched the documents, ColBERT changed the matching function.
Neither touched the query. Both were measured and rejected. This one is measured on
the same terms and gets rejected on the same terms if it does not pay.

What this reports, and why each part is here
--------------------------------------------
overall          the headline, so a gain on paraphrases that costs more elsewhere
                 cannot hide behind a slice table
paraphrase       the slice HyDE is *for*. If it does not move here it has no case,
                 whatever the overall number does
direct           the slice it can most easily damage - these questions already use
                 corpus vocabulary, so a hypothetical can only add drift
95% CI           paired bootstrap on per-question nDCG differences. The weight sweeps
                 in this project have repeatedly produced deltas smaller than their
                 own intervals; a mean is not a result
cache coverage   printed first and non-negotiable. A cache miss makes HyDE degrade
                 silently to the plain query, so an incomplete cache would quietly
                 measure a blend of two systems and report it as one. Coverage below
                 100% aborts unless --allow-partial is passed.

Usage:
    python scripts/sweep_hyde.py
    python scripts/sweep_hyde.py --weights 0.2,0.3,0.5,0.7
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.evaluation.evaluate_gold import aggregate, evaluate_case  # noqa: E402
from newbieduo.paths import GOLD_SET, REPORTS  # noqa: E402
from newbieduo.retrieval import hyde as hyde_module  # noqa: E402
from newbieduo.retrieval import retrieve as R  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"
KINDS = ("paraphrase", "direct", "multi_doc")


def bootstrap_paired(a: list[float], b: list[float], resamples: int = 3000) -> tuple[float, float]:
    """95% interval on the mean per-query difference."""
    rng = random.Random(17)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(
        statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def cache_coverage(cases: list[dict]) -> tuple[int, list[str]]:
    cache = hyde_module._load()
    missing = [c["id"] for c in cases if cache.get(hyde_module._key(c["query"])) is None]
    return len(cases) - len(missing), missing


def run(cases: list[dict], mode: str, w_hyde: float = 0.0) -> tuple[list[dict], list[float]]:
    rows = [
        evaluate_case(
            case,
            # hyde_generate stays False: evaluation reads the cache and never calls
            # out, so a run is reproducible and costs nothing.
            R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER,
                       hyde_mode=mode, w_hyde=w_hyde),
        )
        for case in cases
    ]
    return rows, [r["ndcg_at_10"] for r in rows]


def slice_of(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r.get("kind") == kind]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="0.3,0.5,0.7")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Measure anyway with an incomplete cache. The numbers will describe a "
             "mix of HyDE and plain queries.",
    )
    parser.add_argument("--out", default=str(REPORTS / "hyde_sweep.json"))
    args = parser.parse_args()

    with GOLD_SET.open("r", encoding="utf-8") as f:
        cases = [
            json.loads(line)
            for line in f
            if line.strip() and not json.loads(line)["should_abstain"]
        ]

    have, missing = cache_coverage(cases)
    print(f"hypothetical cache: {have}/{len(cases)} answerable gold questions")
    if missing:
        shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        print(f"  missing {len(missing)}: {shown}")
        if not args.allow_partial:
            raise SystemExit(
                "\nRefusing to measure on a partial cache - misses degrade silently to "
                "the\nplain query, so the result would be a blend reported as a single "
                "system.\nRun:  python scripts/hyde.py build --rpm 10\n"
                "or pass --allow-partial to measure the blend deliberately."
            )
        print("  --allow-partial: measuring the blend anyway")

    # "max" first and unweighted: it has no knob to tune, so if it wins it wins
    # without a hyperparameter fitted on the same 67 questions it is scored on.
    variants: list[tuple[str, str, float]] = [("max", "max", 0.0)]
    variants += [(f"blend {w}", "blend", float(w)) for w in args.weights.split(",")]

    print("\nbaseline (HyDE off)")
    base_rows, base_ndcg = run(cases, "off")
    base = aggregate(base_rows)
    print(f"  overall     nDCG {base['ndcg_at_10']:.4f}  R@10 {base['recall_at_10']:.4f}  "
          f"R@20 {base['recall_at_20']:.4f}  MRR {base['mrr_at_10']:.4f}")
    for kind in KINDS:
        sub = aggregate(slice_of(base_rows, kind))
        print(f"  {kind:<11} nDCG {sub['ndcg_at_10']:.4f}  ({sub['cases']} cases)")

    report = {
        "cache_coverage": [have, len(cases)],
        "missing": missing,
        "baseline": base,
        "runs": [],
    }

    base_misses = {r["id"] for r in base_rows if r.get("ndcg_at_10") == 0.0}

    for label, mode, w in variants:
        rows, ndcg = run(cases, mode, w)
        agg = aggregate(rows)
        low, high = bootstrap_paired(ndcg, base_ndcg)
        delta = agg["ndcg_at_10"] - base["ndcg_at_10"]
        verdict = "significant" if low > 0 or high < 0 else "within noise"
        print(f"\n{label}")
        print(f"  overall     nDCG {agg['ndcg_at_10']:.4f}  ({delta:+.4f})  "
              f"95% CI [{low:+.4f}, {high:+.4f}]  {verdict}")
        print(f"              R@10 {agg['recall_at_10']:.4f}  "
              f"R@20 {agg['recall_at_20']:.4f}  MRR {agg['mrr_at_10']:.4f}")

        # A question that retrieves nothing relevant is the failure a clinician sees;
        # nDCG averages it away against questions that were already fine.
        now_missing = {r["id"] for r in rows if r.get("ndcg_at_10") == 0.0}
        rescued, broken = base_misses - now_missing, now_missing - base_misses
        print(f"  total misses {len(base_misses)} -> {len(now_missing)}"
              f"   rescued {len(rescued)}   newly broken {len(broken)}")
        if rescued:
            print(f"    rescued: {', '.join(sorted(rescued))}")
        if broken:
            print(f"    broken:  {', '.join(sorted(broken))}")

        entry = {"variant": label, "mode": mode, "w_hyde": w, "overall": agg,
                 "delta": delta, "ci": [low, high], "verdict": verdict,
                 "rescued": sorted(rescued), "newly_broken": sorted(broken),
                 "by_kind": {}}
        for kind in KINDS:
            sub_rows, sub_base = slice_of(rows, kind), slice_of(base_rows, kind)
            sub, sub_b = aggregate(sub_rows), aggregate(sub_base)
            s_low, s_high = bootstrap_paired(
                [r["ndcg_at_10"] for r in sub_rows],
                [r["ndcg_at_10"] for r in sub_base],
            )
            d = sub["ndcg_at_10"] - sub_b["ndcg_at_10"]
            print(f"  {kind:<11} nDCG {sub['ndcg_at_10']:.4f}  ({d:+.4f})  "
                  f"CI [{s_low:+.4f}, {s_high:+.4f}]")
            entry["by_kind"][kind] = {
                "ndcg_at_10": sub["ndcg_at_10"], "delta": d,
                "ci": [s_low, s_high], "cases": sub["cases"],
            }
        report["runs"].append(entry)

    best = max(report["runs"], key=lambda r: r["overall"]["ndcg_at_10"])
    print(f"\nbest: {best['variant']} at nDCG {best['overall']['ndcg_at_10']:.4f} "
          f"({best['delta']:+.4f}, {best['verdict']})")
    if best["verdict"] == "within noise":
        print("No weight clears its own confidence interval. On this evidence HyDE joins\n"
              "Contextual Retrieval and ColBERT as measured and rejected.")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
