"""How many candidates should each retriever nominate?

`CANDIDATE_POOL` was the one constant in `retrieve.py` carrying no comment and no
measurement. The project notes record a 60-500 sweep, but that was the lexical era -
before dense retrieval, before the cross-encoder, and before doc2query - so it says
nothing about the pipeline that exists now.

What the depth actually controls
--------------------------------
Each retriever nominates its own top N; the union is deduplicated, fused, truncated to
N again, and the top `rerank_depth` of that goes to the cross-encoder. So depth sets a
hard ceiling on what is *reachable*: a gold chunk outside every retriever's top N cannot
be recovered by any amount of reranking.

Measured over the 244 grade-2 gold chunks in all three answer keys, reachability by
nomination depth is:

    top 30   91.8%      top 120  96.7%  (current)      top 500  100.0%
    top 60   93.9%      top 240  99.2%

So 120 leaves 8 chunks structurally unreachable. But reachable is not retrieved: a chunk
entering the pool at rank 200 carries a weak fused score and will not reach the rerank
window, so the headroom above may be unrealisable. That is what this script measures
rather than assumes, and it reports latency alongside quality because a deeper pool
costs per-candidate metadata scoring on the query path.

Usage:
    python scripts/sweep_candidate_pool.py
    python scripts/sweep_candidate_pool.py --depths 120,240,500
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case  # noqa: E402
from nephrolex.paths import GOLD_SET, REPORTS  # noqa: E402
from nephrolex.retrieval import retrieve as R  # noqa: E402

# Shared with the weight sweep rather than duplicated: the same provenance block, the
# same exact cross-encoder memoisation, the same bootstrap. Two sweeps that disagree
# should differ in what they varied, not in how they measured.
from sweep_doc2query_weight import (  # noqa: E402
    _STATS, memoise_reranker, provenance,
)

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

SETS = {
    "main": str(GOLD_SET),
    "heldout": str(ROOT / "eval" / "heldout_chunkfirst.jsonl"),
    "faq": str(ROOT / "eval" / "faq_clinical.jsonl"),
}


def bootstrap(a: list[float], b: list[float], n: int = 3000) -> tuple[float, float]:
    rng = random.Random(17)
    diffs = [x - y for x, y in zip(a, b)]
    m = len(diffs)
    means = sorted(statistics.fmean(diffs[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="60,120,240,500")
    parser.add_argument("--out", default=str(REPORTS / "candidate_pool_sweep.json"))
    args = parser.parse_args()

    data = {}
    for name, path in SETS.items():
        with open(path, "r", encoding="utf-8") as f:
            data[name] = [c for c in (json.loads(line) for line in f if line.strip())
                          if not c["should_abstain"]]

    depths = [int(d) for d in args.depths.split(",")]
    original = R.CANDIDATE_POOL
    baseline: dict[str, list[float]] = {}

    meta = provenance()
    meta["swept"] = "CANDIDATE_POOL"
    print("run provenance")
    for key in ("run_at", "corpus_chunks", "dense_vectors", "numeric_band_prior"):
        print(f"  {key:<22} {meta[key]}")
    print(f"  {'fusion_weights':<22} {meta['fusion_weights']}")
    for label, info in meta["files"].items():
        print(f"  {label:<22} {info['bytes']:>9,} bytes  modified {info['modified']}")
    queries = sum(len(v) for v in data.values())
    print(f"\n{queries} questions per depth, {len(depths)} depths\n")

    # Timings below are NOT comparable across rows. The cross-encoder memo makes later
    # rows reuse earlier work, so their seconds measure cache hits rather than the
    # setting being swept. Latency has to be measured in its own uncached pass; the
    # `s` and `ms/q` columns are here to show progress, not to be compared.
    memoise_reranker()
    report = {"provenance": meta, "runs": []}
    out = Path(args.out)

    print(f"{'pool':<7}{'main nDCG':<26}{'heldout nDCG':<26}{'faq s@5':<10}"
          f"{'s':<7}{'ms/q':<7}cache", flush=True)
    try:
        for depth in depths:
            R.CANDIDATE_POOL = depth
            row = {"pool": depth}
            line = f"{depth:<7}"
            elapsed_total = 0.0
            for name in ("main", "heldout", "faq"):
                rows, first = [], {}
                started = time.time()
                for case in data[name]:
                    results = R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER)
                    rows.append(evaluate_case(case, results))
                    gold = set(case["relevance"])
                    ids = [r["id"] for r in results]
                    first[case["id"]] = next((i for i, x in enumerate(ids, 1) if x in gold), None)
                elapsed_total += time.time() - started

                scores = [r["ndcg_at_10"] for r in rows]
                stats = aggregate(rows)
                row[name] = stats
                if depth == depths[0]:
                    baseline[name] = scores

                if name == "faq":
                    at5 = sum(1 for v in first.values() if v and v <= 5)
                    row["faq_success_at_5"] = at5
                    line += f"{at5}/{len(data[name]):<7}"
                elif depth == depths[0]:
                    line += f"{stats['ndcg_at_10']:.4f}{'':<19}"
                else:
                    lo, hi = bootstrap(scores, baseline[name])
                    mark = "*" if (lo > 0 or hi < 0) else " "
                    line += f"{stats['ndcg_at_10']:.4f} [{lo:+.3f},{hi:+.3f}]{mark} "
                    row[f"{name}_ci"] = [lo, hi]

            per_query = elapsed_total / queries * 1000
            hit_rate = _STATS["hits"] / _STATS["calls"] if _STATS["calls"] else 0.0
            row["ms_per_query"] = round(per_query, 1)
            row["seconds"] = round(elapsed_total, 1)
            row["rerank_cache_hit_rate"] = round(hit_rate, 3)
            print(line + f"{elapsed_total:<7.0f}{per_query:<7.0f}{hit_rate:.0%}", flush=True)
            report["runs"].append(row)
            # Written after every depth, so an interrupted run keeps what it finished.
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        R.CANDIDATE_POOL = original

    print(f"\n* = 95% CI excludes zero against pool {depths[0]}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
