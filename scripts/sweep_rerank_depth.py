"""How many candidates should the cross-encoder actually score?

`rerank_depth = 30` rests on an earlier finding that 30 beat 50 on both quality and
speed. That was measured before doc2query existed, before the reranker began reading the
generated questions, before the corpus repairs and before the numeric-band prior moved.
Every one of those changes what reaches the window and how it is ordered inside it, so
the number is stale rather than wrong.

Why this is the interesting knob
--------------------------------
Depth is the second of two ceilings. `CANDIDATE_POOL` decides what is *reachable*; this
decides what is actually *judged*. A gold chunk can sit comfortably in the pool and never
be looked at: at depth 30 the cross-encoder sees 30 of up to 120 candidates, and the
other 90 are ranked by fusion alone. Where a sibling table row outranks the right one on
fused score, only the cross-encoder can correct it - and only if it is asked.

Deeper is not obviously better. The cross-encoder is the dominant cost, so depth is close
to linear in latency, and it can also demote a correct chunk by admitting competitors it
prefers - which is exactly how `mon_acr_paraphrase` was lost before the reranker was
given the expansions to read.

Ordering
--------
Depths run deepest first on purpose. The cross-encoder's score for a (query, chunk) pair
does not depend on the depth, so the deepest run populates the memo with a superset of
every shallower run's pairs, and the rest are served almost entirely from cache. Running
shallow-first would pay full price at every step.

Usage:
    python scripts/sweep_rerank_depth.py
    python scripts/sweep_rerank_depth.py --depths 20,30,50,80
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
from sweep_doc2query_weight import _STATS, memoise_reranker, provenance  # noqa: E402

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
    parser.add_argument("--depths", default="20,30,50,80")
    parser.add_argument("--baseline", type=int, default=30, help="Depth the CIs compare against.")
    parser.add_argument("--out", default=str(REPORTS / "rerank_depth_sweep.json"))
    args = parser.parse_args()

    data = {}
    for name, path in SETS.items():
        with open(path, "r", encoding="utf-8") as f:
            data[name] = [c for c in (json.loads(line) for line in f if line.strip())
                          if not c["should_abstain"]]
    queries = sum(len(v) for v in data.values())

    meta = provenance()
    meta["swept"] = "rerank_depth"
    print("run provenance")
    for key in ("run_at", "corpus_chunks", "dense_vectors", "numeric_band_prior"):
        print(f"  {key:<22} {meta[key]}")
    print(f"  {'fusion_weights':<22} {meta['fusion_weights']}")
    print(f"  {'candidate_pool':<22} {R.CANDIDATE_POOL}")
    for label, info in meta["files"].items():
        print(f"  {label:<22} {info['bytes']:>9,} bytes  modified {info['modified']}")

    # Deepest first: it fills the memo with a superset of every shallower run's pairs.
    depths = sorted({int(d) for d in args.depths.split(",")}, reverse=True)
    print(f"\n{queries} questions per depth, {len(depths)} depths, deepest first\n")

    # Timings below are NOT comparable across rows. The cross-encoder memo makes later
    # rows reuse earlier work, so their seconds measure cache hits rather than the
    # setting being swept. Latency has to be measured in its own uncached pass; the
    # `s` and `ms/q` columns are here to show progress, not to be compared.
    memoise_reranker()
    results: dict[int, dict] = {}
    scores_by_depth: dict[int, dict[str, list[float]]] = {}
    report = {"provenance": meta, "runs": []}
    out = Path(args.out)

    print(f"{'depth':<7}{'main nDCG':<12}{'heldout nDCG':<14}{'faq s@5':<10}"
          f"{'s':<7}{'ms/q':<7}cache", flush=True)
    for depth in depths:
        row = {"rerank_depth": depth}
        line = f"{depth:<7}"
        elapsed = 0.0
        scores_by_depth[depth] = {}
        for name in ("main", "heldout", "faq"):
            rows, first = [], {}
            started = time.time()
            for case in data[name]:
                found = R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER, rerank_depth=depth)
                rows.append(evaluate_case(case, found))
                gold = set(case["relevance"])
                ids = [r["id"] for r in found]
                first[case["id"]] = next((i for i, x in enumerate(ids, 1) if x in gold), None)
            elapsed += time.time() - started
            stats = aggregate(rows)
            row[name] = stats
            scores_by_depth[depth][name] = [r["ndcg_at_10"] for r in rows]
            row[f"{name}_success_at_5"] = sum(1 for v in first.values() if v and v <= 5)
            if name == "faq":
                line += f"{row['faq_success_at_5']}/{len(data[name]):<7}"
            else:
                line += f"{stats['ndcg_at_10']:<12.4f}" if name == "main" else f"{stats['ndcg_at_10']:<14.4f}"

        hit_rate = _STATS["hits"] / _STATS["calls"] if _STATS["calls"] else 0.0
        row.update({"seconds": round(elapsed, 1),
                    "ms_per_query": round(elapsed / queries * 1000, 1),
                    "rerank_cache_hit_rate": round(hit_rate, 3)})
        print(line + f"{elapsed:<7.0f}{elapsed / queries * 1000:<7.0f}{hit_rate:.0%}", flush=True)
        results[depth] = row
        report["runs"].append(row)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Confidence intervals against the shipped depth, computed once every run is in.
    base = args.baseline
    if base in scores_by_depth:
        print(f"\npaired bootstrap against depth {base}:")
        for depth in sorted(scores_by_depth):
            if depth == base:
                continue
            for name in ("main", "heldout"):
                lo, hi = bootstrap(scores_by_depth[depth][name], scores_by_depth[base][name])
                mark = "SIGNIFICANT" if (lo > 0 or hi < 0) else "within noise"
                delta = (statistics.fmean(scores_by_depth[depth][name])
                         - statistics.fmean(scores_by_depth[base][name]))
                print(f"  depth {depth:<4} {name:<8} {delta:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {mark}")
                results[depth][f"{name}_ci_vs_{base}"] = [lo, hi]
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ncross-encoder: {_STATS['calls']:,} pair scores requested, "
          f"{_STATS['hits']:,} served from cache, {_STATS['seconds']:.0f}s spent scoring")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
