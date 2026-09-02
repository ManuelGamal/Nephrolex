"""Sweep BAND_SELECTIVITY: how much of the numeric-band prior exclusivity has to earn.

Why
---
The prior asked "does this chunk state a range covering the patient's number?" and
answered yes or no. Asked which band an eGFR of 38 falls in, 7 of the 8 surviving
candidates answered yes - so the signal separated nothing, and a saturated signal is an
absent one. BAND_SELECTIVITY pays part of the prior for how exclusively a chunk's bands
point at the value: one band containing it scores full marks, one of seven scores less.

At 0.0 the formula reduces exactly to the old membership test, so the first row of this
sweep IS the shipped system rather than an approximation of it. Anything kept has to
beat that row on the sets that matter, not on the one query that motivated the change.

Auditability and speed notes are as in sweep_doc2query_weight.py: provenance records
what actually loaded, results are written after each setting, and cross-encoder scores
are memoised because they do not depend on the setting being swept - which also means
the timing columns measure cache hits and are NOT comparable across rows.

Usage:
    python scripts/sweep_band_selectivity.py
    python scripts/sweep_band_selectivity.py --values 0,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case  # noqa: E402
from nephrolex.paths import CORPUS, GOLD_SET, REPORTS  # noqa: E402
from nephrolex.retrieval import retrieve as R  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"

SETS = {
    "main": str(GOLD_SET),
    "heldout": str(ROOT / "eval" / "heldout_chunkfirst.jsonl"),
    "faq": str(ROOT / "eval" / "faq_clinical.jsonl"),
}

_STATS = {"calls": 0, "hits": 0, "seconds": 0.0}


def memoise_reranker() -> None:
    """Cache cross-encoder scores across weight settings.

    Exact, not approximate: the model scores a (query, text) pair, and the fusion weight
    changes only which pairs are asked for. Without this the sweep re-scores the same
    pairs once per weight, which is where nearly all of the wall time goes.
    """
    model = R.get_reranker(RERANKER)
    if getattr(model, "_memoised", False):
        return
    original = model.predict
    cache: dict[tuple[str, str], float] = {}

    def predict(pairs, *args, **kwargs):
        pairs = list(pairs)
        wanted = [p for p in pairs if (p[0], p[1]) not in cache]
        _STATS["calls"] += len(pairs)
        _STATS["hits"] += len(pairs) - len(wanted)
        if wanted:
            started = time.time()
            scored = original(wanted, *args, **kwargs)
            _STATS["seconds"] += time.time() - started
            for pair, score in zip(wanted, scored):
                cache[(pair[0], pair[1])] = float(score)
        return [cache[(p[0], p[1])] for p in pairs]

    model.predict = predict
    model._memoised = True


def provenance() -> dict:
    """What this run actually loaded, so two disagreeing runs can be told apart."""
    indexes = R.load_indexes()
    dense = indexes.dense
    files = {}
    for label, path in (("corpus", CORPUS),
                        ("tfidf", R.INDEX_DIR / "tfidf.pkl"),
                        ("doc2query", R.INDEX_DIR / "doc2query_tfidf.pkl"),
                        ("dense", R.INDEX_DIR / "dense" / R.DEFAULT_DENSE_FILE)):
        if Path(path).exists():
            stat = Path(path).stat()
            files[label] = {"bytes": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}
    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dense_model": DENSE_MODEL,
        "reranker": RERANKER,
        "dense_vectors": None if dense is None else list(dense.shape),
        "corpus_chunks": len(indexes.records),
        "fusion_weights": {"tfidf": R.W_TFIDF, "dense": R.W_DENSE,
                           "meta": R.W_META, "doc2query": R.W_DOC2QUERY},
        "numeric_band_prior": R.NUMERIC_BAND_PRIOR,
        "band_selectivity_default": R.BAND_SELECTIVITY,
        "rerank_with_doc2query": True,
        "files": files,
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
    parser.add_argument("--values", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--out", default=str(REPORTS / "band_selectivity_sweep.json"))
    args = parser.parse_args()

    data = {}
    for name, path in SETS.items():
        with open(path, "r", encoding="utf-8") as f:
            data[name] = [c for c in (json.loads(line) for line in f if line.strip())
                          if not c["should_abstain"]]
    queries = sum(len(v) for v in data.values())

    meta = provenance()
    print("run provenance")
    for key in ("run_at", "corpus_chunks", "dense_vectors", "numeric_band_prior"):
        print(f"  {key:<22} {meta[key]}")
    print(f"  {'fusion_weights':<22} {meta['fusion_weights']}")
    for label, info in meta["files"].items():
        print(f"  {label:<22} {info['bytes']:>9,} bytes  modified {info['modified']}")
    print(f"\n{queries} questions per setting, {len(args.values.split(','))} weights\n")

    # Timings below are NOT comparable across rows. The cross-encoder memo makes later
    # rows reuse earlier work, so their seconds measure cache hits rather than the
    # setting being swept. Latency has to be measured in its own uncached pass; the
    # `s` and `ms/q` columns are here to show progress, not to be compared.
    memoise_reranker()
    weights = [float(w) for w in args.values.split(",")]
    baseline: dict[str, list[float]] = {}
    report = {"provenance": meta, "runs": []}
    out = Path(args.out)

    print(f"{'sel':<7}{'main nDCG':<26}{'heldout nDCG':<26}{'faq s@5':<10}{'s':<7}{'ms/q':<7}cache")
    for w in weights:
        row = {"band_selectivity": w}
        R.BAND_SELECTIVITY = w
        line = f"{w:<7}"
        started = time.time()
        for name in ("main", "heldout", "faq"):
            rows, first = [], {}
            for case in data[name]:
                results = R.retrieve(case["query"], 20, DENSE_MODEL, RERANKER)
                rows.append(evaluate_case(case, results))
                gold = set(case["relevance"])
                ids = [r["id"] for r in results]
                first[case["id"]] = next((i for i, x in enumerate(ids, 1) if x in gold), None)
            scores = [r["ndcg_at_10"] for r in rows]
            stats = aggregate(rows)
            row[name] = stats
            row[f"{name}_success_at_5"] = sum(1 for v in first.values() if v and v <= 5)
            if w == weights[0]:
                baseline[name] = scores
            if name == "faq":
                line += f"{row['faq_success_at_5']}/{len(data[name]):<8}"
            elif w == weights[0]:
                line += f"{stats['ndcg_at_10']:.4f}{'':<19}"
            else:
                lo, hi = bootstrap(scores, baseline[name])
                row[f"{name}_ci"] = [lo, hi]
                line += f"{stats['ndcg_at_10']:.4f} [{lo:+.3f},{hi:+.3f}]{'*' if (lo > 0 or hi < 0) else ' '} "

        elapsed = time.time() - started
        hit_rate = _STATS["hits"] / _STATS["calls"] if _STATS["calls"] else 0.0
        row.update({"seconds": round(elapsed, 1),
                    "ms_per_query": round(elapsed / queries * 1000, 1),
                    "rerank_cache_hit_rate": round(hit_rate, 3)})
        print(line + f"{elapsed:<7.0f}{elapsed / queries * 1000:<7.0f}{hit_rate:.0%}", flush=True)

        report["runs"].append(row)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n* = 95% CI excludes zero against selectivity={weights[0]} (the shipped system)")
    print(f"cross-encoder: {_STATS['calls']:,} pair scores requested, "
          f"{_STATS['hits']:,} served from cache, {_STATS['seconds']:.0f}s spent scoring")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
