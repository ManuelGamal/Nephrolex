"""Sweep the additive numeric-band weight against the gold set.

W_BAND was added because the band signal was multiplicative only: it scaled a
relevance the answer row never earned, since "38" shares no vocabulary with
"30 - 44". The row that answers the staging question sat at rank 15 and never
reached the model, which then refused correctly on the evidence it was shown.

w_band=0.0 is the published baseline and must reproduce 0.7250, otherwise the
comparison is measuring something else. Loaded state is printed before any
number is believed: a silent fall back to lexical-only has faked a result here
before.
"""
import json
import random
from pathlib import Path

from nephrolex.evaluation.evaluate_gold import (
    GOLD_PATH, read_jsonl, evaluate_case, aggregate,
)
from nephrolex.retrieval.retrieve import retrieve, load_indexes

DENSE = "abhinand/MedEmbed-large-v0.1"
RERANK = "BAAI/bge-reranker-v2-m3"
WEIGHTS = [0.0, 0.10, 0.20, 0.30]
TOP_K = 20

idx = load_indexes()
assert idx.dense is not None, "dense matrix absent - would be a lexical-only run"
print(f"loaded state: dense={idx.dense.shape} corpus={len(idx.records)} "
      f"dense_model={DENSE} reranker={RERANK}", flush=True)

cases = read_jsonl(Path(GOLD_PATH))
print(f"gold cases: {len(cases)}\n", flush=True)

per_case: dict[float, dict[str, float]] = {}
summary: dict[float, dict] = {}

for wb in WEIGHTS:
    rows = [
        evaluate_case(c, retrieve(c["query"], TOP_K, DENSE, RERANK, w_band=wb))
        for c in cases
    ]
    overall = aggregate(rows)
    staging = aggregate([r for r in rows if r.get("category") == "staging"])
    per_case[wb] = {r["id"]: r["ndcg_at_10"] for r in rows if "ndcg_at_10" in r}
    summary[wb] = {"overall": overall, "staging": staging}
    print(f"w_band={wb:.2f}  nDCG@10={overall['ndcg_at_10']:.4f}  "
          f"recall@20={overall['recall_at_20']:.4f}  "
          f"total_miss={overall['total_miss']}  "
          f"staging_nDCG={staging['ndcg_at_10']:.4f}", flush=True)

# paired bootstrap against the w_band=0 baseline
base = per_case[WEIGHTS[0]]
rng = random.Random(20260902)
ids = sorted(base)
print("\npaired bootstrap vs w_band=0.00 (95% CI, 10000 resamples):", flush=True)
for wb in WEIGHTS[1:]:
    diffs = [per_case[wb][i] - base[i] for i in ids]
    point = sum(diffs) / len(diffs)
    boot = []
    for _ in range(10000):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo, hi = boot[249], boot[9750]
    flag = "clears zero" if lo > 0 else ("negative" if hi < 0 else "inside noise")
    print(f"  w_band={wb:.2f}  {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {flag}", flush=True)
    summary[wb]["delta_vs_baseline"] = {"point": point, "ci_low": lo, "ci_high": hi}

out = Path("reports/band_weight_sweep.json")
out.write_text(json.dumps({
    "provenance": {"dense_model": DENSE, "reranker": RERANK,
                   "dense_vectors": list(idx.dense.shape),
                   "corpus_chunks": len(idx.records), "top_k": TOP_K},
    "sweep": {str(k): v for k, v in summary.items()},
}, indent=2), encoding="utf-8")
print(f"\nwrote {out}", flush=True)
