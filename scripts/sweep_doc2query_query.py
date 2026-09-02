"""A/B: score doc2query against the expanded query, or the raw one.

Every other lexical signal already uses the expanded query. doc2query did not, which
cost it exactly the queries document-side expansion exists to serve - the answer row
for "filtration rate ... which band" took a 0 on a 0.29-weight signal and sat at
rank 15, outside the 8 chunks generation reads.

False is the shipped behaviour and must reproduce 0.7250, or the comparison is
measuring something else.
"""
import json, random
from pathlib import Path
from nephrolex.evaluation.evaluate_gold import GOLD_PATH, read_jsonl, evaluate_case, aggregate
from nephrolex.retrieval.retrieve import retrieve, load_indexes

DENSE, RERANK, TOP_K = "abhinand/MedEmbed-large-v0.1", "BAAI/bge-reranker-v2-m3", 20
idx = load_indexes()
assert idx.dense is not None, "dense absent - would be a lexical-only run"
print(f"loaded: dense={idx.dense.shape} corpus={len(idx.records)} rerank={RERANK}", flush=True)

cases = read_jsonl(Path(GOLD_PATH))
per, summ = {}, {}
for on_exp in (False, True):
    rows = [evaluate_case(c, retrieve(c["query"], TOP_K, DENSE, RERANK,
                                      doc2query_on_expanded=on_exp)) for c in cases]
    o = aggregate(rows)
    st = aggregate([r for r in rows if r.get("category") == "staging"])
    per[on_exp] = {r["id"]: r["ndcg_at_10"] for r in rows if "ndcg_at_10" in r}
    summ[str(on_exp)] = {"overall": o, "staging": st}
    print(f"on_expanded={str(on_exp):5}  nDCG@10={o['ndcg_at_10']:.4f}  "
          f"recall@20={o['recall_at_20']:.4f}  total_miss={o['total_miss']}  "
          f"mrr@10={o['mrr_at_10']:.4f}  staging={st['ndcg_at_10']:.4f}", flush=True)

ids = sorted(per[False])
diffs = [per[True][i] - per[False][i] for i in ids]
point = sum(diffs) / len(diffs)
rng = random.Random(20260902)
boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
lo, hi = boot[249], boot[9750]
print(f"\npaired bootstrap (True - False): {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"{'clears zero' if lo > 0 else 'inside noise' if hi > 0 else 'negative'}", flush=True)
moved = [(i, per[True][i] - per[False][i]) for i in ids if abs(per[True][i] - per[False][i]) > 1e-9]
print(f"\ncases changed: {len(moved)} of {len(ids)}")
for i, d in sorted(moved, key=lambda x: x[1])[:6]:
    print(f"   {d:+.4f}  {i}")
print("   ...")
for i, d in sorted(moved, key=lambda x: -x[1])[:6]:
    print(f"   {d:+.4f}  {i}")

summ["delta"] = {"point": point, "ci_low": lo, "ci_high": hi, "n_changed": len(moved)}
Path("reports/doc2query_expanded_query.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
print("\nwrote reports/doc2query_expanded_query.json", flush=True)
