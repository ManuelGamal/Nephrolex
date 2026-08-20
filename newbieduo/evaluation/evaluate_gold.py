"""Graded-relevance retrieval evaluation against the pinned CKD gold set.

Unlike the earlier smoke evaluator, a "hit" here means the retriever returned a
specific chunk a clinician would cite - not that some returned chunk happened to
carry a matching topic label. Topic labels are excluded from scoring on purpose:
the retriever boosts on those labels, so scoring with them is circular.

Metrics (answerable cases only):
    nDCG@10   - graded, position-weighted. Headline number.
    recall@k  - fraction of gold chunks found in the top k (k = 10, 20).
    MRR@10    - reciprocal rank of the first grade-2 chunk.
    P@5       - fraction of the top 5 that are gold-relevant.

Abstention cases (scope / unanswerable) are scored separately: they have no gold
chunks, so retrieval "success" is meaningless. What is reported instead is the
top-1 fused score, which is the signal an abstention gate would threshold on.

Usage:
    python scripts/evaluate_gold.py
    python scripts/evaluate_gold.py --slice dev --top-k 20
    python scripts/evaluate_gold.py --dense-model BAAI/bge-m3 --reranker-model BAAI/bge-reranker-v2-m3
    python scripts/evaluate_gold.py --tag baseline_lexical --compare reports/gold_eval_baseline_lexical.json
"""

from __future__ import annotations

import argparse
import json
import re
import math
import sys
from pathlib import Path

from newbieduo.retrieval.retrieve import retrieve


from newbieduo.paths import ROOT
GOLD_PATH = ROOT / "eval" / "ckd_gold_eval.jsonl"
REPORTS_DIR = ROOT / "reports"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_grades: list[int], relevance: dict[str, int], k: int) -> float:
    actual = dcg([float(grade) for grade in ranked_grades[:k]])
    ideal = dcg([float(grade) for grade in sorted(relevance.values(), reverse=True)[:k]])
    return actual / ideal if ideal else 0.0


# KDIGO prints each recommendation twice, so the gold set - which pins resolved chunk
# IDs - lists both printings as separately-required evidence. That means recall@k
# rewards returning the same words twice, and a system that returns the recommendation
# once is marked down. 19 of the 51 answerable cases are affected.
#
# On since the metric was audited. It raises nDCG@10 by 0.0134 - inside the noise
# floor - so it changes no conclusion in this project, including the duplicate-
# suppression decision that led to it being noticed; suppression stays off.
#
# The 18-stage history in reports/ABLATION.md is NOT re-scored under it, and cannot
# be: those runs stored only their top 10 (so recall@20 is unrecoverable) and their
# chunk IDs resolve against corpora that have since been rebuilt - 0% of the original
# baseline's IDs exist today. Re-scoring them would silently find no duplicates and
# report a confident "no change" that measured nothing. That table stays as measured,
# labelled; only runs from the current corpus carry the corrected number.
#
# Disable with --no-collapse-duplicates to reproduce a pre-audit figure.
COLLAPSE_DUPLICATES = True

_FINGERPRINTS: dict[str, str] | None = None


def _fingerprints() -> dict[str, str]:
    """chunk id -> a fingerprint of its text, so two printings share one identity.

    On the whole text, not a prefix. This keyed on the first 160 characters until it was
    checked: a `table_row` chunk repeats its table's header before its own payload, so
    every row of a wide table produced the same key and was scored as one chunk. Seven
    rows of Table 31 (nephrotoxic drugs and their alternatives) and six of Table 32
    (medications to stop before surgery) collapsed that way - retrieving "ACEi/ARB ->
    hypotension" counted as retrieving "metformin -> lactic acidosis". 29 chunks across
    8 groups were affected, all of them table rows, which are exactly the chunks the
    numeric-accuracy claims rest on.

    Full-text matching still merges 197 genuine reprints; KDIGO's summary and chapter
    printings are exact, so nothing real is lost by dropping the prefix.
    """
    global _FINGERPRINTS
    if _FINGERPRINTS is None:
        from newbieduo.paths import CORPUS

        _FINGERPRINTS = {}
        with CORPUS.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                _FINGERPRINTS[record["id"]] = re.sub(
                    r"[^a-z0-9]", "", record["raw_text"].lower())
    return _FINGERPRINTS


def _collapse(relevance: dict, retrieved_ids: list[str]) -> tuple[dict, list[str]]:
    """Re-express gold and results in text space, crediting each text once."""
    fp = _fingerprints()
    collapsed: dict[str, int] = {}
    for chunk_id, grade in relevance.items():
        key = fp.get(chunk_id, chunk_id)
        collapsed[key] = max(collapsed.get(key, 0), grade)

    seen: set[str] = set()
    mapped: list[str] = []
    for chunk_id in retrieved_ids:
        key = fp.get(chunk_id, chunk_id)
        # A second copy of an already-credited text is not a second hit. It keeps its
        # rank slot - it did occupy one - but contributes no further gain.
        mapped.append(f"__dup__{chunk_id}" if key in seen else key)
        if key in collapsed:
            seen.add(key)
    return collapsed, mapped


def evaluate_case(case: dict, results: list[dict]) -> dict:
    relevance = case["relevance"]
    retrieved_ids = [result["id"] for result in results]
    original_ids = list(retrieved_ids)   # what was actually returned, for the report
    if COLLAPSE_DUPLICATES and not case["should_abstain"]:
        relevance, retrieved_ids = _collapse(relevance, retrieved_ids)
    grades = [relevance.get(chunk_id, 0) for chunk_id in retrieved_ids]

    row = {
        "id": case["id"],
        "slice": case["slice"],
        "kind": case["kind"],
        "category": case["category"],
        "query": case["query"],
        "gold_count": len(relevance),
        "top_score": round(results[0]["score"], 4) if results else 0.0,
        "retrieved": original_ids[:10],
    }

    if case["should_abstain"]:
        # No gold chunks. Report what an abstention gate would see, plus whether any
        # scope-limiting evidence was surfaced (for the paediatric case, which does
        # have a gold chunk despite being out of scope).
        found = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevance]
        row["scope_evidence_found"] = found[:3]
        return row

    first_primary = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade == 2), None
    )
    found_at_10 = sum(1 for chunk_id in retrieved_ids[:10] if chunk_id in relevance)
    found_at_20 = sum(1 for chunk_id in retrieved_ids[:20] if chunk_id in relevance)

    row.update(
        {
            "ndcg_at_10": round(ndcg_at_k(grades, relevance, 10), 4),
            "recall_at_10": round(found_at_10 / len(relevance), 4),
            "recall_at_20": round(found_at_20 / len(relevance), 4),
            "rr_at_10": round(1 / first_primary, 4) if first_primary and first_primary <= 10 else 0.0,
            "p_at_5": round(sum(1 for g in grades[:5] if g > 0) / 5, 4),
            "first_primary_rank": first_primary,
            "missed": [chunk_id for chunk_id in relevance if chunk_id not in retrieved_ids],
        }
    )
    return row


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate(rows: list[dict]) -> dict:
    scored = [row for row in rows if "ndcg_at_10" in row]
    return {
        "cases": len(scored),
        "ndcg_at_10": mean([row["ndcg_at_10"] for row in scored]),
        "recall_at_10": mean([row["recall_at_10"] for row in scored]),
        "recall_at_20": mean([row["recall_at_20"] for row in scored]),
        "mrr_at_10": mean([row["rr_at_10"] for row in scored]),
        "p_at_5": mean([row["p_at_5"] for row in scored]),
        "total_miss": sum(1 for row in scored if row["ndcg_at_10"] == 0.0),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Graded CKD retrieval evaluation.")
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument(
        "--collapse-duplicates", action=argparse.BooleanOptionalAction, default=None,
        help="Treat two printings of the same text as one gold item. Omitted: module default.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval depth (>=20 for recall@20).")
    parser.add_argument("--slice", default="all", choices=["all", "dev", "test"])
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--tag", default="current", help="Names the report file.")
    parser.add_argument("--compare", default=None, help="Path to an earlier report to diff against.")
    parser.add_argument("--show-failures", type=int, default=8, help="How many worst cases to print.")
    # Defaults are deliberately None so retrieve() supplies them. Declaring them
    # here duplicated the values in two files, and when retrieve.py's weights were
    # changed this evaluator kept passing the old ones explicitly - reporting "no
    # change" for a change it had silently overridden.
    parser.add_argument("--hyde-mode", default=None, choices=["off", "max", "blend"],
                        help="Search with a cached hypothetical guideline passage too.")
    parser.add_argument("--w-hyde", type=float, default=None,
                        help="Blend weight. Only read when --hyde-mode blend.")
    parser.add_argument("--w-tfidf", type=float, default=None)
    parser.add_argument("--w-doc2query", type=float, default=None,
                        help="Lexical weight on generated clinician-voice questions.")
    parser.add_argument("--w-dense", type=float, default=None)
    parser.add_argument("--w-meta", type=float, default=None)
    parser.add_argument("--w-rerank", type=float, default=None)
    parser.add_argument("--rerank-depth", type=int, default=None)
    parser.add_argument(
        "--dense-file",
        default=None,
        help="Embedding file to use (data/indexes/dense/<model>.npy). Defaults to the main one.",
    )
    args = parser.parse_args()

    if args.dense_file:
        from newbieduo.retrieval.retrieve import set_dense_path

        set_dense_path(args.dense_file)

    if args.collapse_duplicates is not None:
        globals()["COLLAPSE_DUPLICATES"] = args.collapse_duplicates

    cases = read_jsonl(Path(args.gold))
    if args.slice != "all":
        cases = [case for case in cases if case["slice"] == args.slice]

    weights = {
        key: value
        for key, value in (
            ("hyde_mode", args.hyde_mode), ("w_hyde", args.w_hyde),
            ("w_tfidf", args.w_tfidf), ("w_doc2query", args.w_doc2query),
            ("w_dense", args.w_dense), ("w_meta", args.w_meta),
            ("w_rerank", args.w_rerank), ("rerank_depth", args.rerank_depth),
        )
        if value is not None
    }

    # Record what was actually used, not what was requested.
    import inspect

    effective = {
        name: weights.get(name, parameter.default)
        for name, parameter in inspect.signature(retrieve).parameters.items()
        if name.startswith(("w_", "rerank_", "hyde_"))
    }

    rows = [
        evaluate_case(
            case,
            retrieve(case["query"], args.top_k, args.dense_model, args.reranker_model, **weights),
        )
        for case in cases
    ]

    from newbieduo.retrieval.retrieve import load_indexes

    dense_matrix = load_indexes().dense
    loaded_dense = None if dense_matrix is None else dense_matrix.shape

    scored = [row for row in rows if "ndcg_at_10" in row]
    abstain = [row for row in rows if "ndcg_at_10" not in row]

    overall = aggregate(rows)
    by_slice = {
        name: aggregate([row for row in rows if row["slice"] == name])
        for name in sorted({row["slice"] for row in rows})
    }
    by_kind = {
        name: aggregate([row for row in rows if row["kind"] == name])
        for name in sorted({row["kind"] for row in scored})
    }
    by_category = {
        name: aggregate([row for row in rows if row["category"] == name])
        for name in sorted({row["category"] for row in scored})
    }

    report = {
        "tag": args.tag,
        "config": {
            "top_k": args.top_k,
            "slice": args.slice,
            "dense_model": args.dense_model,
            # Which vectors were actually loaded, not which were asked for. Reports
            # written before this recorded dense_model="MedEmbed" whether or not the
            # embeddings were found, so a lexical-only run and a hybrid run left
            # identical-looking config blocks 0.064 nDCG apart.
            "dense_vectors": (
                None if loaded_dense is None
                else f"{loaded_dense[0]} vectors, dim {loaded_dense[1]}"
            ),
            "reranker_model": args.reranker_model,
            **effective,
        },
        "overall": overall,
        "by_slice": by_slice,
        "by_kind": by_kind,
        "by_category": by_category,
        "abstention_cases": {
            "count": len(abstain),
            "top_score_mean": mean([row["top_score"] for row in abstain]),
            "answerable_top_score_mean": mean([row["top_score"] for row in scored]),
        },
        "results": rows,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"gold_eval_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ console
    cfg = f"top_k={args.top_k} slice={args.slice}"
    cfg += f" dense={args.dense_model}" if args.dense_model else " dense=off"
    cfg += f" rerank={args.reranker_model}" if args.reranker_model else " rerank=off"
    print(f"\n=== {args.tag} === {cfg}")
    print(
        f"\nAnswerable cases: {overall['cases']}\n"
        f"  nDCG@10    {overall['ndcg_at_10']:.4f}\n"
        f"  recall@10  {overall['recall_at_10']:.4f}\n"
        f"  recall@20  {overall['recall_at_20']:.4f}\n"
        f"  MRR@10     {overall['mrr_at_10']:.4f}\n"
        f"  P@5        {overall['p_at_5']:.4f}\n"
        f"  total miss {overall['total_miss']} cases returned zero gold evidence"
    )

    print("\nBy slice:")
    for name, stats in by_slice.items():
        print(f"  {name:<6} n={stats['cases']:<3} nDCG@10={stats['ndcg_at_10']:.4f} recall@20={stats['recall_at_20']:.4f}")

    print("\nBy question kind:")
    for name, stats in by_kind.items():
        print(f"  {name:<12} n={stats['cases']:<3} nDCG@10={stats['ndcg_at_10']:.4f} recall@20={stats['recall_at_20']:.4f}")

    print("\nBy category:")
    for name, stats in sorted(by_category.items(), key=lambda item: item[1]["ndcg_at_10"]):
        print(f"  {name:<18} n={stats['cases']:<3} nDCG@10={stats['ndcg_at_10']:.4f}")

    print(
        f"\nAbstention separability (higher gap = easier to gate):\n"
        f"  mean top-1 score, answerable  {report['abstention_cases']['answerable_top_score_mean']:.4f}\n"
        f"  mean top-1 score, abstain     {report['abstention_cases']['top_score_mean']:.4f}"
    )

    worst = sorted(scored, key=lambda row: row["ndcg_at_10"])[: args.show_failures]
    if worst:
        print(f"\nWorst {len(worst)} cases:")
        for row in worst:
            print(f"  {row['ndcg_at_10']:.3f}  [{row['kind']}] {row['id']}: {row['query'][:70]}")
            for chunk_id in row["missed"][:2]:
                print(f"           missed: {chunk_id[:88]}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print(f"\nvs {baseline['tag']}:")
        for metric in ("ndcg_at_10", "recall_at_10", "recall_at_20", "mrr_at_10", "p_at_5"):
            before = baseline["overall"][metric]
            after = overall[metric]
            arrow = "+" if after >= before else ""
            print(f"  {metric:<12} {before:.4f} -> {after:.4f}  ({arrow}{after - before:+.4f})")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
