"""Embedding-model ablation: build, evaluate and time each candidate.

Each model gets its own embedding file so runs are reproducible and comparable
without rebuilding, and each is evaluated with its own query instruction (see
embedding_models.py - evaluating a prefixed model without its prefix measures the
harness, not the model).

Reported per model: retrieval quality on the pinned gold set, corpus encode
throughput, and mean query latency. The last one decides what is viable in a live
demo, which a quality number alone cannot.

Usage:
    python scripts/ablate_embeddings.py
    python scripts/ablate_embeddings.py --models BAAI/bge-small-en-v1.5 BAAI/bge-m3
    python scripts/ablate_embeddings.py --skip-build      # reuse existing vectors
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Tools live outside the package; make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nephrolex.retrieval import retrieve as R
from nephrolex.retrieval.embedding_models import DEFAULT_ABLATION, get, slug
from nephrolex.evaluation.evaluate_gold import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "indexes"
DENSE_DIR = INDEX_DIR / "dense"
REPORTS = ROOT / "reports"
CORPUS = ROOT / "data" / "chunks_v2" / "retrieval_corpus.jsonl"
GOLD = ROOT / "eval" / "ckd_gold_eval.jsonl"

LATENCY_QUERIES = [
    "When should an adult with CKD be referred to specialist kidney care?",
    "What blood pressure target applies to adults with CKD?",
    "My patient's filtration rate came back at 38. Which band does that put them in?",
    "How often should eGFR be monitored in people with CKD?",
]


def load_gold() -> list[dict]:
    with GOLD.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(model_name: str, batch_size: int) -> dict | None:
    """Embed the corpus with one model, into that model's own file."""
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    out = DENSE_DIR / f"{slug(model_name)}.npy"
    if out.exists():
        print(f"    vectors already present: {out.name}")
        return json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))

    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(Path(__file__).parent / "build_indexes.py"),
        "--corpus", str(CORPUS),
        "--dense-model", model_name,
        "--dense-out", str(out),
        "--batch-size", str(batch_size),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        print(f"    BUILD FAILED: {' | '.join(tail)}")
        return None
    return json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))


def measure(model_name: str, cases: list[dict], top_k: int) -> tuple[dict, float]:
    out = DENSE_DIR / f"{slug(model_name)}.npy"
    R.set_dense_path(out)

    rows = [evaluate_case(case, R.retrieve(case["query"], top_k, model_name)) for case in cases]
    stats = aggregate(rows)

    R.retrieve(LATENCY_QUERIES[0], top_k, model_name)  # warm the model
    started = time.time()
    for query in LATENCY_QUERIES:
        R.retrieve(query, top_k, model_name)
    latency_ms = (time.time() - started) / len(LATENCY_QUERIES) * 1000

    by_slice = {
        name: aggregate([r for r in rows if r["slice"] == name])
        for name in ("dev", "test")
    }
    stats["dev_ndcg"] = by_slice["dev"]["ndcg_at_10"]
    stats["test_ndcg"] = by_slice["test"]["ndcg_at_10"]
    return stats, latency_ms


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_ABLATION)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--out", default=str(REPORTS / "embedding_ablation.json"))
    args = parser.parse_args()

    cases = load_gold()
    results: list[dict] = []

    # Lexical-only reference point, so the dense contribution is visible.
    R.set_dense_path(None)
    lexical_rows = [evaluate_case(case, R.retrieve(case["query"], args.top_k)) for case in cases]
    lexical = aggregate(lexical_rows)
    print(f"lexical baseline: nDCG@10={lexical['ndcg_at_10']:.4f} recall@20={lexical['recall_at_20']:.4f}\n")

    for model_name in args.models:
        spec = get(model_name)
        print(f"[{spec.short}] {spec.params_m}M params")
        meta = None if args.skip_build else build(model_name, args.batch_size)
        if meta is None and not args.skip_build:
            continue
        try:
            stats, latency_ms = measure(model_name, cases, args.top_k)
        except Exception as error:  # noqa: BLE001 - one bad model must not end the sweep
            print(f"    EVAL FAILED: {type(error).__name__}: {error}")
            continue

        entry = {
            "model": model_name,
            "short": spec.short,
            "params_m": spec.params_m,
            "dims": (meta or {}).get("shape", [0, spec.dims])[1],
            "query_prefix": bool(spec.query_prefix),
            "device": (meta or {}).get("device"),
            "encode_seconds": (meta or {}).get("encode_seconds"),
            "chunks_per_second": (meta or {}).get("chunks_per_second"),
            "query_latency_ms": round(latency_ms, 1),
            **{k: stats[k] for k in ("ndcg_at_10", "recall_at_10", "recall_at_20", "mrr_at_10", "p_at_5", "total_miss")},
            "dev_ndcg": stats["dev_ndcg"],
            "test_ndcg": stats["test_ndcg"],
        }
        results.append(entry)
        print(
            f"    nDCG@10={entry['ndcg_at_10']:.4f} recall@20={entry['recall_at_20']:.4f} "
            f"dev/test={entry['dev_ndcg']:.3f}/{entry['test_ndcg']:.3f} "
            f"query={entry['query_latency_ms']:.0f}ms\n"
        )

    # Merge with any previous sweep so re-running a single model (e.g. after a
    # transient download failure) tops up the table instead of replacing it.
    out_path = Path(args.out)
    merged: dict[str, dict] = {}
    if out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        merged = {entry["model"]: entry for entry in previous.get("models", [])}
    merged.update({entry["model"]: entry for entry in results})
    results = sorted(merged.values(), key=lambda e: -e["ndcg_at_10"])

    payload = {"lexical_baseline": lexical, "models": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if results:
        print("\n| Embedding model | Params | Dim | nDCG@10 | recall@20 | dev | test | Query ms |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        print(
            f"| _lexical only_ | - | - | {lexical['ndcg_at_10']:.4f} | "
            f"{lexical['recall_at_20']:.4f} | - | - | - |"
        )
        for entry in sorted(results, key=lambda e: -e["ndcg_at_10"]):
            print(
                f"| {entry['short']} | {entry['params_m']}M | {entry['dims']} | "
                f"{entry['ndcg_at_10']:.4f} | {entry['recall_at_20']:.4f} | "
                f"{entry['dev_ndcg']:.3f} | {entry['test_ndcg']:.3f} | "
                f"{entry['query_latency_ms']:.0f} |"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
