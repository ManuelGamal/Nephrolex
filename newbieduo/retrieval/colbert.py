"""Late-interaction (ColBERT) retrieval over the CKD corpus.

Why try it at all
-----------------
The component ablation found BM25 was hurting and removed it from scoring, which
leaves exact-token matching - "G3b", "1.6.2", "70 mg/mmol" - resting on TF-IDF
alone. Late interaction is the mechanism that recovers lexical precision inside a
semantic model: instead of pooling a passage into one vector, it keeps a vector per
token and scores by summing, for each query token, its best match in the passage
(MaxSim). That is exactly the behaviour a single-vector bi-encoder loses.

Implementation note - this is an approximate ColBERT
----------------------------------------------------
`answerdotai/answerai-colbert-small-v1` stores its 384 -> 96 projection as
`linear.weight`, which `AutoModel` discards as an unexpected key; it is loaded
directly from the checkpoint here. What is *not* reproduced is ColBERT's query
augmentation (padding the query with [MASK] tokens) and its [Q]/[D] marker tokens,
which a full implementation such as PyLate provides. MaxSim itself is exact. So a
weak result here is evidence against late interaction *as implemented*, not proof
that a proper implementation would not help - and that distinction is stated
because it changes what the measurement licenses us to claim.

What it measured
----------------
Late interaction works, but it is redundant with the cross-encoder we already run,
and weaker. Gold set, 51 answerable questions:

    ColBERT alone                              nDCG@10 0.5931    55 ms
    hybrid, no reranker                                0.6546    46 ms
    hybrid + ColBERT (w=0.30)                          0.6760   111 ms
    hybrid + MiniLM cross-encoder                      0.6741   106 ms
    hybrid + bge-reranker-v2-m3  <- shipped            0.7280  1323 ms
    hybrid + bge + ColBERT (w=0.40)                    0.6970  1156 ms   [-0.060, -0.004]

So the hypothesis that motivated this - that dropping BM25 left exact-token matching
underserved, and MaxSim would recover it - is not supported. Added *underneath* the
cross-encoder, ColBERT costs 0.031 nDCG and the paired bootstrap interval excludes
zero: the cross-encoder was already supplying that signal, better. The default weight
is therefore 0.0 and the pipeline is unchanged.

What it is genuinely good for is a fast mode. At w=0.30 without any cross-encoder it
edges out the MiniLM cross-encoder on nDCG at comparable latency, and reaches 93% of
the full pipeline's quality for 8% of its cost. That is the trade this corpus does not
need - 1,523 chunks, one user, a plugged-in laptop - but it is the configuration to
reach for if the corpus grows or the demo has to run on CPU.

Usage:
    python scripts/colbert.py build          # encode the corpus (~1 min on GPU)
    python scripts/colbert.py search "when should CKD be referred?"
    python scripts/colbert.py evaluate       # against the gold set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


from newbieduo.paths import ROOT

CORPUS = ROOT / "data" / "chunks_v2" / "retrieval_corpus.jsonl"
OUT_DIR = ROOT / "data" / "indexes" / "colbert"
MODEL = "answerdotai/answerai-colbert-small-v1"

MAX_DOC_TOKENS = 192
MAX_QUERY_TOKENS = 48


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(name: str = MODEL):
    """BERT encoder plus the ColBERT projection that AutoModel drops."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    weights = load_file(hf_hub_download(name, "model.safetensors"))
    projection = weights.get("linear.weight")
    if projection is None:
        raise SystemExit("checkpoint has no linear.weight; cannot build ColBERT vectors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, projection.to(device), device


def encode(texts: list[str], tokenizer, model, projection, device, max_tokens: int,
           batch_size: int = 32) -> list[np.ndarray]:
    """Per-token, projected, L2-normalised embeddings for each text."""
    import torch

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt"
            ).to(device)
            hidden = model(**encoded).last_hidden_state          # (B, T, 384)
            vectors = hidden @ projection.T                       # (B, T, 96)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            mask = encoded["attention_mask"].bool()
            for row, keep in zip(vectors, mask):
                out.append(row[keep].to(torch.float16).cpu().numpy())
    return out


def maxsim(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
    """ColBERT scoring: for each query token, its best match in the document."""
    if doc_vectors.size == 0:
        return 0.0
    return float((query_vectors.astype(np.float32) @ doc_vectors.astype(np.float32).T).max(axis=1).sum())


def build(batch_size: int) -> None:
    records = read_jsonl(CORPUS)
    tokenizer, model, projection, device = load_model()
    print(f"encoding {len(records)} chunks on {device}")

    texts = [
        (" > ".join(r["metadata"].get("section_path", [])) + "\n" + r["raw_text"]).strip()
        for r in records
    ]
    started = time.time()
    vectors = encode(texts, tokenizer, model, projection, device, MAX_DOC_TOKENS, batch_size)
    elapsed = time.time() - started

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ragged per-document lengths, so store one flat array plus offsets.
    lengths = np.array([v.shape[0] for v in vectors], dtype=np.int32)
    flat = np.concatenate(vectors, axis=0)
    np.save(OUT_DIR / "tokens.npy", flat)
    np.save(OUT_DIR / "lengths.npy", lengths)
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "model": MODEL, "chunks": len(records), "dim": int(flat.shape[1]),
        "total_tokens": int(flat.shape[0]), "encode_seconds": round(elapsed, 1),
        "megabytes": round(flat.nbytes / 1e6, 1), "max_doc_tokens": MAX_DOC_TOKENS,
    }, indent=2), encoding="utf-8")
    print(f"{flat.shape[0]} tokens x {flat.shape[1]} dims -> {flat.nbytes / 1e6:.0f} MB in {elapsed:.0f}s")


def load_index() -> tuple[list[np.ndarray], list[dict]]:
    flat = np.load(OUT_DIR / "tokens.npy")
    lengths = np.load(OUT_DIR / "lengths.npy")
    records = read_jsonl(CORPUS)
    docs, offset = [], 0
    for length in lengths:
        docs.append(flat[offset : offset + length])
        offset += length
    return docs, records


def score_all(query: str, docs, tokenizer, model, projection, device) -> np.ndarray:
    qv = encode([query], tokenizer, model, projection, device, MAX_QUERY_TOKENS)[0]
    return np.array([maxsim(qv, d) for d in docs], dtype=np.float32)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "search", "evaluate"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "build":
        build(args.batch_size)
        return

    docs, records = load_index()
    tokenizer, model, projection, device = load_model()

    if args.command == "search":
        scores = score_all(args.query, docs, tokenizer, model, projection, device)
        for idx in np.argsort(scores)[::-1][: args.top_k]:
            meta = records[idx]["metadata"]
            print(f"  {scores[idx]:7.2f}  {meta.get('citation')}")
            print(f"           {' '.join(records[idx]['raw_text'].split())[:88]}")
        return

    # ------------------------------------------------------------ evaluation
    from newbieduo.evaluation.evaluate_gold import aggregate, evaluate_case

    gold = [g for g in read_jsonl(ROOT / "eval" / "ckd_gold_eval.jsonl") if not g["should_abstain"]]
    rows = []
    started = time.time()
    for case in gold:
        scores = score_all(case["query"], docs, tokenizer, model, projection, device)
        order = np.argsort(scores)[::-1][:20]
        results = [{"id": records[i]["id"], "score": float(scores[i]),
                    "raw_text": records[i]["raw_text"], "metadata": records[i]["metadata"]}
                   for i in order]
        rows.append(evaluate_case(case, results))
    stats = aggregate(rows)
    latency = (time.time() - started) / len(gold) * 1000

    print(f"ColBERT late interaction alone, {len(gold)} questions")
    for key in ("ndcg_at_10", "recall_at_10", "recall_at_20", "mrr_at_10", "p_at_5"):
        print(f"  {key:<14} {stats[key]:.4f}")
    print(f"  {'query latency':<14} {latency:.0f} ms")
    (ROOT / "reports" / "colbert_eval.json").write_text(
        json.dumps({"model": MODEL, "overall": stats, "query_latency_ms": round(latency)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
