from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from nephrolex.paths import ROOT, CORPUS as SHIPPED_CORPUS
CHUNKS_DIR = ROOT / "data" / "chunks"
INDEX_DIR = ROOT / "data" / "indexes"
REPORTS_DIR = ROOT / "reports"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?|[<>]=?|\bG[1-5][ab]?\b|\bA[1-3]\b", text.lower())


def indexable_text(record: dict, mode: str) -> str:
    """Choose what actually goes into the lexical indexes.

    Measured on the 59-case gold set (lexical only, top_k=20), nDCG@10:

        raw                0.4164   source text only
        raw+section        0.4813   best recall@20 (0.693), fewest total misses
        template           0.4919   original: boilerplate + section + topics + hype
        raw+hype           0.4930
        raw+section+hype   0.5154   <- default

    The ~458-character template header was expected to be the problem and is not:
    stripping it alone *lost* 7.5 nDCG points. Because the header is near-identical
    across all 1,562 chunks, IDF already neutralises it; what actually carries signal
    are the varying parts it happened to contain - the section path and the generated
    questions. Keeping those without the constant prose is what wins.
    """
    if mode == "template":
        return record["text"] + "\n" + "\n".join(record.get("hype_questions", []))
    if mode == "raw":
        return record["raw_text"]
    if mode == "raw+hype":
        return _context(record) + record["raw_text"] + "\n" + "\n".join(record.get("hype_questions", []))
    if mode == "raw+section":
        # The dense default. Context must reach this one above all: paraphrase failures
        # are semantic, and this is the text the bi-encoder actually embeds.
        section = " > ".join(record["metadata"].get("section_path", []))
        return f"{section}\n{_context(record)}{record['raw_text']}"
    if mode == "raw+section+hype":
        section = " > ".join(record["metadata"].get("section_path", []))
        hype = "\n".join(record.get("hype_questions", []))
        return f"{section}\n{_context(record)}{record['raw_text']}\n{hype}"
    raise ValueError(f"unknown index-text mode: {mode}")


def _context(record: dict) -> str:
    """Generated situating context, if apply_context.py has run.

    This is the only place the context enters the system. It joins the *indexed*
    text, never `raw_text`, so what a chunk claims the guideline says stays byte-for-
    byte what the PDF says - quotes, citation checking and the answer verifier are all
    unaffected. It leads the composed text so a short query matches it early and it
    sits inside every encoder's truncation window.

    Absent for corpora built without context, which is what makes the contextual and
    non-contextual variants comparable on the same code path.

    The "raw" and "template" modes deliberately exclude it: they exist to reproduce
    specific earlier runs, and silently changing what they mean would invalidate the
    comparisons in reports/ABLATION.md that were measured with them.
    """
    situating = record.get("context")
    return f"{situating}\n" if situating else ""


def build_tfidf(records: list[dict], mode: str) -> dict:
    texts = [indexable_text(record, mode) for record in records]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b[\w./<>-]+\b")
    matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "matrix": matrix}


def build_dense(records: list[dict], model_name: str, batch_size: int, mode: str) -> dict:
    """Embed the corpus.

    Hype questions are deliberately excluded from the dense text even though they
    help the lexical index: they are template-generated and near-identical across
    chunks, so they pull embeddings toward a common centroid. Section context is
    kept, because it disambiguates otherwise-similar passages.
    """
    import time

    from sentence_transformers import SentenceTransformer

    env = os.environ.copy()
    if "HF_TOKEN" in env:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])

    from nephrolex.retrieval.embedding_models import get as get_model

    from nephrolex.retrieval.medcpt import LOGICAL_NAME, MedCPTEncoder, is_medcpt

    spec = get_model(model_name)
    if is_medcpt(model_name):
        # A pair of separately-trained encoders, not one model with prefixes: the
        # corpus goes through the article encoder, queries through the query encoder.
        # Loading either through SentenceTransformer would mean-pool a [CLS] model.
        model = MedCPTEncoder("article")
        model_name = LOGICAL_NAME
    else:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    texts = [spec.passage_prefix + indexable_text(record, mode) for record in records]
    started = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    elapsed = time.time() - started

    device = str(getattr(model, "device", "unknown"))
    return {
        "model_name": model_name,
        "text_mode": mode,
        "query_prefix": spec.query_prefix,
        "passage_prefix": spec.passage_prefix,
        "device": device,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "encode_seconds": round(elapsed, 1),
        "chunks_per_second": round(len(texts) / elapsed, 1) if elapsed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full retrieval indexes for CKD evidence atoms.")
    # Defaulted to data/chunks/ (the superseded v1 corpus) while the system has
    # served data/chunks_v2/ for the whole project, so every real build had to pass
    # --corpus explicitly. Pointed at the canonical path instead.
    parser.add_argument("--corpus", default=str(SHIPPED_CORPUS))
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    parser.add_argument("--skip-dense", action="store_true", help="Build the lexical index only.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dense-text",
        default="raw+section",
        choices=["raw", "raw+section", "raw+section+hype"],
        help="What text is embedded. Hype questions are excluded by default.",
    )
    parser.add_argument(
        "--dense-out", default=None, help="Write embeddings here instead of the default path."
    )
    parser.add_argument(
        "--index-out", default=None,
        help="Directory for records/tfidf. Required when --corpus is not the shipped one.",
    )
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help="Permit an experimental corpus to replace the shipped indexes.",
    )
    parser.add_argument(
        "--index-text",
        default="raw+section+hype",
        choices=["raw", "raw+hype", "raw+section", "raw+section+hype", "template"],
        help="What text feeds the lexical indexes. 'template' reproduces the original runs.",
    )
    args = parser.parse_args()

    # records.json and tfidf.pkl are written to fixed paths, so building an
    # experimental corpus silently replaces the shipped lexical indexes - and the
    # served system then measures as that experiment. This happened twice while
    # evaluating Contextual Retrieval; the second time it was caught only because a
    # later baseline read 0.7041 instead of 0.7280. An experiment must name where it
    # goes.
    default_corpus = Path(args.corpus).resolve() == SHIPPED_CORPUS.resolve()
    if not default_corpus and not args.allow_overwrite and not args.index_out:
        raise SystemExit(
            f"refusing to overwrite the shipped indexes in {INDEX_DIR} with a build from\n"
            f"  {args.corpus}\n"
            "Pass --index-out DIR to write elsewhere, or --allow-overwrite if you really\n"
            "mean to replace what the demo and the evaluation both load."
        )

    index_dir = Path(args.index_out) if args.index_out else INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(Path(args.corpus))

    (index_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    with (index_dir / "tfidf.pkl").open("wb") as f:
        pickle.dump(build_tfidf(records, args.index_text), f)

    dense_status = {"enabled": False}
    if not args.skip_dense:
        dense = build_dense(records, args.dense_model, args.batch_size, args.dense_text)
        out_path = Path(args.dense_out) if args.dense_out else index_dir / "dense_embeddings.npy"
        np.save(out_path, dense["embeddings"])
        # The query is embedded at search time by whichever model is named on the CLI.
        # Recording the build model lets retrieve.py refuse a mismatch instead of
        # silently comparing vectors from two different embedding spaces.
        out_path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "model_name": dense["model_name"],
                    "text_mode": dense["text_mode"],
                    "query_prefix": dense["query_prefix"],
                    "passage_prefix": dense["passage_prefix"],
                    "device": dense["device"],
                    "shape": list(dense["embeddings"].shape),
                    "encode_seconds": dense["encode_seconds"],
                    "chunks_per_second": dense["chunks_per_second"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        dense_status = {
            "enabled": True,
            "model_name": dense["model_name"],
            "text_mode": dense["text_mode"],
            "shape": list(dense["embeddings"].shape),
            "encode_seconds": dense["encode_seconds"],
            "chunks_per_second": dense["chunks_per_second"],
            "output": str(out_path),
        }

    hype_records = []
    for record in records:
        for question in record.get("hype_questions", []):
            hype_records.append({"chunk_id": record["id"], "question": question, "metadata": record["metadata"]})
    (index_dir / "hype_questions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in hype_records), encoding="utf-8"
    )

    report = {
        "project": "Nephrolex",
        "stage": "full_accuracy_indexing",
        "record_count": len(records),
        "index_text_mode": args.index_text,
        "mean_indexed_chars": round(
            sum(len(indexable_text(record, args.index_text)) for record in records) / max(len(records), 1)
        ),
        "tfidf": str(INDEX_DIR / "tfidf.pkl"),
        "dense": dense_status,
        "hype_question_count": len(hype_records),
    }
    (REPORTS_DIR / "index_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
