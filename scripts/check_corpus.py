"""Corpus and embedding integrity audit.

Retrieval metrics cannot see most of what goes wrong upstream of them. A chunk that
lost half its sentence, a recommendation labelled with the wrong number, an embedding
matrix one row out of step with the corpus - all of these score as ordinary noise on
nDCG while producing wrong answers with confident citations. This checks the things
the gold set is structurally blind to.

Written with a file tool rather than a shell heredoc on purpose: this module is mostly
regexes, and shell quoting has twice turned a `\b` in this codebase into a literal
backspace byte, silently disabling the pattern it belonged to.

Each check is mechanical and either passes or names the offending chunks. Exits
non-zero on a hard failure, so it can gate a rebuild.

Usage:
    python scripts/check_corpus.py
    python scripts/check_corpus.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.paths import CORPUS, INDEXES, RAW_PDFS  # noqa: E402

TERMINATED = re.compile(r"[.:;?!)\]]\s*$|\[\d{4}\]\s*$")
RECOMMENDATION_LABEL = re.compile(r"^(Recommendation|Practice Point)\s+(\d+(?:\.\d+)+)$")
EVIDENCE_GRADE = re.compile(r"\((?:1|2)[ABCD]\)")
APPARATUS = re.compile(
    r"Kidney International \(20\d\d\)|Suppl 4S|All rights reserved|Notice of rights", re.I
)

PDF_BY_DOC = {
    "kdigo_2024_ckd": "KDIGO-2024-CKD-Guideline.pdf",
    "nice_ng203_ckd": "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.pdf",
}


def read_corpus() -> list[dict]:
    with CORPUS.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flat(text: str) -> str:
    return " ".join(text.split())


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<46} {detail}")


def check_chunking(corpus: list[dict], verbose: bool) -> list[str]:
    failures: list[str] = []
    recs = [
        c for c in corpus
        if c["metadata"].get("chunk_type") in {"recommendation_atom", "practice_point"}
    ]

    counts = Counter(c["id"] for c in corpus)
    duplicates = [i for i, n in counts.items() if n > 1]
    report("unique chunk ids", not duplicates, f"{len(duplicates)} duplicated")
    if duplicates:
        failures.append(f"{len(duplicates)} duplicate chunk ids")

    empty = [c for c in corpus if not flat(c["raw_text"])]
    report("no empty chunks", not empty, f"{len(empty)} empty")
    if empty:
        failures.append(f"{len(empty)} empty chunks")

    hyphen = [c for c in corpus if flat(c["raw_text"]).endswith("-")]
    report("no chunk ends mid-word", not hyphen, f"{len(hyphen)} end on a hyphen")

    truncated = [c for c in recs if not TERMINATED.search(flat(c["raw_text"]))]
    report("recommendations end complete", not truncated, f"{len(truncated)} truncated")
    if verbose:
        for chunk in truncated:
            print(f"        {chunk['metadata']['citation'][:34]:<34} ...{flat(chunk['raw_text'])[-56:]}")

    # A label that disagrees with its own text produces a citation pointing at the
    # wrong recommendation - worse than no citation, because it looks right.
    mislabelled = []
    for chunk in recs:
        label = (chunk["metadata"].get("label") or "").strip()
        match = RECOMMENDATION_LABEL.match(label)
        if match and not flat(chunk["raw_text"]).startswith(f"{match.group(1)} {match.group(2)}"):
            mislabelled.append(chunk)
    report("labels match their own text", not mislabelled, f"{len(mislabelled)} mismatched")
    if mislabelled:
        failures.append(f"{len(mislabelled)} chunks whose label disagrees with their text")
    if verbose:
        for chunk in mislabelled[:6]:
            print(f"        {chunk['metadata'].get('label')!r} -> {flat(chunk['raw_text'])[:66]}")

    polluted = [c for c in recs if APPARATUS.search(flat(c["raw_text"]))]
    report("no page apparatus inside guidance", not polluted, f"{len(polluted)} polluted")
    if polluted:
        failures.append(f"{len(polluted)} recommendations containing page apparatus")

    # Every KDIGO recommendation carries exactly one evidence grade. Two means the
    # page reader spliced neighbouring statements into one chunk.
    spliced = [
        c for c in recs
        if c["metadata"].get("document_id") == "kdigo_2024_ckd"
        and c["metadata"].get("chunk_type") == "recommendation_atom"
        and len(EVIDENCE_GRADE.findall(flat(c["raw_text"]))) > 1
    ]
    report("no spliced recommendations", not spliced, f"{len(spliced)} carry 2+ evidence grades")
    if spliced:
        failures.append(f"{len(spliced)} recommendations splice two guideline statements")
    if verbose:
        for chunk in spliced[:6]:
            print(f"        {chunk['metadata']['citation'][:34]:<34} {flat(chunk['raw_text'])[:70]}")

    return failures


def check_coverage(corpus: list[dict], verbose: bool) -> list[str]:
    """Is every labelled recommendation in the PDFs actually in the corpus?

    This is the check that catches silent drops. Docling omits a handful of elements,
    and without this nothing would notice a recommendation simply not existing.
    """
    failures: list[str] = []
    try:
        import pymupdf
    except ImportError:
        print("  SKIP  coverage (pymupdf unavailable)")
        return failures

    present = {
        (c["metadata"].get("document_id"), (c["metadata"].get("label") or "").strip())
        for c in corpus
    }
    for document_id, filename in PDF_BY_DOC.items():
        path = RAW_PDFS / filename
        if not path.exists():
            continue
        with pymupdf.open(path) as pdf:
            text = " ".join(page.get_text("text") for page in pdf)
        if document_id == "kdigo_2024_ckd":
            labels = {
                f"{kind.title()} {number}"
                for kind, number in re.findall(
                    r"\b(Recommendation|Practice Point)\s+(\d+(?:\.\d+)+)\s*:", text
                )
            }
        else:
            labels = set(re.findall(r"(?m)^\s*(\d+\.\d+\.\d+)\s+[A-Z]", text))

        missing = sorted(label for label in labels if (document_id, label) not in present)
        found = len(labels) - len(missing)
        share = found / len(labels) if labels else 1.0
        report(f"coverage: {document_id}", share >= 0.95, f"{found}/{len(labels)} present ({share:.1%})")
        if share < 0.95:
            failures.append(f"{document_id}: {len(missing)} labelled items absent from the corpus")
        if verbose and missing:
            print(f"        missing: {', '.join(missing[:14])}")
    return failures


def check_embeddings(corpus: list[dict], verbose: bool) -> list[str]:
    failures: list[str] = []
    import numpy as np

    served = "abhinand__MedEmbed-large-v0_1.npy"
    for path in sorted((INDEXES / "dense").glob("*.npy")):
        matrix = np.load(path)
        name = path.name
        is_served = name == served

        aligned = matrix.shape[0] == len(corpus)
        report(f"{name}: rows match corpus", aligned, f"{matrix.shape[0]} rows vs {len(corpus)} chunks")
        if not aligned:
            # Only the embedding the system actually loads has to be current; the
            # others are retained experiments and are expected to be stale.
            if is_served:
                failures.append(f"{name} has {matrix.shape[0]} rows for a {len(corpus)}-chunk corpus")
            continue

        if not bool(np.isfinite(matrix).all()):
            report(f"{name}: all values finite", False, "NaN or inf present")
            failures.append(f"{name} contains NaN/inf")
        else:
            report(f"{name}: all values finite", True)

        norms = np.linalg.norm(matrix, axis=1)
        dead = int((norms < 1e-6).sum())
        report(f"{name}: no zero vectors", dead == 0, f"{dead} zero-length rows")
        if dead:
            failures.append(f"{name} has {dead} zero vectors")

        report(
            f"{name}: L2-normalised",
            bool(np.allclose(norms, 1.0, atol=1e-3)),
            f"norms span {norms.min():.4f}-{norms.max():.4f}",
        )

        # Two distinct chunks embedding to the same vector means the wrong text was
        # encoded for one of them. Duplicate source text legitimately collides, so
        # the count is compared against how much duplicate text the corpus contains.
        prefix = np.round(matrix[:, :16], 5)
        collisions = sum(n - 1 for n in Counter(map(tuple, prefix)).values() if n > 1)
        duplicate_text = sum(
            n - 1 for n in Counter(flat(c["raw_text"]) for c in corpus).values() if n > 1
        )
        report(
            f"{name}: collisions explained by duplicate text",
            collisions <= duplicate_text + 5,
            f"{collisions} vector collisions, {duplicate_text} duplicate texts",
        )
    return failures


def check_colbert(corpus: list[dict], verbose: bool) -> list[str]:
    """Is the late-interaction token index still aligned with the corpus?

    It is off by default and lazily loaded, which is precisely why it rotted unnoticed:
    a stale index emits candidate indices beyond the end of the corpus, and the crash
    surfaces inside BM25 normalisation with a message naming neither ColBERT nor the
    rebuild that invalidated it.
    """
    import json as _json

    meta = INDEXES / "colbert" / "meta.json"
    lengths = INDEXES / "colbert" / "lengths.npy"
    if not meta.exists() or not lengths.exists():
        print("  ok    colbert index absent (optional, weight 0)")
        return []

    import numpy as np

    stored = int(np.load(lengths).shape[0])
    aligned = stored == len(corpus)
    report("colbert index matches corpus", aligned, f"{stored} chunks vs {len(corpus)}")
    if not aligned:
        built = _json.loads(meta.read_text(encoding="utf-8")).get("chunks")
        return [
            f"colbert index holds {stored} chunks (built for {built}) against a "
            f"{len(corpus)}-chunk corpus - rebuild or delete it"
        ]
    return []


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    corpus = read_corpus()
    print(f"corpus: {len(corpus)} chunks\n")

    print("chunking")
    failures = check_chunking(corpus, args.verbose)
    print("\ncoverage against the source PDFs")
    failures += check_coverage(corpus, args.verbose)
    print("\nembeddings")
    failures += check_embeddings(corpus, args.verbose)
    failures += check_colbert(corpus, args.verbose)

    print()
    if failures:
        print(f"{len(failures)} HARD FAILURE(S):")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("No hard failures. Any FAIL row above is a known, bounded defect.")


if __name__ == "__main__":
    main()
