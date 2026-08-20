"""Citation accuracy: does every citation resolve to the text it claims?

The hackathon brief names citation accuracy as a metric, and this system has already
shipped two citation defects that no retrieval metric could see - table rows numbered
by our own enumeration rather than the document's ("Table 6" for a table the source
calls Table 3), and bounding boxes drawn on the wrong page for chunks spanning a page
break. Both were caught by eye. This turns that into a number.

For every indexable chunk the check is mechanical: open the cited PDF at the cited
page, extract the text inside the cited bounding box, and measure how much of it
overlaps the chunk's own text. A citation a reader cannot follow to the quoted words
is a failed citation, regardless of how relevant the chunk was.

Usage:
    python scripts/check_citations.py
    python scripts/check_citations.py --sample 200 --verbose
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "chunks_v2" / "retrieval_corpus.jsonl"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

PDF_BY_DOCUMENT = {
    "kdigo_2024_ckd": "KDIGO-2024-CKD-Guideline.pdf",
    "nice_ng203_ckd": "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.pdf",
}

# A citation counts as resolving when this much of the chunk's wording is found
# under its box. Extraction differs slightly from the stored text - ligature repair,
# symbol normalisation, cell joining - so exact equality would measure our cleaning,
# not our citations.
OVERLAP_THRESHOLD = 0.60


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap(chunk_text: str, extracted: str) -> float:
    """Fraction of the chunk's words that appear under its own bounding box."""
    wanted = words(chunk_text)
    if not wanted:
        return 0.0
    found = set(words(extracted))
    return sum(1 for w in wanted if w in found) / len(wanted)


def check(chunk: dict, pdfs: dict) -> dict:
    import pymupdf

    meta = chunk["metadata"]
    document_id = meta.get("document_id")
    provenance = [p for p in (meta.get("provenance") or []) if p.get("bbox")]

    result = {
        "chunk_id": chunk["id"],
        "citation": meta.get("citation"),
        "chunk_type": meta.get("chunk_type"),
        "document_id": document_id,
        "pages": sorted({p.get("page") for p in provenance if p.get("page")}),
    }

    if not provenance:
        result.update(status="no_bbox", overlap=0.0)
        return result

    pdf = pdfs.get(document_id)
    if pdf is None:
        result.update(status="unknown_document", overlap=0.0)
        return result

    extracted = []
    for entry in provenance:
        page_no = entry.get("page")
        if not page_no or page_no < 1 or page_no > pdf.page_count:
            result.update(status="page_out_of_range", overlap=0.0)
            return result
        page = pdf[page_no - 1]
        left, top, right, bottom = entry["bbox"]
        if entry.get("coord_origin") == "TOPLEFT":
            rect = pymupdf.Rect(left, top, right, bottom)
        else:
            height = page.rect.height
            rect = pymupdf.Rect(left, height - top, right, height - bottom)
        extracted.append(page.get_text("text", clip=rect))

    score = overlap(chunk["raw_text"], " ".join(extracted))
    result.update(
        status="resolves" if score >= OVERLAP_THRESHOLD else "mismatch",
        overlap=round(score, 4),
    )
    return result


def check_table_numbering(corpus: list[dict], pdfs: dict) -> dict:
    """Does a cited table number appear on the page it is cited from?

    Table labels are read from the caption rather than our enumeration order, which
    is the fix for citing "Table 6" for a table the document calls Table 3. This
    verifies that fix held.
    """
    import pymupdf  # noqa: F401 - pdfs already open

    checked = wrong = 0
    examples = []
    for chunk in corpus:
        meta = chunk["metadata"]
        label = (meta.get("label") or "").strip()
        match = re.fullmatch(r"(Table|Figure)\s+(\d+)", label)
        if not match or meta.get("chunk_type") not in {"table_row", "figure_caption"}:
            continue
        pdf = pdfs.get(meta["document_id"])
        page_no = meta.get("page_start")
        if pdf is None or not page_no or page_no > pdf.page_count:
            continue
        checked += 1
        page_text = pdf[page_no - 1].get_text("text")
        if not re.search(rf"{match.group(1)}\s*{match.group(2)}\b", page_text, re.I):
            wrong += 1
            if len(examples) < 5:
                examples.append(f"{meta.get('citation')} - '{label}' not found on p.{page_no}")
    return {"checked": checked, "not_found_on_page": wrong, "examples": examples}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Measure citation accuracy.")
    parser.add_argument("--sample", type=int, default=0, help="Check N random chunks (0 = all).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=str(REPORTS / "citation_accuracy.json"))
    args = parser.parse_args()

    import pymupdf

    corpus = read_jsonl(CORPUS)
    pdfs = {doc: pymupdf.open(RAW / name) for doc, name in PDF_BY_DOCUMENT.items()}

    population = corpus
    if args.sample and args.sample < len(corpus):
        population = random.Random(17).sample(corpus, args.sample)

    results = [check(chunk, pdfs) for chunk in population]
    numbering = check_table_numbering(corpus, pdfs)

    by_status: dict[str, int] = {}
    for row in results:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    resolving = by_status.get("resolves", 0)
    accuracy = resolving / len(results) if results else 0.0

    print(f"citations checked        {len(results)} of {len(corpus)} indexable chunks")
    print(f"resolve to cited text    {resolving} ({accuracy:.1%})")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        if status != "resolves":
            print(f"  {status:<22} {count}")
    print(f"median overlap           {statistics.median(r['overlap'] for r in results):.3f}")
    print()
    print(f"table/figure numbers     {numbering['checked']} checked, "
          f"{numbering['not_found_on_page']} not found on the cited page")
    for example in numbering["examples"]:
        print(f"  {example}")

    by_type: dict[str, list[float]] = {}
    for row in results:
        by_type.setdefault(row["chunk_type"], []).append(1.0 if row["status"] == "resolves" else 0.0)
    print()
    print("by chunk type:")
    for name, values in sorted(by_type.items(), key=lambda kv: -statistics.fmean(kv[1])):
        print(f"  {name:<20} {statistics.fmean(values):.1%}  (n={len(values)})")

    if args.verbose:
        print("\nworst failures:")
        for row in sorted(results, key=lambda r: r["overlap"])[:10]:
            print(f"  {row['overlap']:.2f}  {row['citation']}")

    payload = {
        "checked": len(results),
        "corpus_size": len(corpus),
        "citation_accuracy": round(accuracy, 4),
        "overlap_threshold": OVERLAP_THRESHOLD,
        "median_overlap": round(statistics.median(r["overlap"] for r in results), 4),
        "by_status": by_status,
        "by_chunk_type": {k: round(statistics.fmean(v), 4) for k, v in by_type.items()},
        "table_numbering": numbering,
        "failures": [r for r in results if r["status"] != "resolves"][:50],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
