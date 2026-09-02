from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

try:
    import pymupdf
except ImportError:
    pymupdf = None


from nephrolex.paths import ROOT
RAW_DIR = ROOT / "data" / "raw"
PARSED_DIR = ROOT / "data" / "parsed"
DOCLING_DIR = ROOT / "data" / "docling"
REPORTS_DIR = ROOT / "reports"


DOCS = {
    "KDIGO-2024-CKD-Guideline.pdf": {
        "document_id": "kdigo_2024_ckd",
        "title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "publisher": "KDIGO",
        "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "version_year": 2024,
    },
    "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.pdf": {
        "document_id": "nice_ng203_ckd",
        "title": "NICE NG203 Chronic kidney disease: assessment and management",
        "publisher": "NICE",
        "source_url": "https://www.nice.org.uk/guidance/ng203",
        "version_year": 2021,
    },
}


@dataclass
class PageRecord:
    document_id: str
    title: str
    publisher: str
    source_url: str
    page: int
    parser_primary: str
    text: str
    pdfplumber_text: str
    pymupdf_text: str
    line_count: int
    word_count: int
    parser_agreement: float
    parser_flags: list[str]


@dataclass
class TableRecord:
    document_id: str
    title: str
    publisher: str
    page: int
    table_index: int
    parser: str
    rows: list[list[str]]
    row_count: int
    column_count: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return normalize_text(str(value)).replace("\n", " ").strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def parser_flags(pdfplumber_text: str, pymupdf_text: str, agreement: float) -> list[str]:
    flags: list[str] = []
    combined = pdfplumber_text + "\n" + pymupdf_text
    if len(pdfplumber_text) < 100 and len(pymupdf_text) < 100:
        flags.append("low_text_extraction")
    if agreement < 0.55:
        flags.append("parser_disagreement")
    if re.search(r"(eGFR|ACR|KFRE|G[1-5]|A[1-3])", combined):
        flags.append("ckd_threshold_page")
    if "|" in combined or re.search(r"\bTable\b", combined, re.I):
        flags.append("possible_table_page")
    return flags


def iter_pdf_outline(reader: PdfReader) -> list[str]:
    outline_items: list[str] = []

    def walk(items: Any) -> None:
        for item in items or []:
            if isinstance(item, list):
                walk(item)
                continue
            title = getattr(item, "title", None)
            if title:
                outline_items.append(str(title))

    try:
        walk(reader.outline)
    except Exception:
        return []
    return outline_items


def run_docling(docling_bin: str, force: bool, use_ocr: bool) -> dict[str, Any]:
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DOCLING_DIR.glob("*.json")) + list(DOCLING_DIR.glob("*.md"))
    if existing and not force:
        return {"status": "skipped_existing", "output_dir": str(DOCLING_DIR), "file_count": len(existing)}

    cmd = [
        docling_bin,
        "convert",
        str(RAW_DIR),
        "--from",
        "pdf",
        "--to",
        "json",
        "--to",
        "md",
        "--pipeline",
        "standard",
        "--tables",
        "--table-mode",
        "accurate",
        "--ocr" if use_ocr else "--no-ocr",
        "--device",
        "cpu",
        "--num-threads",
        "4",
        "--output",
        str(DOCLING_DIR),
        "-v",
    ]
    if use_ocr:
        insert_at = cmd.index("--device")
        cmd[insert_at:insert_at] = ["--ocr-mode", "pdf_aware_layout_regions"]

    env = os.environ.copy()
    if "HF_TOKEN" in env:
        env.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])
    env.setdefault("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "false")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)
    (REPORTS_DIR / "docling_stdout.log").write_text(result.stdout, encoding="utf-8")
    (REPORTS_DIR / "docling_stderr.log").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": cmd,
        "output_dir": str(DOCLING_DIR),
        "stdout_log": str(REPORTS_DIR / "docling_stdout.log"),
        "stderr_log": str(REPORTS_DIR / "docling_stderr.log"),
    }


def pymupdf_page_text(pdf_path: Path) -> list[str]:
    if pymupdf is None:
        return []
    doc = pymupdf.open(str(pdf_path))
    try:
        return [normalize_text(page.get_text("text")) for page in doc]
    finally:
        doc.close()


def parse_pdf(pdf_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    document_id = meta["document_id"]
    pages_path = PARSED_DIR / f"{document_id}_pages.jsonl"
    tables_path = PARSED_DIR / f"{document_id}_tables.jsonl"
    metadata_path = PARSED_DIR / f"{document_id}_metadata.json"

    page_count = 0
    table_count = 0
    table_rows = 0
    extraction_methods = Counter()
    reader = PdfReader(str(pdf_path))
    outlines = iter_pdf_outline(reader)
    pymupdf_pages = pymupdf_page_text(pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf, pages_path.open("w", encoding="utf-8") as page_out, tables_path.open(
        "w", encoding="utf-8"
    ) as table_out:
        for page_index, page in enumerate(pdf.pages, start=1):
            pdfplumber_text = normalize_text(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
            pymupdf_text = pymupdf_pages[page_index - 1] if page_index <= len(pymupdf_pages) else ""
            agreement = jaccard(pdfplumber_text, pymupdf_text)
            primary = pdfplumber_text if len(pdfplumber_text) >= len(pymupdf_text) * 0.75 else pymupdf_text
            text = normalize_text(primary)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            words = re.findall(r"\S+", text)
            page_record = PageRecord(
                document_id=document_id,
                title=meta["title"],
                publisher=meta["publisher"],
                source_url=meta["source_url"],
                page=page_index,
                parser_primary="pdfplumber" if primary == pdfplumber_text else "pymupdf",
                text=text,
                pdfplumber_text=pdfplumber_text,
                pymupdf_text=pymupdf_text,
                line_count=len(lines),
                word_count=len(words),
                parser_agreement=round(agreement, 4),
                parser_flags=parser_flags(pdfplumber_text, pymupdf_text, agreement),
            )
            page_out.write(json.dumps(asdict(page_record), ensure_ascii=False) + "\n")
            page_count += 1
            extraction_methods[page_record.parser_primary] += 1

            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table_index, table in enumerate(tables, start=1):
                rows = [[clean_cell(cell) for cell in row] for row in table if row]
                if not rows:
                    continue
                max_cols = max(len(row) for row in rows)
                normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
                table_record = TableRecord(
                    document_id=document_id,
                    title=meta["title"],
                    publisher=meta["publisher"],
                    page=page_index,
                    table_index=table_index,
                    parser="pdfplumber",
                    rows=normalized_rows,
                    row_count=len(normalized_rows),
                    column_count=max_cols,
                )
                table_out.write(json.dumps(asdict(table_record), ensure_ascii=False) + "\n")
                table_count += 1
                table_rows += len(normalized_rows)

    flagged_pages = []
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["parser_flags"]:
                flagged_pages.append(
                    {
                        "page": record["page"],
                        "flags": record["parser_flags"],
                        "parser_agreement": record["parser_agreement"],
                        "word_count": record["word_count"],
                    }
                )

    metadata = {
        **meta,
        "pdf_filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "sha256": file_sha256(pdf_path),
        "page_count": page_count,
        "table_count": table_count,
        "table_rows": table_rows,
        "pdf_metadata": {k: str(v) for k, v in (reader.metadata or {}).items()},
        "outline_titles": outlines,
        "extraction_methods": dict(extraction_methods),
        "flagged_pages": flagged_pages,
        "outputs": {
            "pages_jsonl": str(pages_path),
            "tables_jsonl": str(tables_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CKD PDFs with Docling plus PyMuPDF/pdfplumber cross-checks.")
    parser.add_argument("--run-docling", action="store_true", help="Run Docling JSON/Markdown conversion first.")
    parser.add_argument("--docling-ocr", action="store_true", help="Enable Docling OCR. Use for scanned or low-text pages.")
    parser.add_argument("--force-docling", action="store_true", help="Rerun Docling even if outputs already exist.")
    parser.add_argument("--docling-bin", default="docling", help="Path to the docling executable.")
    args = parser.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    docling_status = run_docling(args.docling_bin, args.force_docling, args.docling_ocr) if args.run_docling else None

    parsed: list[dict[str, Any]] = []
    missing: list[str] = []
    for filename, meta in DOCS.items():
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            missing.append(filename)
            continue
        parsed.append(parse_pdf(pdf_path, meta))

    summary = {
        "project": "Nephrolex",
        "stage": "full_accuracy_pdf_parsing",
        "docling": docling_status,
        "parsed_documents": parsed,
        "missing_documents": missing,
        "python": sys.executable,
    }
    summary_path = REPORTS_DIR / "day1_parsing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
