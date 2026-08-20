"""Render the generated evaluation reports to a single PDF.

The reports are produced as markdown by make_eval_report.py and make_ablation_report.py
directly from the stored per-query runs, so this stage only typesets them - it never
restates a number. That ordering matters: the previous PDF in reports/ was written by
hand and still carried the pre-tuning baseline (nDCG@10 0.2760) eight hours after the
system measured 0.7280.

Uses pymupdf, already a dependency for citation checking, so there is nothing to
install. Markdown support is deliberately partial - headings, tables, lists, inline
emphasis and code - because that is all the generated reports use.

Usage:
    python scripts/make_eval_pdf.py
    python scripts/make_eval_pdf.py --sources reports/EVALUATION.md --out reports/eval.pdf
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [ROOT / "reports" / "EVALUATION.md", ROOT / "reports" / "ABLATION.md"]
DEFAULT_OUT = ROOT / "reports" / "CKD_Gold_Evaluation_and_Metrics_Specification.pdf"

CSS = """
body { font-family: sans-serif; font-size: 9.5px; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19px; color: #0b3d5c; margin-top: 14px; margin-bottom: 2px; }
h2 { font-size: 13px; color: #0b3d5c; margin-top: 15px; margin-bottom: 3px; }
h3 { font-size: 11px; color: #245c7a; margin-top: 11px; margin-bottom: 2px; }
p  { margin-top: 4px; margin-bottom: 4px; }
li { margin-top: 1px; margin-bottom: 1px; }
code { font-family: monospace; font-size: 9px; color: #7a2518; }
table { width: 100%; margin-top: 5px; margin-bottom: 7px; }
th { background-color: #e8eef2; font-size: 9px; padding: 3px; text-align: left; }
td { font-size: 9px; padding: 3px; border-bottom: 1px solid #dddddd; }
.sub { color: #566; font-size: 10px; margin-bottom: 10px; }
"""

INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"`(.+?)`"), r"<code>\1</code>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])"), r"<i>\1</i>"),
]


def inline(text: str) -> str:
    text = html.escape(text)
    for pattern, replacement in INLINE:
        text = pattern.sub(replacement, text)
    return text


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def alignments(separator: str) -> list[str]:
    out = []
    for cell in split_row(separator):
        if cell.endswith(":") and cell.startswith(":"):
            out.append("center")
        elif cell.endswith(":"):
            out.append("right")
        else:
            out.append("left")
    return out


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip())) and "-" in line


def to_html(markdown: str, shift: int = 0) -> str:
    """`shift` demotes every heading, so a source H1 sits under the document title."""
    lines = markdown.splitlines()
    out: list[str] = []
    index = 0
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # ---------------------------------------------------------- tables
        if stripped.startswith("|") and index + 1 < len(lines) and is_separator(lines[index + 1]):
            close_list()
            header = split_row(stripped)
            align = alignments(lines[index + 1])
            index += 2
            out.append("<table>")
            out.append("<tr>" + "".join(
                f'<th align="{align[i] if i < len(align) else "left"}">{inline(cell)}</th>'
                for i, cell in enumerate(header)
            ) + "</tr>")
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = split_row(lines[index])
                out.append("<tr>" + "".join(
                    f'<td align="{align[i] if i < len(align) else "left"}">{inline(cell)}</td>'
                    for i, cell in enumerate(cells)
                ) + "</tr>")
                index += 1
            out.append("</table>")
            continue

        # ------------------------------------------------------- headings
        heading = re.match(r"(#{1,4})\s+(.*)", stripped)
        if heading:
            close_list()
            level = min(len(heading.group(1)) + shift, 3)
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        # ---------------------------------------------------------- lists
        bullet = re.match(r"[-*]\s+(.*)", stripped)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(bullet.group(1))}</li>")
            index += 1
            continue

        if not stripped:
            close_list()
            index += 1
            continue

        close_list()
        out.append(f"<p>{inline(stripped)}</p>")
        index += 1

    close_list()
    return "\n".join(out)


# A single Story silently drops content once the document passes a certain size: this
# report rendered 4 complete pages, and adding one line of header made it 3 pages with
# a section missing and `place()` still reporting it had finished. No error, no partial
# page - the tail simply vanished. Every header variant reproduced it, so the trigger
# is total size rather than any particular markup.
#
# So the document is cut into pieces small enough to stay well clear of it, split at
# section boundaries. Each piece is its own Story, and make_eval_pdf verifies afterwards
# that every measured value in the source markdown appears in the PDF - which is what
# caught this in the first place.
BLOCK_BUDGET_CHARS = 4500


def split_blocks(html_body: str, budget: int = BLOCK_BUDGET_CHARS) -> list[str]:
    """Split rendered HTML at section boundaries into pieces under `budget` chars."""
    # Sources are shifted down a level, so their sections are <h3>; <h2> is the
    # document title. Split on both or the whole document stays one block.
    parts = re.split(r"(?=<h[23]>)", html_body)
    blocks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > budget:
            blocks.append(current)
            current = part
        else:
            current += part
    if current.strip():
        blocks.append(current)
    return blocks or [html_body]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=[str(p) for p in DEFAULT_SOURCES])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--title", default="NewbieDuo — CKD Guideline RAG")
    args = parser.parse_args()

    import pymupdf

    sources = [Path(p) for p in args.sources]
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise SystemExit(
            "missing source report(s): " + ", ".join(str(p) for p in missing)
            + "\nregenerate them first: python scripts/make_eval_report.py"
        )

    body = "\n".join(to_html(p.read_text(encoding="utf-8"), shift=1) for p in sources)
    header = (
        f"<h1>{html.escape(args.title)}</h1>"
        f'<p class="sub">Evaluation, metrics and retrieval ablation &#183; '
        f"generated {date.today().isoformat()} from "
        f'{", ".join(p.name for p in sources)}</p>'
    )

    # One Story per source document rather than one Story for all of them.
    # Concatenating the HTML silently dropped content: the combined document rendered
    # 6 pages and lost EVALUATION.md's closing notes while still emitting ABLATION.md's
    # sections, though EVALUATION.md alone renders complete in 4. Rendering each
    # document independently removes the interaction, and the per-source page counts
    # are checkable.
    writer = pymupdf.DocumentWriter(args.out)
    page_box = pymupdf.paper_rect("a4")
    content_box = page_box + (52, 46, -52, -46)

    documents: list[str] = []
    for position, source in enumerate(sources):
        blocks = split_blocks(to_html(source.read_text(encoding="utf-8"), shift=1))
        if position == 0:
            blocks[0] = header + blocks[0]
        documents.extend(blocks)

    pages = 0
    rendered: list[int] = []
    for body in documents:
        story = pymupdf.Story(html=f"<html><body>{body}</body></html>", user_css=CSS)
        more = True
        guard = 0
        while more and guard < 100:
            device = writer.begin_page(page_box)
            more, _ = story.place(content_box)
            story.draw(device)
            writer.end_page()
            pages += 1
            guard += 1
        if more:
            raise SystemExit(f"story did not finish within {guard} pages - refusing a truncated PDF")
        print(f"  block {len(rendered) + 1}: {guard} page(s)")
        rendered.append(guard)
    writer.close()

    # Verify the typeset output actually carries the current headline figure, so a
    # silently-truncated or stale render cannot pass as a finished deliverable.
    document = pymupdf.open(args.out)
    text = "\n".join(page.get_text() for page in document)
    print(f"wrote {args.out}")
    print(f"  {pages} pages, {len(text.split())} words, from {len(sources)} source report(s)")

    headline = re.search(r"nDCG@10[^\n]*", text)
    if headline:
        print(f"  contains: {headline.group(0)[:70]}")
    for source in sources:
        source_numbers = set(re.findall(r"0\.\d{4}", source.read_text(encoding="utf-8")))
        lost = sorted(source_numbers - set(re.findall(r"0\.\d{4}", text)))
        if lost:
            print(f"  WARNING  {len(lost)} value(s) from {source.name} did not render: {lost[:6]}")
            raise SystemExit(1)
    print("  all measured values from the source reports are present in the PDF")


if __name__ == "__main__":
    main()
