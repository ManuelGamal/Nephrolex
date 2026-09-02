"""Build the CKD evidence corpus from Docling's layout-aware parse.

Why this replaces the page-text chunker
---------------------------------------
KDIGO 2024 is a two-column journal PDF. The previous chunker consumed PyMuPDF
linear page text, which reads straight across both columns and interleaves them
line by line. Measured against the Docling output, 913 of 1,069 KDIGO chunks
(85%) were not contiguous text. Example of the same practice point:

    page text : "An eGFRcr level <90 ml/min per population of interest and which
                 has been shown to be most 1.73 m2 can be flagged as 'low' ..."
    Docling   : "An eGFRcr level <90 ml/min per 1.73 m2 can be flagged as 'low'
                 in children and adolescents over the age of 2 years."

Docling emits elements in corrected reading order with a label, a page number and
a bounding box per element. That gives three things the old pipeline could not
have: uncorrupted sentences, real section paths for citations, and the bbox
needed to highlight a cited sentence on the rendered PDF page.

Outputs (written to data/chunks_v2/ so the existing corpus stays intact):
    all_chunks.jsonl       every chunk, including non-indexable, for traceability
    retrieval_corpus.jsonl indexable chunks only, in the shape build_indexes.py reads
    chunking_summary.json  counts and quality-flag breakdown

Usage:
    python scripts/chunk_docling.py
    python scripts/chunk_docling.py --out data/chunks_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


from nephrolex.paths import ROOT
DOCLING_DIR = ROOT / "data" / "docling"
DEFAULT_OUT = ROOT / "data" / "chunks_v2"
REPORTS_DIR = ROOT / "reports"

# Narrative text is packed into passages of roughly this size before a new chunk
# is started. Recommendations and practice points are never split.
PASSAGE_TARGET_CHARS = 900
MIN_CHUNK_CHARS = 90


DOCUMENTS = [
    {
        "document_id": "kdigo_2024_ckd",
        "docling": "KDIGO-2024-CKD-Guideline.json",
        "document": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "short": "KDIGO 2024",
        "publisher": "KDIGO",
        "version_year": 2024,
        "source_url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "pdf": "KDIGO-2024-CKD-Guideline.pdf",
    },
    {
        "document_id": "nice_ng203_ckd",
        "docling": "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.json",
        "document": "NICE NG203 Chronic kidney disease: assessment and management",
        "short": "NICE NG203",
        "publisher": "NICE",
        "version_year": 2021,
        "source_url": "https://www.nice.org.uk/guidance/ng203",
        "pdf": "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.pdf",
        # Docling treats NICE's "1.1.13" recommendation numbers as list markers and
        # drops them. Those numbers are exactly what a clinician cites, so they are
        # recovered from the page text. NICE is single-column, so unlike KDIGO its
        # page text was never column-interleaved and is safe to use for this.
        "recover_numbers_from": "nice_ng203_ckd_pages.jsonl",
    },
]

PARSED_DIR = ROOT / "data" / "parsed"


TOPIC_PATTERNS = {
    "ckd_definition": r"\b(defined as|definition of ckd|markers of kidney damage|abnormalities of kidney)\b",
    "egfr": r"\b(egfr|gfr|creatinine|cystatin)\b",
    "albuminuria_acr": r"\b(albuminuria|acr\b|uacr|proteinuria|albumin[- ]creatinine)\b",
    "risk_stratification": r"\b(risk of|prognosis|risk categor|adverse outcomes|classification of ckd)\b",
    "kfre": r"\b(kidney failure risk equation|kfre|5-year risk|2-year risk)\b",
    "referral": r"\b(refer|referral|specialist kidney care|nephrology|specialist assessment)\b",
    "monitoring": r"\b(monitor|frequency of monitoring|follow-up|repeat testing)\b",
    "medication_safety": r"\b(nsaid|nephrotoxic|drug dos|contrast|metformin|over-the-counter|dose adjust)\b",
    "ras_inhibitors": r"\b(ace inhibitor|arb\b|angiotensin|renin-angiotensin|rasi)\b",
    "sglt2": r"\b(sglt2|sodium-glucose)\b",
    "blood_pressure": r"\b(blood pressure|systolic|hypertension|mmhg)\b",
    "diabetes": r"\b(diabet|hba1c|glycemic|glycaemic)\b",
    "anemia": r"\b(anaemia|anemia|haemoglobin|hemoglobin|erythropo|\besa\b|ferritin|iron)\b",
    "mineral_bone": r"\b(phosphate|parathyroid|calcium|vitamin d|ckd-mbd|bone)\b",
    "dialysis": r"\b(dialysis|kidney replacement therapy|renal replacement|transplant)\b",
    "pregnancy": r"\b(pregnan|gestation|lactation|breastfeed)\b",
    "pediatric": r"\b(child|children|young people|adolescen|infant|paediatric|pediatric)\b",
    "diet": r"\b(protein intake|dietary|sodium intake|salt intake|nutrition)\b",
}

# Out-of-declared-scope topics; used by the safety gate, never as a retrieval boost.
SCOPE_FLAG_TOPICS = {"pregnancy", "pediatric", "dialysis"}


# --------------------------------------------------------------------- cleaning

LIGATURE_TAIL = re.compile(r"\b(ffi|ffl|fi|fl|ff)\s+(?=[a-z])")
LIGATURE_HEAD = re.compile(r"([a-z])\s+(ffi|ffl|fi|fl|ff)(?=[a-z])")

SYMBOL_FIXES = [
    ("‡", ">="),   # mangled >= in KDIGO body text
    ("†", ""),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("’", "'"),
    ("‘", "'"),
    ("“", '"'),
    ("”", '"'),
    ("–", "-"),
    ("—", "-"),
    ("�", ""),
]


COMMON_WORDS = {
    "the", "and", "for", "with", "categories", "category", "range", "description",
    "risk", "kidney", "disease", "albuminuria", "increased", "decreased", "normal",
    "moderately", "severely", "mildly", "persistent", "prognosis", "min", "per",
}


def repair_mirrored_text(text: str) -> str:
    """Reverse text that a PDF extractor read right-to-left.

    Rotated table axis labels come out backwards, e.g.
        ")2m 37.1/nim/lm( egnar dna seirogetac noitpircseD RFG"
    which reversed is
        "GFR Description categories and range (ml/min/1.73 m2)"

    Detection is by evidence, not by rule: whichever direction yields more
    recognisable words wins, so ordinary text is never touched.
    """
    if len(text) < 12:
        return text
    reversed_text = text[::-1]

    def score(candidate: str) -> int:
        return sum(1 for word in re.findall(r"[a-z]{3,}", candidate.lower()) if word in COMMON_WORDS)

    return reversed_text if score(reversed_text) > score(text) else text


def normalize_text(text: str) -> str:
    """Repair the artefacts Docling leaves in text extracted from these PDFs."""
    if not text:
        return ""
    for bad, good in SYMBOL_FIXES:
        text = text.replace(bad, good)
    # Docling splits ligatures into standalone tokens: "fl agged", "de fi ned".
    # Neither "fi" nor "fl" is an English word, so rejoining is safe.
    text = LIGATURE_TAIL.sub(r"\1", text)
    text = LIGATURE_HEAD.sub(r"\1\2", text)
    text = re.sub(r"\bm\s+2\b", "m2", text)
    # The PDF font maps the "greater than or equal" glyph to a dollar sign, so KDIGO's
    # thresholds come out as "urine ACR $ 30 mg/g" and "eGFR $ 60 ml/min". 69 places,
    # against 71 where the real glyph survived - the same threshold written both ways.
    # It is wrong text on a citable chunk before it is a retrieval problem, and it also
    # hides the bound from the numeric band matcher, which is why "is 95 in the normal
    # band?" could not match the G1 row. Currency is not a risk here: every occurrence
    # has a space before the digit, none is adjacent, and no cost word appears near one.
    text = re.sub(r"\$\s+(?=\d)", "≥", text)
    text = re.sub(r"(\d)\s*/\s*4\b", r"\1", text)
    text = re.sub(r"\bN\s*1/4\s*", "N = ", text)
    return repair_mirrored_text(" ".join(text.split()).strip())


# --------------------------------------------------------------------- sections

# A chunk whose text ends without terminal punctuation is unfinished; one that opens
# lower-case is the continuation of something unfinished. Together they identify a
# recommendation split across a page break without over-matching a new sentence.
# NICE repeats each recommendation's number above a rationale heading:
#     text          "Recommendation 1.7.2"
#     section_header "Why the committee made the recommendation"
# Number recovery bound the number to the heading, producing a chunk whose entire
# text was "1.7.2 Why the committee made the recommendation" - cited as if it were
# recommendation 1.7.2. A wrong citation is worse than a missing one, so these
# headings are never treated as recommendations.
# KDIGO's journal page footer is emitted as an ordinary element and was being
# absorbed into the recommendation above it, so Recommendation 3.6.2 ended
# "...with or without diabetes (1B). S158 Kidney International (2024) 105 (Suppl 4S),
# S117-S314" - a citation string presented to a reader as part of clinical guidance.
# Every KDIGO recommendation carries exactly one evidence grade, at its end: "(1A)",
# "(2C)". Two of them in one recovered block means the column-aware page reader
# spliced neighbouring recommendations together - which produced a Recommendation
# 3.6.2 reading "...A2 and A3) with diabetes (1B)" when the guideline actually says
# "...A2) without diabetes (2C)". Wrong population, wrong grade, fully citable.
# A hole in the corpus is recoverable; a confidently wrong recommendation is not.
# Which chunks *define* a category, as opposed to reporting something stratified by one.
#
# "eGFR 38, which band?" is answered with certainty by the row stating 30-44 for the
# same quantity. Retrieval cannot reach it by similarity - 38 appears nowhere in
# "30 - 44" - and the numeric band signal cannot single it out either, because KDIGO
# Tables 23, 24 and 29 report laboratory values *by* GFR category and state the very
# same ranges. To the retriever those rows look identical to Table 2's, and all of them
# score a perfect 1.0 on band containment.
#
# The guidelines already draw the distinction, in the caption. "Table 2| GFR categories
# in CKD" enumerates the categories; "Table 24| Variation of laboratory values ... by
# age group, sex, and eGFR" is stratified by them and announces a different subject.
# This reads that declaration instead of guessing from the row contents, so it stays a
# property of the corpus rather than a rule fitted to the questions asked of it.
DEFINES_CATEGORY = re.compile(
    r"\b(?:GFR|albuminuria|proteinuria|ACR)\s+categor(?:y|ies)\b"
    r"|\bcategories\s+(?:in|of)\s+(?:CKD|chronic kidney disease)\b"
    r"|\bnomenclature\b"
    r"|\bclassification\s+of\s+CKD\b",
    re.I,
)

# "by <x> category" is the stratification marker; the others announce another subject.
# "rating guideline recommendations" is the grading nomenclature (1A, 2B), not a
# disease category.
STRATIFIED_BY_CATEGORY = re.compile(
    r"\bby\b[^.]{0,40}\bcategor|\bvariation of\b|\brisk of\b|\bimpact of\b"
    r"|\bassociations?\b|\boutcomes\b|\brating guideline\b|\bindications\b",
    re.I,
)

# Docling loses some table captions, leaving the literal "Table" as the leaf. The
# section directly above names the table in every case that matters here.
GENERIC_LEAF = {"table", "figure", ""}


def defines_categories(section_path: list[str] | None) -> bool:
    """Does this chunk sit in a table or section that enumerates disease categories?"""
    parts = [part for part in (section_path or []) if part and part.strip()]
    if not parts:
        return False
    caption = parts[-1]
    if caption.strip().lower() in GENERIC_LEAF and len(parts) > 1:
        caption = parts[-2]
    return bool(DEFINES_CATEGORY.search(caption)) and not bool(
        STRATIFIED_BY_CATEGORY.search(caption)
    )


EVIDENCE_GRADE_RE = re.compile(r"\((?:1|2)[ABCD]\)")

JOURNAL_FOOTER_RE = re.compile(
    r"\s*S?\d{0,4}\s*Kidney International\s*\(\d{4}\).*$", re.I)

RATIONALE_HEADING_RE = re.compile(
    r"^(why the committee|how the recommendations? might affect|"
    r"what the .* found|the committee('s)? discussion)", re.I)

TERMINATED_RE = re.compile(r"[.:;?!)\]]\s*$|\[\d{4}\]\s*$")
CONTINUES_RE = re.compile(r"^[a-z(]")

RECOMMENDATION_RE = re.compile(r"^(Recommendation|Practice Point)\s+(\d+(?:\.\d+)+)\s*:?\s*", re.I)
NICE_REC_RE = re.compile(r"^(\d+\.\d+\.\d+)\s+")
NICE_SECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.+)$")
CHAPTER_RE = re.compile(r"^(Chapter\s+\d+|Appendix\s+[A-Z])\b", re.I)


def header_level(text: str) -> int:
    """Infer heading depth so section paths nest correctly."""
    if CHAPTER_RE.match(text):
        return 0
    if NICE_REC_RE.match(text):
        return 2
    if NICE_SECTION_RE.match(text):
        return 1
    if re.match(r"^\d+(\.\d+){2,}\s", text):
        return 2
    if text.isupper() and len(text) > 8:
        return 0
    return 2


class SectionStack:
    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, text: str) -> None:
        level = header_level(text)
        self._stack = [(lvl, txt) for lvl, txt in self._stack if lvl < level]
        self._stack.append((level, text))

    @property
    def path(self) -> list[str]:
        return [txt for _, txt in self._stack]


# ----------------------------------------------------------------- term mining

THRESHOLD_RE = re.compile(
    r"(?:eGFR|GFR|ACR|UACR|albumin|creatinine|potassium|haemoglobin|hemoglobin|ferritin|"
    r"systolic|blood pressure|bicarbonate|protein)\s*"
    r"(?:of\s+|is\s+)?(?:<|>|>=|<=|less than|more than|at least|below|above|under|over)?\s*"
    r"\d+(?:\.\d+)?\s*"
    r"(?:mg/mmol|mg/g|ml/min|mmol/l|g/l|mmHg|micrograms?/litre|%|g/kg)?",
    re.I,
)
CATEGORY_RE = re.compile(r"\b(G[1-5][ab]?|A[1-3])\b")


def extract_exact_terms(text: str) -> list[str]:
    terms = {match.group(0).strip() for match in THRESHOLD_RE.finditer(text)}
    terms |= set(CATEGORY_RE.findall(text))
    return sorted(term for term in terms if len(term) > 1)[:12]


def detect_topics(text: str) -> list[str]:
    lower = text.lower()
    topics = [name for name, pattern in TOPIC_PATTERNS.items() if re.search(pattern, lower)]
    return topics or ["general_ckd"]


def build_hype_questions(chunk_type: str, topics: list[str], text: str, label: str | None) -> list[str]:
    """Template question generation.

    Measured on the gold set, adding these to the lexical index is worth about
    +0.077 nDCG@10 over indexing source text alone, because they inject
    question-shaped phrasing that user queries actually match.
    """
    questions: list[str] = []
    if label:
        questions.append(f"What does {label} say?")
    templates = {
        "referral": "When should a person with CKD be referred to specialist kidney care?",
        "monitoring": "How often should CKD be monitored?",
        "egfr": "How should eGFR be used in CKD evaluation?",
        "albuminuria_acr": "How should albuminuria or ACR be measured in CKD?",
        "kfre": "When should the Kidney Failure Risk Equation be used?",
        "sglt2": "When should an SGLT2 inhibitor be used in CKD?",
        "ras_inhibitors": "When should ACE inhibitors or ARBs be used in CKD?",
        "blood_pressure": "What blood pressure target applies in CKD?",
        "anemia": "How should anaemia of CKD be managed?",
        "mineral_bone": "How should mineral and bone disorder in CKD be managed?",
        "medication_safety": "Which medicines need care in CKD?",
        "ckd_definition": "How is chronic kidney disease defined?",
        "risk_stratification": "How is risk classified in CKD?",
        "diet": "What dietary advice applies in CKD?",
    }
    for topic in topics:
        if topic in templates:
            questions.append(templates[topic])
    for term in extract_exact_terms(text)[:3]:
        questions.append(f"What CKD guidance mentions {term}?")
    return list(dict.fromkeys(questions))[:6]


# ----------------------------------------------------------------- chunk record

# Apparatus rather than guidance: contents pages, figure/table indexes, the
# evidence-review supplement. These were reaching the retrieval corpus and
# competing with recommendations for the same queries.
NON_GUIDANCE_TEXT = re.compile(
    r"S\d{3}\s+[A-Z].{0,60}?S\d{3}\s+"
    r"|(?:Figure|Table)\s*\d+[.|].{0,400}?(?:Figure|Table)\s*\d+[.|]"
    r"|Work Group membership|Executive Committee|Reference keys|Conversion factors"
    r"|Abbreviations and acronyms|SUPPLEMENT(?:ARY)?\s+(?:TO|MATERIAL)|Patient foreword"
    r"|disclosure|Search strategies|PRISMA|Appendix [A-Z][.\s]"
    # KDIGO's methods appendix: the PICOS review scaffolding behind the guideline,
    # not the guideline. 74 such chunks were outranking the actual definitions.
    r"|PICOS|systematic review topics|Existing systematic review"
    r"|Clinical questions? and systematic|Evidence Review Team",
    re.I,
)

NON_GUIDANCE_SECTION = re.compile(
    r"^(FIGURES|TABLES|SUPPLEMENTARY MATERIAL|Abbreviations|Reference keys|Notice"
    r"|Conversion factors|Work Group|Biographic|Methods for guideline development"
    r"|Appendix|Data supplement|Summary of ?findings|Clinical question"
    r"|Methods for guideline|Systematic review)",
    re.I,
)


# Top-level sections that are apparatus rather than guidance.
#
# This began as an allowlist of guidance sections, which is the safer shape - but it
# cannot work here, because section roots are not reliable. Docling's heading order
# leaves KDIGO's page-22 staging tables rooted under "EVIDENCE REVIEW TEAM", a front
# matter heading the section stack never popped. Allowlisting by root therefore
# discarded the GFR and albuminuria category tables, which are core content.
#
# So this is a denylist, but a *complete* one: every distinct root section in both
# documents was enumerated and classified, rather than patterns being added as noise
# was noticed on screen. `check_corpus_hygiene` in build_gold_eval.py asserts the
# result, so a regression fails the build instead of surfacing in a demo.
APPARATUS_ROOTS = [
    r"^Contents$",
    r"^VOL \d+ \| ISSUE",
    r"^SECTION [I]+:",
    r"^Appendix\b",
    r"^SUPPLEMENTARY MATERIAL$",
    r"^KDIGO EXECUTIVE COMMITTEE$",
    r"^WORK GROUP",
    r"^METHODS COMMITTEE",
    r"^Your responsibility$",
    r"^Overview$",
    r"^Who is it for\?$",
    r"^Chronic kidney disease: assessment and management$",   # NICE title page
    # Research recommendations describe what future studies should investigate,
    # which reads like guidance but tells a clinician nothing about care.
    r"^Chapter 6",
]


def in_guidance_section(document_id: str, section_path: list[str]) -> bool:
    """False when the chunk sits under a section that carries no clinical guidance."""
    root = (section_path or [""])[0].strip()
    if not root:
        return False
    return not any(re.search(pattern, root, re.I) for pattern in APPARATUS_ROOTS)


def is_non_guidance(text: str, section_path: list[str]) -> bool:
    if NON_GUIDANCE_TEXT.search(text):
        return True
    return any(NON_GUIDANCE_SECTION.match(part.strip()) for part in section_path)


FIGURE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s*\d+", re.I)


def _is_unlabelled_numeric_row(text: str) -> bool:
    """Reject hazard-ratio grids: rows that are bare numbers with no column labels.

    KDIGO's pp.118/125 risk figures extract as cells like "1.6: Ref 1.0; 2.2: 1.3"
    with no row or column meaning attached. They are unciteable and pure retrieval
    noise.
    """
    # The row text carries its caption as a prefix for retrievability. Judging the
    # whole string counts the caption's words as "labels" and hides that the row
    # itself is nothing but numbers, which is how Figure 6's hazard-ratio grid
    # slipped through. Judge the cells only.
    body = re.split(r"\bRow \d+:\s*", text, maxsplit=1)[-1]
    body = re.sub(r"^Table[^.]*\.\s*", "", body)
    body = re.sub(r"^row \d+\.\s*", "", body)

    # Table-of-contents rows leak in as "1.3 Frequency of monitoring ......... 63".
    if "......" in body:
        return True
    # Evidence-review bookkeeping, not clinical guidance.
    if re.search(r"citations screened|prisma|records identified|studies included", body, re.I):
        return True
    # Abbreviation glossaries pair a short acronym with its expansion, repeatedly.
    glossary = re.findall(r"\b[A-Z][A-Za-z]{1,6}\b\s*:", body)
    if len(glossary) >= 3 and len(body) < 400:
        return True

    tokens = [t.strip() for t in re.split(r"[;:]", body) if t.strip()]
    if len(tokens) < 3:
        return False

    # A hazard-ratio grid repeats one column header across every column, so half
    # the tokens look like labels while carrying no information. Judge the numeric
    # share among *distinct* tokens instead.
    distinct = list(dict.fromkeys(tokens))
    numeric = sum(1 for t in distinct if re.fullmatch(r"[\d.,\s%<>=()-]+|Ref [\d.]+", t))
    return numeric / len(distinct) >= 0.7


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:limit] or "chunk"


# Chunk IDs issued during this build, so a collision can be detected and renamed.
_ISSUED_IDS: set[str] = set()


def make_chunk(
    doc: dict,
    chunk_type: str,
    text: str,
    section_path: list[str],
    provenance: list[dict],
    label: str | None = None,
    parent_id: str | None = None,
    seq: int = 0,
) -> dict:
    pages = [p["page"] for p in provenance] or [0]
    page_start, page_end = min(pages), max(pages)
    topics = detect_topics(text)
    exact_terms = extract_exact_terms(text)

    anchor = slugify(label or text[:60])
    chunk_id = f"{doc['document_id']}_p{page_start}_{chunk_type}_{anchor}_{seq}"

    # Two different tables on one page can carry the same label - an unlabelled table
    # and "Table 1", or a table split across a page - so their rows built identical
    # IDs for different content. Eight such collisions existed, and while retrieval is
    # positional and was unaffected, anything keyed by chunk_id was not: contextual
    # retrieval would have attached one chunk's generated context to another.
    #
    # Only the colliding chunk is renamed, by a hash of its own text. Every other ID
    # is untouched, which matters because the gold set stores resolved chunk IDs.
    # The hash alone is not enough: two rows with identical text hash identically and
    # would still collide, so the loop falls back to a counter and always terminates.
    if chunk_id in _ISSUED_IDS:
        base = f"{chunk_id}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]}"
        chunk_id, attempt = base, 2
        while chunk_id in _ISSUED_IDS:
            chunk_id = f"{base}_{attempt}"
            attempt += 1
    _ISSUED_IDS.add(chunk_id)

    citation = f"{doc['short']}"
    if label:
        citation += f", {label}"
    elif section_path:
        citation += f", {section_path[-1]}"
    citation += f", p.{page_start}" if page_start == page_end else f", pp.{page_start}-{page_end}"

    quality_flags: list[str] = []
    # Only narrative passages are dropped for length. A short recommendation is
    # still a recommendation - "Do not use age alone to determine treatment of
    # anaemia of CKD" is 76 characters and fully citable.
    if chunk_type == "section_passage" and len(text) < MIN_CHUNK_CHARS:
        quality_flags.append("too_short")
    elif len(text) < 25:
        quality_flags.append("too_short")
    if chunk_type == "table_row" and _is_unlabelled_numeric_row(text):
        quality_flags.append("unlabelled_numeric_table")
    if is_non_guidance(text, section_path):
        quality_flags.append("non_guidance")
    if not in_guidance_section(doc["document_id"], section_path):
        quality_flags.append("outside_guidance_section")
    if chunk_type == "table_row" and FIGURE_CAPTION.match(text):
        # Docling extracts the data grid behind a chart as a table. Those numbers
        # are plot coordinates, and a row of them reads as evidence while meaning
        # nothing without the chart.
        quality_flags.append("figure_data")
    # A recommendation that is only its own number is a cross-reference heading,
    # not guidance: "Recommendation 1.5.1: to 1.5.9" carries no content but was
    # ranking first, because the type prior favours recommendations.
    if chunk_type in {"recommendation_atom", "practice_point"}:
        body = re.sub(r"^(?:Recommendation|Practice Point)?\s*[\d.]+:?\s*", "", text).strip()
        if len(body) < 40:
            quality_flags.append("stub_recommendation")
    scope_flags = sorted(set(topics) & SCOPE_FLAG_TOPICS)

    return {
        "id": chunk_id,
        "document_id": doc["document_id"],
        "document": doc["document"],
        "publisher": doc["publisher"],
        "source_url": doc["source_url"],
        "chunk_type": chunk_type,
        "label": label,
        # Figures and tables this chunk defers to. Self-references are dropped so a
        # figure body does not point at itself.
        "references": sorted(
            {
                f"{m.group(1).title()} {m.group(2)}"
                for m in CROSS_REFERENCE_RE.finditer(text)
            }
            - {(label or "").strip()}
        ),
        "page_start": page_start,
        "page_end": page_end,
        "provenance": provenance,
        "section_path": section_path,
        "raw_text": text,
        "topics": topics,
        "exact_terms": exact_terms,
        "citation": citation,
        "parent_id": parent_id,
        "citable": chunk_type != "router_summary",
        "indexable": not quality_flags,
        "quality_flags": quality_flags,
        "scope_flags": scope_flags,
        "hype_questions": build_hype_questions(chunk_type, topics, text, label),
    }


# --------------------------------------------------------------------- parsing

def element_prov(element: dict) -> list[dict]:
    out = []
    for prov in element.get("prov") or []:
        bbox = prov.get("bbox") or {}
        out.append(
            {
                "page": prov.get("page_no"),
                "bbox": [bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b")],
                "coord_origin": bbox.get("coord_origin", "BOTTOMLEFT"),
            }
        )
    return out


def build_number_recovery(doc: dict) -> dict[int, list[tuple[str, str]]]:
    """Map each page to the (normalised text prefix -> recommendation number) pairs
    visible in that page's raw text, so numbers Docling dropped can be restored."""
    source = doc.get("recover_numbers_from")
    if not source:
        return {}
    path = PARSED_DIR / source
    if not path.exists():
        return {}

    by_page: dict[int, list[tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            page = json.loads(line)
            text = " ".join((page.get("text") or "").split())
            pairs: list[tuple[str, str]] = []
            for match in re.finditer(r"\b(\d+\.\d+\.\d+)\s+([A-Za-z][^\n]{0,90})", text):
                pairs.append((_key(match.group(2)), match.group(1)))
            by_page[page["page"]] = pairs
    return by_page


KEY_LENGTH = 64
MIN_KEY_OVERLAP = 30

# How much of the shorter of (chunk key, page candidate) the match must cover.
# Below this the overlap is a shared opening, not the same sentence.
COMPLETE_MATCH_RATIO = 0.9

# Every NICE recommendation closes with a year stamp - "[2021]", "[2008, amended
# 2021]". It marks the end of that recommendation, and is needed as a boundary
# because the lookahead otherwise relies on recognising the *next* number: once
# number recovery correctly refuses an ambiguous one, that boundary disappears and
# the following recommendation is swallowed by the previous one.
YEAR_STAMP_RE = re.compile(r"\[\d{4}[^\]]*\]\s*$")


def _key(text: str, length: int = KEY_LENGTH) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:length]


def _common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def recover_number(recovery: dict[int, list[tuple[str, str]]], page: int | None, text: str) -> str | None:
    """Restore a NICE recommendation number that Docling dropped as a list marker.

    Matching must be unambiguous, not merely close. NICE 1.6.1 and 1.6.2 both begin
    "In adults with CKD and an ACR ..." and differ only at "under 70" vs "of 70 or
    more", so a short prefix match would silently attach the wrong number to the
    chunk - a wrong citation, which is worse than no citation. The best candidate
    must therefore beat the runner-up by a clear margin.
    """
    if not recovery or page is None:
        return None
    key = _key(text)
    if len(key) < MIN_KEY_OVERLAP:
        return None

    scored = sorted(
        ((_common_prefix(key, candidate), number) for candidate, number in recovery.get(page, [])),
        reverse=True,
    )
    if not scored or scored[0][0] < MIN_KEY_OVERLAP:
        return None
    if len(scored) > 1 and scored[1][0] >= scored[0][0]:
        return None  # ambiguous: refuse rather than guess a citation

    # A tie is not the only way to pick the wrong number. If the true recommendation
    # is missing from the page table there is nothing to tie against, and a merely
    # long partial overlap wins by default: NICE 1.1.11 ("...proteinuria in adults")
    # took 1.1.10's number ("...proteinuria in children") on a 44-character overlap
    # of a 64-character key, because 1.1.11 was absent from the table entirely. The
    # match must therefore be near-complete against the shorter string, not just
    # longer than everyone else's.
    best, number = scored[0]
    candidate = next(c for c, n in recovery.get(page, []) if n == number)
    if best < COMPLETE_MATCH_RATIO * min(len(key), len(candidate)):
        return None
    return number


def find_references_cutoff(texts: list[dict]) -> int:
    """Index of the References heading; everything after it is bibliography."""
    for idx in range(len(texts) - 1, -1, -1):
        element = texts[idx]
        if element.get("label") != "section_header":
            continue
        if re.match(r"^references\s*$", (element.get("text") or "").strip(), re.I):
            return idx
    return len(texts)


def collect_captions(texts: list[dict]) -> dict[int, list[tuple[float, str]]]:
    """Caption elements keyed by page, with their vertical position."""
    by_page: dict[int, list[tuple[float, str]]] = {}
    for element in texts:
        if element.get("label") != "caption":
            continue
        for prov in element.get("prov") or []:
            page = prov.get("page_no")
            top = (prov.get("bbox") or {}).get("t", 0.0) or 0.0
            by_page.setdefault(page, []).append((top, normalize_text(element.get("text") or "")))
    return by_page


def nearest_caption(captions: dict[int, list[tuple[float, str]]], provenance: list[dict]) -> str:
    """Docling leaves `captions` empty on most of these tables, but emits the caption
    as a separate text element on the same page. Match them back by proximity."""
    if not provenance:
        return ""
    page = provenance[0].get("page")
    candidates = captions.get(page) or []
    if not candidates:
        return ""
    table_top = (provenance[0].get("bbox") or [None, 0, None, None])[1] or 0.0
    return min(candidates, key=lambda item: abs(item[0] - table_top))[1]


DOC_TABLE_NUMBER = re.compile(r"^\s*Table\s+(\d+)", re.I)


def document_table_label(caption: str, page: int | None) -> str:
    """The table number the *document* uses, not our enumeration order.

    Tables were previously labelled by their index in Docling's table array, so
    NICE's "Table 3 Example of high-dose intravenous iron regimen" was cited as
    "Table 6" - a table number that does not exist in the document. A citation a
    reader cannot follow is worse than no citation, so the number is read from the
    caption, and where no caption states one the label falls back to the page.
    """
    match = DOC_TABLE_NUMBER.match(caption or "")
    if match:
        return f"Table {match.group(1)}"
    if caption:
        return f"Table ({caption.strip()[:48]})"
    return "Table"


def row_provenance(table: dict, row_index: int, page: int | None) -> list[dict] | None:
    """Bounding box of one table row, from the union of that row's cell boxes.

    Without this every row of a table inherits the *table's* box, so citing a single
    row highlights the whole table - on KDIGO's abbreviations page that meant one
    citation drawing a rectangle over 73% of the page, and 40 rows sharing one box.

    Cell boxes are TOPLEFT-origin while element boxes are BOTTOMLEFT, so the origin
    is recorded per entry rather than assumed downstream.
    """
    cells = (table.get("data") or {}).get("table_cells") or []
    boxes = [
        cell["bbox"]
        for cell in cells
        if cell.get("bbox") and cell.get("start_row_offset_idx") == row_index
    ]
    if not boxes:
        return None
    return [
        {
            "page": page,
            "bbox": [
                min(b["l"] for b in boxes),
                min(b["t"] for b in boxes),
                max(b["r"] for b in boxes),
                max(b["b"] for b in boxes),
            ],
            "coord_origin": boxes[0].get("coord_origin", "TOPLEFT"),
        }
    ]


# A recommendation may carry its content by reference: Practice Point 5.1.1 is
# entirely "Refer adults with CKD to specialist kidney care services in the
# circumstances listed in Figure 48". Recording the target lets the answer layer
# supply that figure as evidence in its own right, which is the only way to resolve
# the reference without splicing text into a chunk that is quoted verbatim.
CROSS_REFERENCE_RE = re.compile(r"\b(Figure|Table)\s*(\d+)\b", re.I)


# A figure is only worth rebuilding if its content is readable text. KDIGO draws two
# very different things as figures: flowcharts whose boxes carry criteria (Figure 48
# is the entire referral criteria set, 12% numeric) and data plots whose "text" is
# axis ticks and cell values (Figure 5 is a risk heatmap, 86% numeric). Reassembling
# the second kind produced a 2,457-character wall of hazard ratios - "1.6 2.2 2.9 4.3
# 5.8 1.1 1.4 ..." - which matched band queries because it contains "30-44" and was
# then quoted to the reader as evidence. The caption chunk still describes those
# figures; only the unreadable body is dropped.
MAX_FIGURE_NUMERIC_SHARE = 0.35


def numeric_share(text: str) -> float:
    """Fraction of word-or-number tokens that are numbers."""
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token[0].isdigit()) / len(tokens)


# Fragments whose vertical centres fall within this many points are treated as one
# visual row of the figure, so a flowchart box reads left-to-right rather than by
# whatever order the extractor happened to emit.
FIGURE_ROW_BAND = 6.0


def figure_regions(payload: dict) -> dict[int, list[dict]]:
    """Picture regions carrying a Figure/Table caption, grouped by page.

    KDIGO draws several of its most clinically useful artefacts as flowcharts - the
    whole referral criteria set is Figure 48. Docling emits such a figure as one
    picture region plus dozens of loose text fragments, one per box, in no useful
    order. Those fragments previously fell through to whatever narrative passage was
    being accumulated, which is why Practice Point 5.1.1's "circumstances listed in
    Figure 48" resolved to a caption containing no criteria, while the criteria
    themselves sat scrambled inside an unrelated passage.
    """
    texts = payload.get("texts") or []
    regions: dict[int, list[dict]] = {}
    for picture in payload.get("pictures") or []:
        prov = (picture.get("prov") or [{}])[0]
        bbox, page = prov.get("bbox"), prov.get("page_no")
        if not bbox or not page:
            continue
        label = None
        for ref in picture.get("captions") or []:
            reference = str(ref.get("$ref", "")).rsplit("/", 1)[-1]
            if not reference.isdigit() or int(reference) >= len(texts):
                continue
            caption = normalize_text(texts[int(reference)].get("text") or "")
            match = re.match(r"\s*((?:Figure|Table)\s*\d+)", caption, re.I)
            if match:
                label = match.group(1).replace("|", "").strip()
        if label:
            regions.setdefault(page, []).append({"bbox": bbox, "label": label})
    return regions


def _within_region(bbox: dict, region: dict) -> bool:
    """Is this fragment's centre inside the picture region?

    Centres rather than corners: a box border may overlap the region edge, and a
    fragment that merely touches the figure is not part of it.
    """
    area = region["bbox"]
    x = (bbox["l"] + bbox["r"]) / 2
    y = (bbox["t"] + bbox["b"]) / 2
    return (
        area["l"] <= x <= area["r"]
        and min(area["b"], area["t"]) <= y <= max(area["b"], area["t"])
    )


def figure_fragment_indices(payload: dict, regions: dict[int, list[dict]]) -> set[int]:
    """Text elements that belong to a figure, so the main loop can skip them."""
    inside: set[int] = set()
    for index, element in enumerate(payload.get("texts") or []):
        prov = (element.get("prov") or [{}])[0]
        page, bbox = prov.get("page_no"), prov.get("bbox")
        if not page or not bbox or element.get("label") == "caption":
            continue
        if any(_within_region(bbox, region) for region in regions.get(page, [])):
            inside.add(index)
    return inside


def parse_figures(
    doc: dict, payload: dict, regions: dict[int, list[dict]], sections_by_page: dict
) -> list[dict]:
    """Rebuild each figure's body from its fragments, in reading order."""
    texts = payload.get("texts") or []
    chunks: list[dict] = []
    seq = 0
    for page, page_regions in sorted(regions.items()):
        for region in page_regions:
            fragments: list[tuple[float, float, str]] = []
            for element in texts:
                prov = (element.get("prov") or [{}])[0]
                if prov.get("page_no") != page or not prov.get("bbox"):
                    continue
                if element.get("label") == "caption" or not _within_region(prov["bbox"], region):
                    continue
                body = normalize_text(element.get("text") or "")
                if not body or body.lower().startswith(region["label"].lower()):
                    continue
                bbox = prov["bbox"]
                fragments.append(((bbox["t"] + bbox["b"]) / 2, bbox["l"], body))

            if not fragments:
                continue
            # Bottom-left origin, so a larger y sits higher on the page. Banding the
            # vertical centre groups a visual row; sorting by x within it reads that
            # row left to right.
            fragments.sort(key=lambda f: (-round(f[0] / FIGURE_ROW_BAND), f[1]))
            body = f"{region['label']}: " + " ".join(fragment[2] for fragment in fragments)
            if len(body) < MIN_CHUNK_CHARS:
                continue
            if numeric_share(body) >= MAX_FIGURE_NUMERIC_SHARE:
                # A data plot, not guidance. Its caption chunk still describes it.
                continue

            seq += 1
            provenance = [{
                "page": page,
                "bbox": [region["bbox"]["l"], region["bbox"]["t"],
                         region["bbox"]["r"], region["bbox"]["b"]],
                "coord_origin": region["bbox"].get("coord_origin", "BOTTOMLEFT"),
            }]
            chunks.append(
                make_chunk(
                    doc, "figure_body", body,
                    sections_by_page.get(page, []), provenance,
                    label=region["label"], seq=seq,
                )
            )
    return chunks


def parse_tables(
    doc: dict,
    tables: list[dict],
    captions: dict[int, list[tuple[float, str]]],
    sections: dict[int, list[str]] | None = None,
) -> list[dict]:
    """Emit one chunk per table row, carrying enough context to be retrievable.

    A bare row like "G5; Kidney failure; <15" is unfindable for "at what eGFR is
    someone in kidney failure" - it never says eGFR and carries no units. Each row
    therefore inherits the table's caption and its repaired header row.

    Rows also inherit the document section they sit under. Without it a table under
    "Abbreviations and acronyms" looks like ordinary content to the non-guidance
    filter, and KDIGO's acronym glossary ends up as 39 retrievable chunks.
    """
    chunks: list[dict] = []
    for table_no, table in enumerate(tables, start=1):
        grid = (table.get("data") or {}).get("grid") or []
        if not grid:
            continue
        provenance = element_prov(table)
        caption = nearest_caption(captions, provenance)
        page = provenance[0].get("page") if provenance else None
        section = list((sections or {}).get(page) or [])

        header_cells = [normalize_text(cell.get("text", "")) for cell in grid[0]] if grid else []
        header_is_usable = sum(1 for cell in header_cells if len(cell) > 1) >= 2
        offset = 1 if header_is_usable else 0

        for row_no, row in enumerate(grid[offset:], start=1):
            values = [normalize_text(cell.get("text", "")) for cell in row]
            if sum(1 for value in values if value) < 2:
                continue
            if header_is_usable:
                # A cell whose column header is blank keeps its value unlabelled rather
                # than being dropped. Requiring `header` discarded exactly the cells
                # that identify the row: in a cross-tab the corner cell is empty by
                # convention, so every row label went with it. NICE's monitoring-
                # frequency table came out as six rows of bare counts - "A1: 1; A2: 1;
                # A3: 2" - with nothing saying which GFR category the row was for.
                # 245 label cells across 25 tables were affected.
                #
                # Where the table has no caption, the header text becomes the row's
                # context (below) and is not repeated per cell. Doing both made each
                # NICE monitoring row state the three ACR definitions twice with the
                # answering numbers buried between them; doing neither is worse still,
                # because a bare "G1; Normal or high; >=90" never says "GFR" and cannot
                # be found by a question that does - which is what the header text is
                # there to supply.
                pairs = [
                    f"{header}: {value}" if header and caption else value
                    for header, value in zip(header_cells, values)
                    if value
                ]
                body = "; ".join(pairs) if pairs else "; ".join(v for v in values if v)
            else:
                body = "; ".join(value for value in values if value)
            if len(body) < 12:
                continue
            label = document_table_label(caption, page)
            # Context first, so the row reads as a statement about a named thing.
            #
            # The headers go in the context or in the body, never both. An uncaptioned
            # table used to fall back to the header text as its context while the body
            # also prefixed every cell with its header, so each row of NICE's
            # monitoring-frequency table stated the three ACR category definitions
            # twice and the numbers that answer the question were buried between them.
            # Docling repeats a spanning header across every column it covers, so a
            # header row can be the same phrase seven times. Emitted verbatim that
            # became keyword stuffing: each row of Table 23 ("Variation of laboratory
            # values ... by age group, sex, and eGFR") carried "GFR category (ml/min
            # per 1.73 m2)" seven times and outranked the actual GFR staging table for
            # "which GFR category covers 60 to 89 ml/min". Repeating a term does not
            # make a chunk more about it; the header names the columns once.
            distinct_headers = list(dict.fromkeys(c for c in header_cells if c))
            context = caption or " ".join(distinct_headers).strip()
            if header_is_usable and caption:
                context = f"{caption} ({' / '.join(distinct_headers)})"
            text = f"{context}. Row {row_no}: {body}" if context else f"{label} row {row_no}. {body}"

            row_prov = row_provenance(table, row_no - 1 + offset, page)
            if row_prov and provenance:
                context_box = dict(provenance[0])
                context_box["role"] = "table_context"
                row_prov = row_prov + [context_box]
            row_prov = row_prov or provenance
            chunks.append(
                make_chunk(
                    doc, "table_row", text,
                    section + [caption or label],
                    row_prov, label=label, seq=row_no,
                )
            )
    return chunks


def parse_document(doc: dict) -> list[dict]:
    payload = json.loads((DOCLING_DIR / doc["docling"]).read_text(encoding="utf-8"))
    texts = payload.get("texts", [])
    cutoff = find_references_cutoff(texts)
    recovery = build_number_recovery(doc)
    regions = figure_regions(payload)
    figure_fragments = figure_fragment_indices(payload, regions)

    stack = SectionStack()
    sections_by_page: dict[int, list[str]] = {}
    chunks: list[dict] = []
    passage_parts: list[str] = []
    passage_prov: list[dict] = []
    passage_section: list[str] = []
    seq = 0

    def flush_passage() -> None:
        nonlocal passage_parts, passage_prov, passage_section, seq
        body = normalize_text(" ".join(passage_parts))
        if len(body) >= MIN_CHUNK_CHARS:
            # A glossary entry stores its term in the heading and its meaning in the
            # body, so the definition chunk never names what it defines. NICE's CKD
            # definition reads "Abnormalities of kidney function or structure present
            # for more than 3 months" - no "chronic kidney disease" anywhere in it,
            # which makes it unretrievable for "how is CKD defined". Restore the term.
            heading = passage_section[-1] if passage_section else ""
            if heading and len(heading) < 70 and not NICE_SECTION_RE.match(heading):
                head_key = re.sub(r"[^a-z]", "", heading.lower())[:16]
                if head_key and head_key not in re.sub(r"[^a-z]", "", body.lower())[:220]:
                    body = f"{heading}: {body}"
            seq += 1
            chunks.append(
                make_chunk(doc, "section_passage", body, passage_section, passage_prov, seq=seq)
            )
        passage_parts, passage_prov, passage_section = [], [], []

    idx = 0
    while idx < cutoff:
        element = texts[idx]
        label_type = element.get("label")
        text = normalize_text(element.get("text") or "")

        if label_type in {"page_header", "page_footer"} or not text:
            idx += 1
            continue

        # Fragments belonging to a figure are rebuilt as one chunk by parse_figures.
        # Left in the stream they join whatever passage is open, which both scrambles
        # the figure and pollutes the passage: Figure 48's referral criteria arrived
        # as "Further evaluation and specialist management based on diagnosis Causes
        # Circumstances category ..." inside an unrelated narrative chunk.
        if idx in figure_fragments:
            flush_passage()
            idx += 1
            continue

        if label_type == "caption":
            # KDIGO renders its monitoring-frequency grid as an image, so Docling
            # extracts no table for it at all - the caption is the only searchable
            # description that exists. Buried inside a neighbouring passage it could
            # never win a query like "frequency of GFR monitoring".
            flush_passage()
            figure = re.match(r"^\s*((?:Figure|Table)\s*\d+)", text, re.I)
            seq += 1
            chunks.append(
                make_chunk(
                    doc, "figure_caption", text, stack.path, element_prov(element),
                    label=figure.group(1).replace("|", "").strip() if figure else None,
                    seq=seq,
                )
            )
            idx += 1
            continue

        page_no = (element.get("prov") or [{}])[0].get("page_no")
        rec_match = RECOMMENDATION_RE.match(text)
        # NICE numbers are recovered from page text rather than matched inline.
        # The inline pattern is not used for KDIGO, where "1.1.1 Detection of CKD"
        # is a subsection heading, not a recommendation.
        recovered = None
        if recovery and label_type in {"list_item", "section_header", "text"} and not rec_match:
            # Some NICE recommendations keep their number inline (they are emitted as
            # section headers ending in ":" with their criteria as following bullets);
            # most have it stripped as a list marker and need recovery from page text.
            inline = NICE_REC_RE.match(text)
            if inline:
                recovered = inline.group(1)
                text = NICE_REC_RE.sub("", text).strip()
            else:
                recovered = recover_number(recovery, page_no, text)

        # A numbered heading that introduces a recommendation absorbs the list
        # items beneath it, so "1.5.5 Refer adults ... if they have:" stays with
        # its own criteria rather than being split from them.
        if label_type == "section_header" and not (rec_match or recovered):
            flush_passage()
            stack.push(text)
            if page_no:
                sections_by_page[page_no] = stack.path
            idx += 1
            continue

        if (rec_match or recovered) and RATIONALE_HEADING_RE.match(
                RECOMMENDATION_RE.sub("", text).strip()):
            # A rationale heading, not guidance. Fall through to passage handling so
            # the text is still searchable, just never citable as a recommendation.
            recovered = None
            rec_match = None

        if (rec_match or recovered) and label_type == "section_header":
            # Docling sometimes emits a heading that is a truncated duplicate of the
            # list_item beneath it - NICE 1.11.5 appeared as "Before starting phosphate
            # binders for adults, children and young people", stopping before the
            # "optimise:" clause and its list. Prefer the fuller element.
            probe = idx + 1
            while probe < cutoff and (
                texts[probe].get("label") in {"page_header", "page_footer"}
                or not normalize_text(texts[probe].get("text") or "")
            ):
                probe += 1
            if probe < cutoff:
                follower = normalize_text(texts[probe].get("text") or "")
                bare = RECOMMENDATION_RE.sub("", text).strip()
                if len(follower) > len(bare) and follower.startswith(bare[:60]):
                    idx += 1
                    continue

        if rec_match or recovered:
            flush_passage()
            if rec_match:
                kind = "recommendation_atom" if rec_match.group(1).lower() == "recommendation" else "practice_point"
                ref = f"{rec_match.group(1).title()} {rec_match.group(2)}"
                text = RECOMMENDATION_RE.sub("", text).strip()
                text = f"{ref}: {text}"
            else:
                kind = "recommendation_atom"
                ref = recovered
                text = f"{ref} {text}"

            parts = [text]
            provenance = element_prov(element)
            lookahead = idx + 1
            while lookahead < cutoff:
                nxt = texts[lookahead]
                nxt_text = normalize_text(nxt.get("text") or "")
                if nxt.get("label") in {"page_header", "page_footer"}:
                    lookahead += 1
                    continue
                if not nxt_text:
                    lookahead += 1
                    continue
                # A recommendation that runs over a page break has its remainder
                # emitted as a plain `text` element rather than a `list_item`, so the
                # label test alone abandoned it. NICE 1.1.20 ended at "...who are
                # taking" and lost its entire drug list - the only clinically useful
                # half - to page 11. Absorb a continuation instead: the text so far
                # has no terminal punctuation and the next element opens lower-case,
                # which a new sentence, heading or recommendation never does.
                joined = " ".join(parts)
                if YEAR_STAMP_RE.search(joined):
                    break
                continuation = bool(
                    not TERMINATED_RE.search(joined) and CONTINUES_RE.match(nxt_text)
                )
                if nxt.get("label") != "list_item" and not continuation:
                    break
                nxt_page = (nxt.get("prov") or [{}])[0].get("page_no")
                if RECOMMENDATION_RE.match(nxt_text) or recover_number(recovery, nxt_page, nxt_text):
                    break
                if parts[-1].endswith("-") and CONTINUES_RE.match(nxt_text):
                    # Line-break hyphenation: "ensure docu-" + "mentation" is one word.
                    parts[-1] = parts[-1][:-1] + nxt_text
                else:
                    parts.append(nxt_text)
                provenance.extend(element_prov(nxt))
                lookahead += 1

            seq += 1
            body = JOURNAL_FOOTER_RE.sub("", " ".join(parts)).strip()
            chunks.append(
                make_chunk(doc, kind, body, stack.path, provenance, label=ref, seq=seq)
            )
            idx = lookahead
            continue

        if page_no and page_no not in sections_by_page:
            sections_by_page[page_no] = stack.path
        passage_parts.append(text)
        passage_prov.extend(element_prov(element))
        if not passage_section:
            passage_section = stack.path
        if sum(len(part) for part in passage_parts) >= PASSAGE_TARGET_CHARS:
            flush_passage()
        idx += 1

    flush_passage()
    chunks.extend(parse_figures(doc, payload, regions, sections_by_page))
    chunks.extend(parse_tables(doc, payload.get("tables", []), collect_captions(texts), sections_by_page))
    chunks.extend(backfill_missing_recommendations(doc, chunks))
    return chunks


REC_BLOCK_RE = re.compile(
    r"\b(Recommendation|Practice Point)\s+(\d+(?:\.\d+)+)\s*:\s*(.{40,700}?)"
    r"(?=\s*(?:Recommendation|Practice Point)\s+\d|\Z)")


def _best_page_reading(page) -> str:
    """Whichever reading of the page yields self-consistent recommendations.

    Column-aware reading is right on most two-column pages and wrong on some. On
    p.43 it spliced Recommendation 3.6.2 into its neighbours - "...A2 and A3) with
    diabetes (1B)" for a recommendation that actually reads "...A2) without diabetes
    (2C)" - while PyMuPDF's plain reading order got it exactly right. Neither method
    wins everywhere, so both are tried and scored: a recommendation carrying exactly
    one evidence grade is intact, and more intact recommendations is the better read.
    """
    candidates = [
        normalize_text(_column_aware_page_text(page)),
        normalize_text(page.get_text("text")),
    ]

    def intact(text: str) -> int:
        return sum(
            1 for m in REC_BLOCK_RE.finditer(text)
            if len(EVIDENCE_GRADE_RE.findall(m.group(3))) == 1
        )

    return max(candidates, key=intact)


def _column_aware_page_text(page) -> str:
    """Read a two-column page in the correct order.

    PyMuPDF's plain `get_text()` reads straight across the page, interleaving the
    columns - the defect that corrupted 85% of the original KDIGO corpus. Reading
    blocks and sorting by (column, vertical position) recovers the true order.
    """
    blocks = [b for b in page.get_text("blocks") if (b[4] or "").strip()]
    if not blocks:
        return ""
    midpoint = page.rect.width / 2
    blocks.sort(key=lambda b: (0 if (b[0] + b[2]) / 2 < midpoint else 1, b[1]))
    return " ".join(" ".join(b[4].split()) for b in blocks)


def backfill_missing_recommendations(doc: dict, chunks: list[dict]) -> list[dict]:
    """Recover recommendations that Docling's parse omits entirely.

    Docling drops a small number of elements (2 of 170 labelled recommendations in
    KDIGO). Rather than accept holes in the corpus, those are re-extracted from the
    PDF with column-aware block ordering.
    """
    pdf_path = ROOT / "data" / "raw" / doc["pdf"]
    if not pdf_path.exists() or doc["document_id"] != "kdigo_2024_ckd":
        return []

    have = {chunk["label"] for chunk in chunks if chunk["label"]}
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - optional at chunk time
        return []

    recovered: list[dict] = []
    spliced: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            text = _best_page_reading(page)
            for match in re.finditer(
                r"\b(Recommendation|Practice Point)\s+(\d+(?:\.\d+)+)\s*:\s*(.{40,700}?)(?=\s*(?:Recommendation|Practice Point)\s+\d|\Z)",
                text,
            ):
                label = f"{match.group(1).title()} {match.group(2)}"
                if label in have:
                    continue
                have.add(label)
                kind = "recommendation_atom" if match.group(1).lower() == "recommendation" else "practice_point"
                # Same journal-footer strip as the main path: this recovery reads
                # raw PDF blocks, so it picks up the page footer just as readily.
                body = JOURNAL_FOOTER_RE.sub("", f"{label}: {match.group(3).strip()}").strip()
                if kind == "recommendation_atom" and len(EVIDENCE_GRADE_RE.findall(body)) != 1:
                    spliced.append(label)
                    continue
                provenance = [{"page": page_index + 1, "bbox": None, "coord_origin": "BOTTOMLEFT"}]
                chunk = make_chunk(doc, kind, body, ["Recovered from PDF"], provenance, label=label, seq=0)
                chunk["quality_flags"] = [f for f in chunk["quality_flags"] if f != "too_short"]
                chunk["indexable"] = not chunk["quality_flags"]
                chunk["recovered"] = True
                recovered.append(chunk)
    if spliced:
        print(f"  backfill: refused {len(spliced)} spliced recovery/recoveries: {', '.join(spliced)}")
    return recovered


# ------------------------------------------------------------------- retrieval

def to_retrieval_record(chunk: dict) -> dict:
    """Shape expected by build_indexes.py and retrieve.py."""
    header = (
        f"{chunk['publisher']} | {' > '.join(chunk['section_path'][-2:])}"
        f"{' | ' + chunk['label'] if chunk['label'] else ''}"
    )
    return {
        "id": chunk["id"],
        "text": f"{header}\n{chunk['raw_text']}",
        "raw_text": chunk["raw_text"],
        "hype_questions": chunk["hype_questions"],
        "metadata": {
            "document_id": chunk["document_id"],
            "document": chunk["document"],
            "publisher": chunk["publisher"],
            "source_url": chunk["source_url"],
            "chunk_type": chunk["chunk_type"],
            "label": chunk["label"],
            # Carried into retrieval so the answer layer can supply a figure whose
            # content a recommendation only points at.
            "references": chunk.get("references") or [],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "provenance": chunk["provenance"],
            "section_path": chunk["section_path"],
            # Read from the guideline's own table caption. The answer layer uses it to
            # resolve "eGFR 38, which category?" by lookup over the chunks that define
            # the categories, rather than hoping similarity ranking surfaces one - see
            # _band_answer in nephrolex/generation/answer.py.
            "defines_categories": defines_categories(chunk["section_path"]),
            "topics": chunk["topics"],
            "exact_terms": chunk["exact_terms"],
            "citation": chunk["citation"],
            "parent_id": chunk["parent_id"],
            "citable": chunk["citable"],
            "indexable": chunk["indexable"],
            "quality_flags": chunk["quality_flags"],
            "scope_flags": chunk["scope_flags"],
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Chunk CKD guidelines from Docling JSON.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    per_doc: dict[str, int] = {}
    for doc in DOCUMENTS:
        chunks = parse_document(doc)
        per_doc[doc["document_id"]] = len(chunks)
        all_chunks.extend(chunks)

    indexable = [chunk for chunk in all_chunks if chunk["indexable"]]

    with (out_dir / "all_chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with (out_dir / "retrieval_corpus.jsonl").open("w", encoding="utf-8") as f:
        for chunk in indexable:
            f.write(json.dumps(to_retrieval_record(chunk), ensure_ascii=False) + "\n")

    summary = {
        "source": "docling_json",
        "total_chunks": len(all_chunks),
        "indexable_chunks": len(indexable),
        "per_document": per_doc,
        "chunk_types": dict(Counter(c["chunk_type"] for c in all_chunks)),
        "indexable_chunk_types": dict(Counter(c["chunk_type"] for c in indexable)),
        "topics": dict(Counter(t for c in indexable for t in c["topics"]).most_common()),
        "scope_flagged": sum(1 for c in indexable if c["scope_flags"]),
        "with_bbox": sum(1 for c in indexable if c["provenance"] and c["provenance"][0].get("bbox")),
        "mean_chars": round(sum(len(c["raw_text"]) for c in indexable) / max(len(indexable), 1)),
    }
    (out_dir / "chunking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "docling_chunking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
