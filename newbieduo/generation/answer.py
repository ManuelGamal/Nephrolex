"""Grounded answer composition for CKD guideline questions.

Every sentence presented as guidance is quoted verbatim from a retrieved chunk and
carries a citation resolving to a guideline, a recommendation number and a page.
Nothing here paraphrases or synthesises clinical content: the pipeline decides
*which* evidence to show and *how it relates*, never what it says. That is the
whole safety argument, and it is why this layer is extractive rather than
generative - a fluent paraphrase of a threshold is indistinguishable from a
hallucinated one at a glance.

Order of operations matters:

    scope gate -> retrieve -> confidence gate -> claim comparison -> render

Scope is checked before retrieval so an out-of-scope question never reaches the
evidence layer at all. The confidence gate then catches in-scope questions the
corpus simply cannot answer, which is a different failure and deserves a different
message.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from newbieduo.generation.claims import analyse
from newbieduo.retrieval.retrieve import numeric_range_signal, retrieve
from newbieduo.safety.scope import classify, in_domain, other_specialty, refusal_text


from newbieduo.paths import ROOT
# Fused scores are min-max normalised within the candidate pool, so the top result
# always sits near 1.0 regardless of whether anything relevant exists. That makes a
# single global threshold useless for abstention: measured on the gold set, raw BM25
# for answerable questions spans 11.7-61.0 and for unanswerable ones 14.8-29.4 -
# fully overlapping. Two conditional gates are used instead.
#
# In-domain questions get a low bar, because the domain check already passed.
MIN_CONFIDENCE = 0.62

# Chunk types that may be shown as evidence. Narrative passages are included but
# labelled, not excluded: leaving them out meant a question whose answer lives in
# prose - "hazard ratios for adverse outcomes", or NICE's definition of CKD - had
# its best-ranked evidence silently dropped and unrelated recommendations shown in
# its place. What matters is that prose is not presented *as* a recommendation, and
# that is a labelling problem, not a filtering one.
CITABLE_TYPES = {"recommendation_atom", "practice_point", "table_row", "section_passage",
                 "figure_caption", "figure_body"}

# How each type is described to the reader.
EVIDENCE_KIND = {
    "recommendation_atom": "recommendation",
    "practice_point": "practice point",
    "table_row": "table",
    "section_passage": "narrative",
    "figure_caption": "figure",
    "figure_body": "figure",
}


VALUE_WITH_UNIT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg\s?/\s?mmol|mg\s?/\s?g|ml\s?/\s?min|mmol\s?/\s?l|mm\s?Hg|g\s?/\s?kg)\b",
    re.I,
)


_CORPUS_VOCABULARY: set[str] | None = None


def _corpus_vocabulary() -> set[str]:
    global _CORPUS_VOCABULARY
    if _CORPUS_VOCABULARY is None:
        from newbieduo.retrieval.retrieve import content_terms, load_indexes

        _CORPUS_VOCABULARY = set()
        for record in load_indexes().records:
            _CORPUS_VOCABULARY |= content_terms(record["raw_text"])
    return _CORPUS_VOCABULARY


def _unknown_concepts(query: str, min_length: int = 5) -> list[str]:
    """Query concepts that appear nowhere in either guideline.

    This replaces a score threshold, which cannot work here. Measured with
    MedEmbed + reranker, "what is the recommended treatment for acute appendicitis"
    scores 0.825 at rank 1 while two genuinely answerable questions score 0.757 and
    0.789 - the cross-encoder confidently ranks the best of a bad list, so no cutoff
    separates them. Vocabulary does: "appendicitis" occurs in zero chunks, whereas
    every content word of the answerable paraphrases occurs somewhere.
    """
    from newbieduo.retrieval.retrieve import content_terms

    vocabulary = _corpus_vocabulary()
    return [
        term
        for term in sorted(content_terms(query))
        if len(term) >= min_length and not term.isdigit() and term not in vocabulary
    ]


def _has_askable_content(query: str) -> bool:
    """Is there enough here to be a question at all?

    "???" and "38" both produced confident guideline answers before this check.
    Retrieval always returns its best candidates, so an input carrying no topical
    content still yields a top-ranked chunk and an answer built around it - which
    is exactly the wrong behaviour in front of someone who mistyped.
    """
    from newbieduo.retrieval.retrieve import content_terms

    terms = content_terms(query)
    return bool(terms) and any(not term.isdigit() for term in terms)


def _is_quotable(result: dict) -> bool:
    meta = result["metadata"]
    return meta.get("citable", True) and meta.get("chunk_type") in CITABLE_TYPES


def _distinct(results: list[dict]) -> list[dict]:
    """Drop restatements of a recommendation already shown.

    KDIGO prints every recommendation twice - once in its summary section and again
    in the chapter body - so retrieval legitimately returns both, and the answer
    would quote Recommendation 3.4.1 from p.97 and p.43 as if they were two pieces
    of evidence. The earliest page is kept, being the summary a reader would cite.
    """
    seen_labels: set[str] = set()
    seen_text: set[str] = set()
    out: list[dict] = []
    for result in results:
        label = (result["metadata"].get("label") or "").strip()
        fingerprint = re.sub(r"[^a-z0-9]", "", result["raw_text"].lower())[:120]
        if (label and label in seen_labels) or fingerprint in seen_text:
            continue
        if label:
            seen_labels.add(label)
        seen_text.add(fingerprint)
        out.append(result)
    return out


def _collapse_duplicates(items: list[dict]) -> list[dict]:
    """Fold identical evidence into the copy that outranks it, keeping its location.

    Presentation only - this runs on the panel that shows what retrieval returned,
    never on scoring. 23% of the corpus is an exact duplicate of another chunk
    because KDIGO prints every recommendation twice, and a duplicate appears in the
    top 10 for 63% of queries, so a reader otherwise sees the same recommendation
    listed twice and reasonably concludes the system is padding.

    Deduplicating *retrieval* was measured instead of assumed and is not done: it
    costs 0.0606 nDCG@10, and on the cases where the gold set does not pin both
    copies it gains 0.0061, well inside the noise floor. There is no evidence it
    helps ranking, so only the display changes.
    """
    by_text: dict[str, dict] = {}
    out: list[dict] = []
    for item in items:
        fingerprint = re.sub(r"[^a-z0-9]", "", (item.get("snippet") or item.get("text") or "").lower())[:120]
        if not fingerprint:
            out.append(item)
            continue
        first = by_text.get(fingerprint)
        if first is None:
            by_text[fingerprint] = item
            out.append(item)
            continue
        # Same words elsewhere in the document: record where, do not list it again.
        page = item.get("page_start")
        if page is not None:
            first.setdefault("also_at", [])
            if page not in first["also_at"]:
                first["also_at"].append(page)
    return out


# How many pieces of evidence are quoted to the reader AND handed to the generator.
#
# These were 3 and 6. The generator cited by index into its own list, so an answer
# could say "[5]" while the interface displayed three quotes - 38 markers across the
# stored runs pointed at evidence the reader could not see. For a system whose claim
# is that every statement is traceable, an unresolvable citation is the worst kind of
# defect, and it is only possible when the two lists differ. One constant now feeds
# both, so they cannot drift apart again.
#
# It also fixes a quieter failure: the end-to-end audit found 9 questions where the
# correct chunk was retrieved and ranked but fell just outside the three shown - four
# of them at rank 4.
#
# Raised 6 -> 8 for a case that shows why the number matters. Asked what blood pressure
# target applies in CKD, the model correctly gave both NICE arms - below 140 mmHg under
# ACR 70, below 130 mmHg at or above it - and was rejected on three counts: "130 mmHg",
# "80 mmHg" and the range "120 to 129" were "not in the cited evidence". They were not.
# NICE 1.6.2 states all three and sat at rank 8, outside the window; the model had read
# them from the neighbouring text, which the prompt marks "for understanding only, do
# not cite" and which verify() deliberately does not count as support.
#
# So the model was clinically right and evidentially wrong, and the check caught exactly
# that. The fix is not to relax the check - it is to let the chunk that states the
# numbers be citable. Only generation and the evidence rail move; retrieval is untouched.
SHOWN_EVIDENCE = 8

# How much evidence the premise gate is allowed to read.
#
# It pooled the raw text of the top 20 retrieved chunks and asked whether the queried
# value appeared anywhere in them. Support could therefore come from a chunk in an
# entirely different clinical context. Asked at what eGFR of 5 ml/min an SGLT2
# inhibitor should be started, the premise was accepted because KDIGO Practice Point
# 5.4.5 states "eGFR 5-15 ml/min per 1.73 m2" - the window for pre-emptive
# transplantation in children, which has nothing to do with SGLT2 initiation. That
# chunk was ranked 8th, so restricting the gate to the citable window changed nothing.
#
# A premise is a stronger claim than a citation: it asserts the value is a real
# threshold for *this* question, not merely that the corpus mentions it somewhere. So
# the gate reads a shallower window than the answer may cite - top-ranked evidence
# only, where a contextually unrelated chunk is unlikely to reach.
#
# Swept over the 75 gold cases plus three probe questions built on false premises:
#
#     depth    gate fires   new catches   false declines
#     3        5            2             1   <- stag_g2_direct
#     4        5            2             1   <- stag_g2_direct
#     5        4            2             0   <- kept
#     6        3            1             0
#     8        2            0             0
#     pooled   2            0             0       the previous behaviour
#
# The honest caveat: the boundary is one case wide on each side. Choosing 5 over 4
# rests on a single false decline, and 5 over 6 on a single probe - and all three
# probes were written by this system's author, the same circularity the gold set
# carries. What is *not* thin is the protective number: zero false declines across all
# 75 gold cases, and the failure mode of tightening is a refusal that lists the closest
# material, never a wrong answer.
#
# Set to None to restore the pooled top-20 behaviour exactly.
PREMISE_EVIDENCE_DEPTH = 5

# How much neighbouring text to show the model per quoted chunk.
NEIGHBOUR_CONTEXT_CHARS = 700


def _neighbour_context(result: dict, budget: int = NEIGHBOUR_CONTEXT_CHARS) -> str:
    """Text immediately around a chunk, for the model to read but not to cite.

    Chunks are evidence atoms - a numbered recommendation, one table row - which is
    what makes citations point at exact lines and bounding boxes highlight exact
    lines. It also means a chunk can be too small to interpret: "Refer adults with
    CKD ... in the circumstances listed in Figure 48" is a whole practice point.

    Enlarging the chunks themselves would fix that and break three other things: the
    bounding box becomes a page-sized rectangle, long passages accumulate query terms
    and outrank the recommendation that answers the question, and - most seriously -
    the answer verifier checks each number against the evidence it cites, so wider
    evidence means more invented numbers slip through.

    So the atom stays the unit of retrieval, citation and verification, and this adds
    surroundings for comprehension only. build_prompt marks it unquotable, and
    verify() never reads it: a number that appears only here is still rejected.
    """
    from newbieduo.retrieval.retrieve import load_indexes

    records = load_indexes().records
    global _RECORD_POSITION
    if _RECORD_POSITION is None:
        _RECORD_POSITION = {record["id"]: index for index, record in enumerate(records)}

    position = _RECORD_POSITION.get(result["id"])
    if position is None:
        return ""

    section = result["metadata"].get("section_path")
    pieces: list[str] = []
    for offset in (-1, 1):
        neighbour = position + offset
        if not 0 <= neighbour < len(records):
            continue
        record = records[neighbour]
        # Only the same section: the chunk before a section boundary is a different
        # subject, and presenting it as context invites the model to blend the two.
        if record["metadata"].get("section_path") != section:
            continue
        pieces.append(" ".join(record["raw_text"].split()))

    return " ... ".join(pieces)[:budget]


_RECORD_POSITION: dict[str, int] | None = None


def _signal_weights() -> dict[str, float]:
    """Active weight per additive signal, read from retrieve()'s own defaults.

    Derived rather than restated: the retriever's signature is the single source of
    truth for what each signal is worth, and a second copy here would drift the first
    time a weight changed - which is the failure this project has hit three times.
    """
    import inspect

    from newbieduo.retrieval.retrieve import SIGNALS, retrieve as retrieve_fn

    defaults = inspect.signature(retrieve_fn).parameters
    weights: dict[str, float] = {}
    for signal in SIGNALS:
        if signal.additive and signal.weight_param in defaults:
            weights[signal.name] = defaults[signal.weight_param].default
    return weights


def _resolve_references(quoted: list[dict], limit: int = 2) -> list[dict]:
    """Add the figures and tables the quoted evidence defers to.

    Some guidance carries its content by reference. Practice Point 5.1.1 is, in full,
    "Refer adults with CKD to specialist kidney care services in the circumstances
    listed in Figure 48" - it ranks first for referral questions and answers nothing
    on its own, because every criterion lives in the figure.

    The referenced chunk is appended as evidence in its own right rather than spliced
    into the quoting chunk. Quotes are presented verbatim and checked against the PDF
    by bounding box, so merging text would break both the quote and its citation; a
    separate entry keeps each piece attributable to where it actually came from.
    """
    if not quoted:
        return quoted

    from newbieduo.retrieval.retrieve import load_indexes

    wanted: list[str] = []
    for result in quoted:
        for reference in result["metadata"].get("references") or []:
            if reference not in wanted:
                wanted.append(reference)
    if not wanted:
        return quoted

    have = {(r["metadata"].get("label") or "").strip() for r in quoted}
    documents = {r["metadata"].get("document_id") for r in quoted}
    by_label: dict[str, dict] = {}
    for record in load_indexes().records:
        meta = record["metadata"]
        label = (meta.get("label") or "").strip()
        # Prefer a rebuilt figure body over a bare caption: the caption names the
        # figure, the body carries the criteria.
        if not label or meta.get("document_id") not in documents:
            continue
        if label in by_label and by_label[label]["metadata"].get("chunk_type") == "figure_body":
            continue
        if meta.get("chunk_type") in {"figure_body", "figure_caption", "table_row"}:
            by_label[label] = record

    # Placed directly after the chunk that defers to it, not appended. Only the first
    # few quotes are shown and passed to the generator, so a figure parked at the end
    # of the list is resolved in name only - the reader still sees the pointer alone.
    resolved: list[dict] = []
    added = 0
    for result in quoted:
        resolved.append(result)
        if added >= limit:
            continue
        for reference in result["metadata"].get("references") or []:
            if reference in have:
                continue
            record = by_label.get(reference)
            if record is None or record["metadata"].get("chunk_type") != "figure_body":
                continue
            entry = dict(record)
            entry["score"] = result["score"]
            entry["resolved_from"] = reference
            resolved.append(entry)
            have.add(reference)
            added += 1
            break
    return resolved


def _unsupported_values(query: str, evidence: list[dict]) -> list[str]:
    """Catch questions built on a threshold that does not exist in the guidelines.

    "What is the ACR threshold of 500 mg/mmol used for?" presupposes a value neither
    guideline states. Answering it produces a confident-sounding fabrication, because
    retrieval will happily return the nearest ACR material. If the query names a
    value with a unit and no retrieved chunk states that value or a band containing
    it, the premise is reported as unverified rather than answered around.
    """
    stated = VALUE_WITH_UNIT.findall(query)
    if not stated:
        return []

    corpus_text = " ".join(r["raw_text"] for r in evidence)
    unsupported: list[str] = []

    for value, unit in stated:
        number = float(value)
        unit_key = _unit_key(unit)
        unit_re = _unit_regex(unit)

        # Values stated against the same unit, e.g. "70 mg/mmol".
        singles = {
            float(match) for match in re.findall(rf"(\d+(?:\.\d+)?)\s*{unit_re}", corpus_text, re.I)
        }
        # Closed bands against the same unit, e.g. "3 to 30 mg/mmol".
        bands = [
            (float(low), float(high))
            for low, high in re.findall(
                rf"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*{unit_re}", corpus_text, re.I
            )
        ]

        # Open-ended bounds are deliberately not accepted as support. ">= 90 ml/min"
        # spans to infinity and would vouch for any large number - which is exactly
        # how a fabricated "500 mg/mmol" threshold slipped through the first version.
        if number in singles or any(low <= number <= high for low, high in bands):
            continue
        unsupported.append(f"{value} {unit_key}")
    return unsupported


def _unit_key(unit: str) -> str:
    return re.sub(r"\s+", "", unit.lower())


def _unit_regex(unit: str) -> str:
    """Match a unit tolerantly: PDFs print "mg/mmol", "mg / mmol" and "mm Hg"."""
    return r"\s?".join(re.escape(char) for char in _unit_key(unit))


def build_answer(
    query: str,
    top_k: int = 10,
    dense_model: str | None = None,
    reranker_model: str | None = None,
    rerank_depth: int = 30,
    generate: bool = False,
) -> dict:
    if not _has_askable_content(query):
        return {
            "query": query,
            "status": "no_question",
            "scope": {"in_scope": False, "category": "no_question", "reason": "", "matched": [], "guidance": ""},
            "text": (
                "That doesn't look like a question yet. Ask about CKD assessment or "
                "management - for example, when to refer, what to monitor, or a "
                "blood pressure or ACR threshold."
            ),
            "evidence": [],
            "comparisons": [],
        }

    verdict = classify(query)
    if not verdict.in_scope:
        return {
            "query": query,
            "status": "declined",
            "scope": verdict.as_dict(),
            "text": refusal_text(verdict),
            "evidence": [],
            "comparisons": [],
        }

    # explain=True asks the retriever for the per-signal breakdown behind each score.
    # It is on by default because the cost is a dict per candidate and the benefit is
    # that every ranking decision this system makes can be inspected rather than
    # taken on trust - which is the whole argument for a scoring model with named,
    # weighted signals instead of an opaque one.
    results = retrieve(query, top_k, dense_model, reranker_model,
                       rerank_depth=rerank_depth, explain=True)
    if not results:
        return {
            "query": query,
            "status": "no_evidence",
            "scope": verdict.as_dict(),
            "text": "No evidence was retrieved from the guideline corpus for this question.",
            "evidence": [],
            "comparisons": [],
        }

    top_score = results[0]["score"]
    quotable = _resolve_references(_distinct([r for r in results if _is_quotable(r)]))
    domain = in_domain(query)

    off_specialty = other_specialty(query)
    if off_specialty or (not domain and _unknown_concepts(query)):
        return {
            "query": query,
            "status": "out_of_domain",
            "scope": verdict.as_dict(),
            "confidence": round(top_score, 3),
            "text": (
                "I can't answer this one. It doesn't appear to be a question about kidney "
                + (f"disease - it asks about {off_specialty}, which neither guideline covers."
                   if off_specialty else
                   "disease, and nothing in KDIGO 2024 or NICE NG203 addresses it.")
                + "\n\nThis system only covers chronic kidney disease in adults."
            ),
            "evidence": [],
            "comparisons": [],
        }

    premise_evidence = (results[:20] if PREMISE_EVIDENCE_DEPTH is None
                        else quotable[:PREMISE_EVIDENCE_DEPTH])
    unsupported = _unsupported_values(query, premise_evidence)
    if unsupported:
        return {
            "query": query,
            "status": "premise_not_found",
            "scope": verdict.as_dict(),
            "confidence": round(top_score, 3),
            "unsupported_values": unsupported,
            "text": (
                "I can't confirm the premise of this question. "
                + "; ".join(unsupported)
                + " does not appear as a threshold in either guideline, so answering as "
                "though it did would invent evidence. The closest retrieved material is "
                "listed below."
            ),
            "evidence": [_evidence_item(r) for r in results[:4]],
            "comparisons": [],
        }

    if top_score < MIN_CONFIDENCE or not quotable:
        return {
            "query": query,
            "status": "insufficient_evidence",
            "scope": verdict.as_dict(),
            "confidence": round(top_score, 3),
            "text": (
                "I can't answer this from KDIGO 2024 or NICE NG203 with enough confidence "
                "to be useful. The closest material retrieved is listed below, but none of "
                "it directly addresses the question - treat it as a starting point, not an "
                "answer."
            ),
            "evidence": [_evidence_item(r) for r in results[:4]],
            "comparisons": [],
        }

    analysis = analyse(results)
    documents = {r["metadata"]["document_id"] for r in quotable}

    # Generation is additive and never replaces the extractive answer. If the
    # model's output fails verification it is discarded, so the response shown can
    # only ever contain guideline text that was actually retrieved.
    generated = None
    if generate:
        from newbieduo.generation.generate import generate as generate_answer

        generated = generate_answer(
            query,
            [
                {
                    "citation": r["metadata"].get("citation"),
                    "text": r["raw_text"],
                    # Where the chunk sits in the guideline. A quarter of the corpus is
                    # under 200 characters, and some of that is a pointer - "Refer
                    # adults with CKD ... in the circumstances listed in Figure 48" is
                    # the whole of Practice Point 5.1.1. Handed that with no context,
                    # the model can only repeat the pointer. The section path says what
                    # the chunk is about even when the chunk itself does not.
                    "section": " > ".join(r["metadata"].get("section_path") or []),
                    # Surroundings for comprehension only - never counted as support.
                    "context": _neighbour_context(r),
                }
                for r in quotable[:SHOWN_EVIDENCE]
            ],
        )

    return {
        "generated": generated,
        "query": query,
        "status": "answered",
        "scope": verdict.as_dict(),
        "confidence": round(top_score, 3),
        "guidelines_used": sorted(documents),
        "quotes": [_evidence_item(r, include_text=True) for r in quotable[:SHOWN_EVIDENCE]],
        # Retrieval's own results, plus anything pulled in to resolve a reference. A
        # figure that is quoted but absent from the "chunks considered" list reads as
        # a contradiction, so it is listed here too and marked as resolved rather than
        # retrieved - the panel is meant to show exactly what fed the answer.
        "evidence": _collapse_duplicates(
            [_evidence_item(r) for r in results[:top_k]]
            + [_evidence_item(r) for r in quotable if r.get("resolved_from")]
        ),
        "retrieval": {
            # The active weight of each additive signal, so the interface can show what
            # a signal *contributed* rather than what it scored. BM25 normalises to
            # ~0.96 on a good match and is multiplied by 0.0, so displaying its raw
            # value next to dense's made a signal that does not affect ranking look
            # like a reason for it.
            "weights": _signal_weights(),
            "candidates_shown": len(results),
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "rerank_depth": rerank_depth if reranker_model else None,
        },
        "comparisons": analysis["comparisons"],
        "divergences": analysis["divergences"],
        "divergence_summary": analysis["summary"],
        "text": render_text(query, quotable, analysis, documents),
    }


def _evidence_item(result: dict, include_text: bool = False) -> dict:
    meta = result["metadata"]
    item = {
        "chunk_id": result["id"],
        "citation": meta.get("citation"),
        "evidence_kind": EVIDENCE_KIND.get(meta.get("chunk_type"), "evidence"),
        # Set when this item was not retrieved on its own merits but pulled in
        # because quoted evidence deferred to it ("...listed in Figure 48").
        "resolved_from": result.get("resolved_from"),
        # Per-signal decomposition of the score, straight from retrieve.SIGNALS. The
        # same registry produces the score, the ablation rows and this breakdown, so
        # what the interface shows a judge is by construction what the ranker did.
        "components": result.get("components"),
        "document_id": meta.get("document_id"),
        "publisher": meta.get("publisher"),
        "label": meta.get("label"),
        "chunk_type": meta.get("chunk_type"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "provenance": meta.get("provenance"),
        "score": round(result["score"], 4),
    }
    if include_text:
        item["text"] = result["raw_text"]
    else:
        # Enough to recognise the chunk in the ranked list without shipping the
        # whole corpus to the browser on every query.
        item["snippet"] = " ".join(result["raw_text"].split())[:240]
    return item


def render_text(query: str, quotable: list[dict], analysis: dict, documents: set[str]) -> str:
    lines = [f"Question: {query}", "", "Guideline evidence:"]

    for result in quotable[:SHOWN_EVIDENCE]:
        meta = result["metadata"]
        lines.append("")
        lines.append(f'  "{ " ".join(result["raw_text"].split()) }"')
        lines.append(f"    -- {meta.get('citation')}")

    lines.append("")
    if analysis["divergences"]:
        lines.append("Guideline agreement: THE SOURCES DIFFER")
        for summary in analysis["summary"]:
            lines.append(f"  - {summary}")
        lines.append("  Both positions are cited above; the choice is a clinical judgement.")
    elif len(documents) > 1:
        lines.append("Guideline agreement: KDIGO 2024 and NICE NG203 are consistent here.")
    else:
        only = "KDIGO 2024" if "kdigo_2024_ckd" in documents else "NICE NG203"
        lines.append(f"Guideline agreement: only {only} was found to address this directly.")

    lines.append("")
    lines.append(
        "Scope: adult, non-pregnant CKD. Not a substitute for clinical judgement; "
        "paediatric, pregnancy and kidney-replacement decisions are out of scope."
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Answer a CKD question from the guideline corpus.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--json", action="store_true", help="Emit the full structured record.")
    args = parser.parse_args()

    result = build_answer(args.query, args.top_k, args.dense_model, args.reranker_model)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["text"])
        print()
        print(f"[status={result['status']}", end="")
        if "confidence" in result:
            print(f" confidence={result['confidence']}", end="")
        print("]")


if __name__ == "__main__":
    main()
