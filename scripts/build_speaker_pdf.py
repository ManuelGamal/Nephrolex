"""Render the speaker script as a plain PDF: one typeface, black on white, no design.

Deliberately unstyled. It is read off a phone or a lectern while talking, so the only
things that matter are legible body type, obvious slide boundaries, and cues that are
distinguishable from the words to be said out loud.

Usage:
    python scripts/build_speaker_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "Nephrolex_speaker_script.pdf"

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=14.5,
                      spaceAfter=6)
SLIDE = ParagraphStyle("slide", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                       spaceBefore=14, spaceAfter=2)
TITLE = ParagraphStyle("title", fontName="Helvetica-Oblique", fontSize=10.5, leading=14,
                       spaceAfter=6)
CUE = ParagraphStyle("cue", fontName="Helvetica-Oblique", fontSize=10, leading=13.5,
                     leftIndent=14, spaceAfter=6)
SEC = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                     spaceBefore=20, spaceAfter=8)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=20,
                    spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=16)

# (slide label, time, title, [blocks]) - a block is ("say"|"cue", text)
SCRIPT = [
    ("SLIDE 01", "0:00 - 0:20", "Guideline evidence, delivered where the decision is made.", [
        ("say", "This is Nephrolex - clinical decision support for chronic kidney disease, over KDIGO 2024 and NICE NG203."),
        ("say", "Three things make it different from a chatbot with a PDF. It answers <b>only</b> from the guidelines. It cites every number to a page and a bounding box. And when the evidence will not support an answer, it <b>refuses</b>."),
        ("say", "Those four numbers on screen are what I will spend the next nine minutes defending."),
        ("cue", "[Do not read the metrics aloud. Let them sit there while you talk.]"),
    ]),
    ("SLIDE 02", "0:20 - 0:50", "The guideline is 250 pages. The consultation is ten minutes.", [
        ("say", "The problem is not that the answer is missing. It is in the guideline."),
        ("say", "The problem is that a consultation is ten minutes. Ask which band an eGFR of 38 falls into, and the answer is one row, of one table, on page 22 - while the threshold that actually changes management sits in a different table, on a different page, in the <i>other</i> guideline."),
        ("say", "A general assistant answers that instantly, with no citation and no page. A clinician cannot act on that, and cannot defend having acted on it."),
        ("say", "So the bar here is not fluency. It is <b>traceability</b>."),
    ]),
    ("SLIDE 03", "0:50 - 1:15", "CDS Hooks: the evidence arrives without being asked for.", [
        ("say", "Which raises the question nobody asks until it is too late: a clinician is not going to leave their record software to visit a website."),
        ("say", "So this is exposed as an HL7 <b>CDS Hooks</b> service. When a chart opens, the record calls us with the patient's eGFR and ACR as FHIR Observations, matched on LOINC codes. Five plain rules decide whether anything is worth saying - and you can read them, they are predicates, not a model."),
        ("say", "Each question then runs the ordinary retrieval path. Same gates, same reranker, same verifier."),
        ("say", "And the part that matters: if retrieval declines, or produces nothing citable, <b>no card is emitted at all</b>."),
    ]),
    ("SLIDE 04", "1:15 - 2:15", "One patient, start to finish.  [LIVE DEMO]", [
        ("cue", "[CLICK: Patient A - eGFR 24, ACR 85. Let the sequence play. Do not talk over the rules appearing.]"),
        ("say", "Let me show you rather than describe it."),
        ("say", "Chart opens. eGFR 24, ACR 85. The hook fires, and five rules evaluate against those two numbers - those are the real predicates from the service, running in this page. All five fire, and the service emits at most three cards."),
        ("say", "Here is what lands in the corner of the chart. <b>Verbatim guideline text</b>, never generated prose - because this is pushed at a clinician who did not ask for it, so it is held to the stricter standard. Each card names the rule that raised it and the page it resolves to."),
        ("cue", "[CLICK: Patient B - eGFR 96, ACR 1.2. This is the strongest moment in the talk. Pause after 'zero cards'.]"),
        ("say", "Now the case that matters more. Healthy kidneys. All five rules go silent. <b>Zero cards.</b> The clinician's screen is untouched."),
        ("say", "That is the outcome this service produces most often - and it is the whole reason it is worth leaving switched on."),
    ]),
    ("SLIDE 05", "2:15 - 2:45", "Nobody sells this to a doctor.", [
        ("say", "You have just watched it work. So - who pays for it?"),
        ("say", "Not the clinician. Clinicians do not buy reference tools - their trust does, centrally, out of the budget line that already funds guideline access. So the unit is an annual <b>site licence</b> banded by headcount, and it is a substitution rather than a new category to create."),
        ("say", "The channel is the record vendor's own marketplace. Epic and Oracle Health both distribute exactly this - CDS Hooks services a trust switches on - and we are <b>already conformant</b>, so the channel does the integration selling."),
        ("say", "One more thing that decides whether this is sellable at all: software that hands a clinician a recommendation is, in most jurisdictions, a <b>medical device</b>. Software that shows them the source and lets them judge it generally is not. Verbatim text, a citation, and a visible refusal are what make that possible - and we built them for grounding, before anyone asked about procurement. A design posture, not a certification."),
        ("say", "And the margins work because the five trigger questions are fixed. First hook after startup is <b>four milliseconds</b>; every patient after that is fifteen milliseconds from cache. Marginal cost per patient is effectively zero, so a site licence is close to pure gross margin."),
    ]),
    ("SLIDE 06", "2:45 - 3:10", "Parsing and chunking: what you index is what you can cite.", [
        ("say", "Guidelines are not prose. They are numbered recommendations, practice points, and cross-tabulated tables - so parsing is layout-aware and chunks are <i>typed</i>. 1,485 indexable chunks across six types, one table row per chunk."),
        ("say", "Three defects on the right, and I would rather name them than have them found. KDIGO's PDF maps 'greater-than-or-equal' to a dollar sign, so 69 thresholds were stored as nonsense. The parser was dropping the very cell that names each table row - 245 of them. And a spanning header got repeated once per column it covered, which is keyword stuffing the guideline never wrote."),
        ("say", "All three were found by measurement, not by reading."),
    ]),
    ("SLIDE 07-08", "3:10 - 3:50", "Every chunk knows exactly where it came from.", [
        ("say", "If you only remember one slide, make it this one."),
        ("say", "These are rendered from the live corpus - the same coordinates the answer UI draws when a clinician opens a source. That is KDIGO Table 2, with the <b>G3b row boxed</b>, and the whole table outlined faintly around it, so a single row is never shown stripped of the header that gives it meaning."),
        ("cue", "[NEXT: second provenance slide.]"),
        ("say", "A NICE recommendation kept whole, so a quote can never be half a clinical instruction. A practice point carrying two boxes."),
        ("say", "And one detail that is load-bearing: Docling emits element boxes bottom-left and table-cell boxes top-left. Flip the wrong one and you get a box that bounds nothing, sitting next to the text it was supposed to prove."),
        ("say", "A citation here is not a page reference. It is a rectangle on a page."),
    ]),
    ("SLIDE 09", "3:50 - 4:10", "The gold set - what every number that follows is measured against.", [
        ("say", "Before I quote a single number, this is what they are measured against."),
        ("say", "Three answer keys, built by different methods - because one key written alongside the system it grades will flatter it. <b>141 questions, 282 chunk-level judgements</b>, graded 2 or 1 rather than yes or no, which is what makes nDCG mean anything."),
        ("say", "Forty-eight dev, twenty-seven test, and the test slice is never tuned on. I will come back at the end to how those judgements are pinned - and to the evaluator that could not fail."),
    ]),
    ("SLIDE 10", "4:10 - 4:35", "Why MedEmbed - and what the comparison actually proved.", [
        ("say", "Seven encoders, same gold set, same everything else. MedEmbed wins - but its real margin is <b>recall at 20</b>: 0.78 against 0.75. It puts more evidence inside the window the reranker can still rescue."),
        ("say", "The more useful finding is on the right. Measured dense-only, two encoders differ by <b>0.267 nDCG</b>. With the cross-encoder re-ordering the top 30, that difference collapses to about <b>0.02</b>."),
        ("say", "The models are not alike. This pipeline's reranker <i>makes</i> them alike. That is a claim about the architecture, not about embeddings - and it is why the effort went into the document side instead of shopping for a better encoder."),
    ]),
    ("SLIDE 11", "4:35 - 5:05", "Four signals, each earning its place.", [
        ("say", "Four signals. None is here because the literature likes it - each is here because switching it off costs something measurable, on a case we can name."),
        ("say", "<b>TF-IDF</b> for exact clinical tokens. The token pattern is the whole point: it keeps ml/min, less-than-15, and recommendation numbers intact instead of shattering them. And the IDF term does the discriminating - in a corpus entirely about CKD, 'patient' and 'kidney' are as uninformative as 'the'."),
        ("say", "<b>Dense</b> for vocabulary that never overlaps: 'protein in the urine' against 'albuminuria'. TF-IDF scores that near zero, and no amount of lexical weighting recovers a word that is not there."),
        ("say", "<b>Metadata</b> for structural grounding. And then doc2query, which needs its own slide."),
    ]),
    ("SLIDE 12", "5:05 - 5:40", "The phrasing gap, and the only thing that closed it.", [
        ("say", "This is the biggest single idea in the system."),
        ("say", "Retrieval swung from <b>0.90 to 0.38 on identical content</b>, depending only on whether the question used guideline vocabulary. A clinician asks about 'a flozin' - a word in neither guideline. About a 'vomiting bug', where KDIGO writes 'acute, dehydrating illness'."),
        ("say", "So an LLM writes, for each chunk, the questions a clinician would actually ask that it answers. 4,172 of them, generated once, cached, free at query time. It lifts the weak side by <b>0.16</b> with an interval that clears zero, and costs nothing on the strong side."),
        ("say", "Two design choices carry it. The expansions live in <b>their own index</b> - they can influence ranking and can never reach a citation, a quote, or the verifier. A hallucinated question costs ranking and nothing else. That is a safety property, not an optimisation."),
        ("say", "And we checked it is not circular rather than assuming: zero of 4,172 match the corpus's own template questions, and 68 percent of the words in a generated question do not appear in its source chunk."),
    ]),
    ("SLIDE 13", "5:40 - 6:05", "One convex combination, two priors, and a cross-encoder.", [
        ("say", "The scoring function. A convex combination of the four signals, normalised by the weight of whatever is switched on - so an ablation measures the signal, not a scale change. Then three deliberately mild multiplicative priors, which nudge ranking and are never allowed to override relevance."),
        ("say", "Then the cross-encoder re-orders the top 30 - and it reads the generated questions as well as the raw text, because otherwise it is blind to the very signal that surfaced the candidate."),
        ("say", "Every constant in that table was swept <i>twice</i>. All of them had originally been chosen against a pipeline that no longer existed. Re-swept on the current one, they came back at the shipped values - and they are defended as <b>indistinguishable from their neighbours</b>, not as an argmax, because with 67 questions anything under about 0.05 is not measurable."),
    ]),
    ("SLIDE 14", "6:05 - 6:35", "Five categorical gates. Three before retrieval runs at all.", [
        ("say", "The system is built so the failure mode is <i>refusing to answer</i>, never <i>answering wrongly with a citation</i>."),
        ("say", "Five categorical gates. <b>Three of them run before retrieval touches the corpus at all</b> - is it a question, is it a genre a guideline answers, is it in scope. Those refuse in under a millisecond. Domain and premise run after, because the premise check needs the evidence to look in."),
        ("say", "None of the five is a tuned threshold. All of them are readable rules a clinician could disagree with."),
        ("say", "Then the verifier, which is mechanical: every number-with-unit in a generated sentence must appear <b>verbatim in the evidence that sentence actually cited</b> - not the wider retrieved set, so a claim cannot borrow a value from a source it never pointed at. If it fails, the generated text is discarded and the guideline's own words are shown instead."),
        ("say", "And retrieval scores are <b>deliberately not used</b> for abstention. Three attempts failed, and the third told us why: 'how many people worldwide have CKD' scores 0.838 cosine and 1.00 rerank - <i>above</i> the median answerable question. The similarity is high and correct. What is wrong is the <i>kind</i> of information asked for, and no threshold on similarity can express that."),
    ]),
    ("SLIDE 15", "6:35 - 7:05", "Three questions about kidneys. One gets an answer.", [
        ("say", "Three questions. All fluent, all genuinely about CKD, and a general assistant answers all three."),
        ("say", "ACR threshold of <b>30</b> - answered, quoting the table that defines it."),
        ("say", "ACR threshold of <b>500</b> - one number changed. 500 appears in no band in either guideline, so the question asserts something untrue. The system refuses, rather than retrieving the nearest plausible table and answering around a false premise."),
        ("say", "And the epidemiology question - refused on genre, despite scoring higher than the median answerable one."),
        ("say", "<b>Zero unsafe answers. Zero false declines.</b> Across 75 audited cases, end to end."),
    ]),
    ("SLIDE 16", "7:05 - 7:30", "Fan-out at build time, fan-in at query time.", [
        ("say", "The whole system. Build time across the top, query time in the middle, and the answer flowing right to left along the bottom."),
        ("say", "The two dashed red paths are <b>the design, not error handling</b>. A question the guidelines do not answer leaves at the gates with a reason. A generated sentence whose number cannot be traced to its own citation is discarded whole, and the quotes stand in its place - so the worst case degrades to the guideline's own words, never to an unsupported claim."),
        ("say", "And notice what the doc2query index is <i>not</i> wired to. It feeds ranking and the reranker. It has no path to a quote, a citation, or the verifier."),
        ("cue", "[This is the 15-point architecture section. Trace the three lanes with a finger. Do not read the boxes.]"),
    ]),
    ("SLIDE 17", "7:30 - 8:05", "The answer key, and the rule that lets it fail.", [
        ("say", "Back to the answer key, for the part that matters."),
        ("say", "The judgements are pinned to <b>what a chunk is</b> - a NICE recommendation number, a KDIGO label, a distinctive phrase - never to a chunk ID, because IDs encode page and type and change whenever the chunker does. The corpus was rebuilt <b>four times</b> and the gold set survived every rebuild without re-annotation. The builder aborts if an anchor stops matching or goes ambiguous."),
        ("say", "Then the rule that makes it able to fail: every anchor was chosen by <b>reading the chunk in the guideline</b>, never by running retrieval and keeping what came back."),
        ("say", "That exact circularity is what made our first evaluator report hit-at-8 of 1.00 and MRR of 1.00. The same code, measured against pinned chunks, scores <b>0.276</b>."),
    ]),
    ("SLIDE 18", "8:05 - 8:35", "Three answer keys, because one can be gamed.", [
        ("say", "Results on all three. <b>0.725</b> nDCG on the main gold set. <b>0.841</b> on the held-out set, written from the chunks before any run. And <b>100 percent success at 5</b> on the clinician FAQ set - the only key whose questions did not originate with this project."),
        ("say", "The line underneath is the journey: 0.276 to 0.725. Questions returning zero relevant evidence, 26 down to 4. And the overfitting ratio - dev over test - from 2.56 down to 1.39."),
    ]),
    ("SLIDE 19", "8:35 - 9:05", "Measured, and rejected.", [
        ("say", "Last slide. What we built, measured, and threw away: MedCPT, HyDE, ColBERT, Contextual Retrieval, BM25 in the score, and three attempts at score-based abstention."),
        ("say", "A negative result <i>with a mechanism</i> is a stronger answer to 'why did you not use X?' than never having measured it."),
        ("say", "And the honesty notes are on the slide on purpose, because they would be found anyway. Our original evaluator could not fail. 16 of the 75 gold cases were written by us. n equals 67 is small for IR."),
        ("say", "Thank you - happy to take questions."),
    ]),
]

CUTS = [
    "<b>Slide 10, Embedding</b> - drop to one line: 'seven encoders measured; the reranker makes them nearly interchangeable, which is why we attacked the document side instead.'",
    "<b>Slide 06, Corpus</b> - keep the dollar-sign defect, drop the other two.",
    "<b>Slide 19, Ablation</b> - say only the 'negative result with a mechanism' line and stop.",
    "<b>Slide 13, Scoring</b> - skip the sweep table; say 'every constant was swept twice and came back where it started.'",
]

QA = [
    ("Did you actually deploy this in an EHR?",
     "No - and say so first. No vendor sandbox, no production FHIR server, no real patient data. What exists is a conformant CDS Hooks service: discovery and hook endpoints, tested over HTTP with a spec-shaped request. First call after startup is 4 milliseconds; a hook arriving during warm-up gets a 503 with Retry-After rather than a held connection. Missing before it could be deployed: JWT authentication and a FHIR fetch fallback. The transcript is in CDS_HOOKS_WIRE_TEST.md."),
    ("If they already have the container, why keep paying?",
     "They are not paying for the container - they are paying for it to stay correct. The corpus goes stale: KDIGO 2024 replaced KDIGO 2012, and NICE NG203 already carries 2021 amendments. Every revision means re-parsing, re-chunking and re-running the 141-question gold set to check retrieval did not silently get worse - which no trust will do for itself. A frozen container keeps answering confidently from superseded guidance, which is worse than not having it."),
    ("And in a year when the guideline does not change?",
     "Then that answer does not hold, and the honest one is that what recurs is not content. Epic ships quarterly releases, FHIR versions shift, LOINC updates twice a year - and the prefetch is matched on LOINC codes, so a service called correctly in January can stop being called in October because the record system moved, not the medicine. It is a container carrying PyTorch and a web server, so CVEs land continuously and no infosec team leaves an unpatched container on a clinical network. And a DCB0129 hazard log is maintained, not filed. That is why it is priced as implementation plus support rather than as a content subscription - the same reason a trust pays maintenance on PACS in a year when nothing about the underlying medicine changed."),
    ("Why not just fine-tune a model, or use a bigger LLM?",
     "Because the requirement is traceability, not fluency. A fine-tuned model cannot show you the rectangle on the page. Everything generated here is subordinate to retrieved text and is discarded if a single number fails to trace."),
    ("Is 67 questions too few?",
     "Yes, and that is stated on the slide. It is why nothing is defended as an argmax - anything under about 0.05 is not measurable at that n, so constants are defended as indistinguishable from their neighbours, and every delta carries a paired bootstrap interval."),
    ("You wrote your own gold set. Is that not circular?",
     "16 of 75 cases, yes, and it is on the slide rather than buried. Two things limit it: every anchor came from reading the guideline rather than from running retrieval, and anchors are text-based so they survived four corpus rebuilds. If you discount the rest, the held-out slice and the externally-sourced FAQ set are the numbers to trust."),
    ("What is the weakest part?",
     "Paraphrase questions - ones sharing no vocabulary with the source. doc2query closed most of the gap but not all of it. And the premise gate catches contamination, not false relationships: if a number is genuinely stated in exactly the context you ask about, a wrong claim about it still gets answered. The verifier is what stands behind that."),
    ("Why is nDCG only 0.72? That does not sound high.",
     "Because it is measured against pinned chunks with graded relevance, not topic labels. The original evaluator on this same project reported 1.00 - and it literally could not fail, because it counted a hit whenever a returned chunk carried a matching topic label and the retriever boosted on those same labels. 0.72 is what honest measurement of this task looks like."),
    ("Could it give a wrong dose?",
     "That is exactly the failure we measure separately, because nDCG cannot see it - returning the wrong row of the right table is a wrong answer with a real citation. Correct-row precision is 19 of 20. It caught a tenfold paediatric salt-dosing error that every aggregate metric scored as ordinary partial credit."),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def main() -> None:
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title="Nephrolex - speaker script", author="Nephrolex")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])

    flow = [Paragraph("Nephrolex - speaker script", H1),
            Paragraph("19 slides, about nine minutes. Times are cumulative. Bracketed lines "
                      "are stage directions, not words to say.", SUB)]

    for label, time, title, blocks in SCRIPT:
        group = [Paragraph(f"{label} &nbsp;&nbsp; {time}", SLIDE),
                 Paragraph(title, TITLE)]
        for kind, text in blocks:
            group.append(Paragraph(text, CUE if kind == "cue" else BODY))
        # keep a slide's header with at least its first line
        flow.append(KeepTogether(group[:3]))
        flow.extend(group[3:])

    flow.append(Paragraph("If you are running long", SEC))
    flow.append(Paragraph("Cut in this order. Each saves roughly 30 seconds.", BODY))
    for c in CUTS:
        flow.append(Paragraph(f"- {c}", BODY))
    flow.append(Paragraph("<b>Never cut:</b> the Patient B silence on slide 04, the boxed "
                          "table row on slide 06, or the 0.276 evaluator story on slide 17. "
                          "Those are the three moments that separate this from a demo.", BODY))

    flow.append(Paragraph("Questions to expect", SEC))
    for q, a in QA:
        flow.append(KeepTogether([Paragraph(f"<b>{q}</b>", BODY), Paragraph(a, BODY)]))
        flow.append(Spacer(1, 4))

    doc.build(flow)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
