"""Scope gating for CKD guideline questions.

The declared scope is adult, non-pregnant, non-dialysis CKD management answerable
from KDIGO 2024 and NICE NG203. Anything else must be declined with a reason, not
answered from whatever the retriever happened to return.

Two design decisions worth stating, because both were wrong in the earlier system:

1. Scope is decided *before* retrieval, not by boosting out-of-scope evidence.
   The previous pipeline added +0.45 to the score of pregnancy/paediatric/dialysis
   chunks when the query matched those topics, which pulled exactly the material
   the system must not answer from into the generator's context. Measured on the
   gold set, that also inverted the confidence signal: out-of-scope questions
   scored *higher* (1.434) than answerable ones (0.853), so any threshold-based
   abstention gate would have abstained on real questions and answered the unsafe
   ones.

2. Matching is on morphology plus phrase patterns, not a keyword list. "Expecting",
   "gravid" and "20 weeks" all mean pregnancy without containing "pregnant"; a
   keyword gate fails silently on exactly the paraphrases a real clinician types.

This gate is deliberately conservative: it prefers a false decline to a false
answer, and it always reports which rule fired so the behaviour is auditable.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field


@dataclass
class ScopeVerdict:
    in_scope: bool
    category: str
    reason: str
    matched: list[str] = field(default_factory=list)
    guidance: str = ""

    def as_dict(self) -> dict:
        return {
            "in_scope": self.in_scope,
            "category": self.category,
            "reason": self.reason,
            "matched": self.matched,
            "guidance": self.guidance,
        }


# Ordered: the first category to match wins, most urgent first.
OUT_OF_SCOPE_RULES: list[tuple[str, list[str], str, str]] = [
    (
        "emergency",
        [
            r"\bECG changes?\b",
            r"\b(?:potassium|K\+)\s*(?:of\s*)?(?:is\s*)?(?:[6-9]|\d{2})(?:\.\d+)?\b",
            r"\bright now\b|\bemergency\b|\bemergenc|\burgently\b|\bcollaps|\bunconscious\b",
            r"\banuri|\bcrush injury\b|\bpulmonary o?edema\b",
            r"\bacute kidney injury\b|\bAKI\b",
        ],
        "This looks like an acute or emergency presentation.",
        "Escalate to acute care immediately. These guidelines cover chronic management "
        "and must not be used for emergency decisions.",
    ),
    (
        "pregnancy",
        [
            r"\bpregnan\w*\b",
            r"\bexpecting\b|\bgravid\w*\b|\bantenatal\b|\bobstetric\w*\b",
            r"\b\d{1,2}\s*weeks(?:'|\s+)?(?:pregnant|gestation)?\b",
            r"\bbreast\s?feed\w*\b|\blactati\w*\b|\bpostpartum\b",
        ],
        "CKD in pregnancy is outside the declared scope of this system.",
        "Pregnancy-specific CKD management requires specialist obstetric and nephrology "
        "input. Neither source guideline is scoped to answer it here.",
    ),
    (
        "paediatric",
        [
            r"\bchild\w*\b|\bchildren\b|\bpaediatric\b|\bpediatric\b",
            r"\binfant\w*\b|\bneonat\w*\b|\btoddler\b|\badolescen\w*\b",
            r"\byoung (?:person|people)\b",
            r"\b\d{1,2}[- ]year[- ]old\b(?=.*\b(?:boy|girl|child)\b)",
            r"\b([1-9]|1[0-7])[- ]year[- ]old\b",
        ],
        "Paediatric CKD is outside the declared adult scope.",
        "Both guidelines contain paediatric recommendations, but this system is scoped "
        "to adults. Refer to the paediatric sections directly with specialist input.",
    ),
    (
        "kidney_replacement",
        [
            # Intervening words are the norm: "start my patient on haemodialysis".
            r"\b(?:start|begin|initiat|commenc)\w*\b[^.?!]{0,40}?\b(?:h(?:a)?emo)?dialysis\b",
            r"\b(?:start|begin|initiat|commenc)\w*\b[^.?!]{0,40}?\b(?:kidney|renal) replacement\b",
            r"\bwhen to (?:start|begin) (?:dialysis|kidney replacement|renal replacement)\b",
            r"\btransplant\w*\b",
            r"\bperitoneal dialysis\b",
        ],
        "Dialysis initiation and transplantation are outside the declared scope.",
        "Neither KDIGO's CKD guideline nor NICE NG203 covers modality selection or the "
        "decision to start kidney replacement therapy.",
    ),
    (
        "individual_prescribing",
        [
            r"\bwhat dose (?:should|do) I (?:give|prescribe)\b",
            r"\bprescribe for (?:my|this) patient\b",
            r"\bhow many (?:mg|milligrams)\b",
        ],
        "This asks for an individualised prescribing decision.",
        "These guidelines state population-level recommendations. Individual dosing "
        "requires a prescriber with the full clinical picture.",
    ),
]


# Vocabulary that marks a question as being about kidneys at all.
# Conditions that belong squarely to another specialty.
#
# The vocabulary test in `_unknown_concepts` asks whether each query term appears
# *anywhere* in the corpus. That leaks: "what about brain tumor?" was answered,
# because "brain" and "tumor" each occur twice in 1,485 chunks - brain natriuretic
# peptide, tumour lysis - so neither is unknown, and the gate never fires. The
# spelling decided the outcome, too: "tumour" is absent from the corpus and was
# refused, while "tumor" was not.
#
# Three general fixes were tried and rejected before settling for a named list:
#   - requiring a kidney term outright: 14 answerable gold questions carry none
#     ("What is a prescribing cascade?", "Their filtration number is 95 ...").
#   - requiring the query's terms to co-occur in one chunk: refuses "should I start
#     a flozin" and "protein in the urine keeps rising", the exact clinician
#     phrasing doc2query exists to serve.
#   - requiring one term to be well covered: "how do I treat asthma" and "should I
#     start a flozin" both peak at 19 chunks, because generic verbs dominate.
#
# So this names *conditions*, never treatments - "we need a precise number before
# dosing chemotherapy" is a real gold question about GFR accuracy - and is consulted
# only when the query carries no kidney vocabulary at all. Anything CKD guidelines
# genuinely manage is deliberately absent: fracture (CKD-MBD), anaemia, diabetes,
# hypertension, gout and transplantation are all in scope elsewhere.
OTHER_SPECIALTY = re.compile(
    r"\b(?:"
    r"brain tumou?rs?|gliomas?|meningiomas?|"
    r"(?:lung|breast|prostate|colon|bowel|ovarian|cervical|pancreatic|gastric|"
    r"oesophageal|esophageal|skin|thyroid) cancers?|"
    r"melanomas?|lymphomas?|leuk(?:a)?emias?|"
    r"asthma|COPD|emphysema|bronchiectasis|"
    r"appendicitis|cholecystitis|diverticulitis|"
    r"migraines?|epilep\w*|seizures?|strokes?|dementia|alzheimer\w*|parkinson\w*|"
    r"glaucoma|cataracts?|"
    r"psoriasis|eczema|"
    r"pneumonias?|tuberculosis"
    r")\b",
    re.I,
)


DOMAIN_TERMS = re.compile(
    r"\b(?:kidney\w*|renal|nephro\w*|CKD|ESKD|eGFR|GFR|glomerular|creatinine|cystatin|"
    r"albumin\w*|proteinuria|ACR|UACR|dialysis|ur(?:a)?emi\w*|KFRE|SGLT2|RASi|"
    r"ACE inhibitor|ARB|nephritis|nephropathy|ha?ematuria|potassium|hyperkal\w*|"
    r"phosphate|parathyroid|an?aemia|ESA|erythropo\w*|ferritin|ha?emoglobin|iron|"
    r"bicarbonate|acidosis|urate|allopurinol|transplant\w*|filtration rate|"
    r"NG203|KDIGO|NICE)\b",
    re.I,
)


def in_domain(query: str) -> bool:
    """Is this a kidney question at all?

    Distinct from the confidence gate, which asks whether the corpus can answer an
    in-domain question. "What is the recommended treatment for acute appendicitis?"
    is not a low-confidence CKD question - it is not a CKD question, and no
    retrieval score threshold reliably says so. Measured on the gold set, the raw
    BM25 top score for answerable questions spans 11.7-61.0 and for unanswerable
    ones 14.8-29.4: fully overlapping, so a score threshold cannot separate them.
    Vocabulary membership can.
    """
    return bool(DOMAIN_TERMS.search(query))


def other_specialty(query: str) -> str | None:
    """A condition from another specialty, in a query with no kidney vocabulary.

    Both halves are required. The list alone would refuse "does CKD change how I
    manage asthma?", which is at least arguably in scope; the absence of kidney
    vocabulary alone refuses fourteen answerable gold questions. Together they catch
    a question that is about something else entirely and mentions kidneys nowhere.
    """
    if in_domain(query):
        return None
    match = OTHER_SPECIALTY.search(query)
    return match.group(0).lower() if match else None


# Question genres a clinical guideline is not written to answer. This is a separate
# axis from topic: "How many people worldwide have CKD?" is squarely about CKD, and
# retrieval scores it above the median answerable question on both cosine (0.838 vs
# 0.821) and the cross-encoder (1.00 vs 0.97). No relevance threshold can separate
# it, because relevance is not the problem - the corpus states what to *do*, not how
# many, where, how much, or who discovered it. Recognising the genre is the only
# signal that works.
NOT_GUIDELINE_QUESTIONS: list[tuple[str, list[str], str]] = [
    (
        "epidemiology",
        [
            r"\bwhich country\b|\bwhat country\b|\bwhich nation\b",
            r"\bhow many (?:people|patients|adults|cases|deaths)\b",
            r"\b(?:most|highest|lowest|fewest) (?:cases|prevalence|incidence|rate)\b",
            r"\bworldwide\b|\bglobally\b|\bacross the world\b",
            r"\bwhat percentage of (?:the )?(?:population|people|adults)\b",
            # Demographics of a patient population. "Average age of people with
            # hypertension" uses only in-corpus vocabulary, so the unknown-concept
            # check finds nothing to object to, and the question sailed through to a
            # confident answer built on unrelated narrative. A guideline states
            # thresholds and actions; it does not publish the age of its readership.
            r"\b(?:average|median|mean|typical) age\b",
            r"\bwhat age (?:do|does|are|is)\b",
            # "prevalence" and "how common" only count as epidemiology when paired
            # with a population-scale marker. "How common is hyperkalaemia after
            # starting an ACE inhibitor?" is a clinical question the guidelines do
            # address, and blocking it would be worse than answering the other kind.
            r"\b(?:prevalence|incidence|how common)\b(?=.{0,60}\b(?:worldwide|globally|population|country|nation|epidemi)\w*)",
            r"\b(?:worldwide|globally|population|country|nation)\b(?=.{0,60}\b(?:prevalence|incidence|how common)\b)",
        ],
        "Guidelines state what care to provide, not epidemiological statistics.",
    ),
    (
        "cost",
        [
            r"\bhow much does .{0,30}cost\b|\bwhat is the cost\b|\bwhat does .{0,20}cost\b",
            r"\bprice of\b|\bcost[- ]effectiveness ratio\b|\bhow expensive\b",
            r"\bfunding\b|\breimbursement\b",
        ],
        "Guidelines state clinical recommendations, not costs or health economics.",
    ),
    (
        "history",
        [
            r"\bwho (?:discovered|invented|founded|first described)\b",
            r"\bwhen was .{0,30}(?:discovered|invented|first described)\b",
            # Publication and revision dates are facts about the document, not
            # guidance from it. The bare "history of" pattern happened to catch these
            # before it was narrowed; they need a rule of their own rather than a
            # side effect of one.
            r"\bwhen (?:was|were|did) .{0,40}(?:published|written|released|issued|updated|revised)\b",
            r"\b(?:publication|release) date\b",
            # "History of" is far more often clinical than bibliographic - a patient
            # with a history of AKI, of diabetes, of stones. Matching it bare refused
            # ordinary phrasing a clinician uses constantly. It now only counts when
            # what follows is the guideline or the field itself.
            r"\bhistory of (?:the )?(?:guideline|guidance|recommendation|KDIGO|NICE|"
            r"nephrology|medicine|the field)\b",
        ],
        "Guidelines state current recommendations, not medical history.",
    ),
]


def question_genre(query: str) -> tuple[str, str] | None:
    """Return (genre, reason) when a question asks for something guidelines never state."""
    for genre, patterns, reason in NOT_GUIDELINE_QUESTIONS:
        if any(re.search(pattern, query, re.I) for pattern in patterns):
            return genre, reason
    return None


# Phrasing that places an acute event in the past. Deliberately narrow: it must be a
# retrospective marker sitting right before the event, so "what do I give right now
# for AKI" is untouched while "monitoring after an episode of AKI" is allowed through.
POST_ACUTE_CONTEXT = re.compile(
    r"\b(?:after|following|since|post|history of|episode of|recovery from|recovered from|"
    r"previous|prior|past)\b[^.?!]{0,40}?"
    r"\b(?:acute kidney injury|AKI)\b",
    re.I,
)


def classify(query: str) -> ScopeVerdict:
    """Decide whether a question may be answered from the CKD guideline corpus."""
    genre = question_genre(query)
    if genre:
        name, reason = genre
        return ScopeVerdict(
            in_scope=False,
            category=f"not_a_guideline_question:{name}",
            reason=reason,
            matched=[],
            guidance=(
                "KDIGO 2024 and NICE NG203 are clinical practice guidelines: they say what "
                "to assess, when to refer and which thresholds to act on. For this you need "
                "an epidemiological or health-economics source instead."
            ),
        )

    # An acute event named as history is not an acute presentation. "How long should
    # someone be monitored for CKD after an episode of acute kidney injury?" is
    # chronic follow-up - NICE 1.1.25 answers it directly, for at least 3 years - and
    # the emergency rule refused it purely for containing the phrase "acute kidney
    # injury". Refusing a question the guideline explicitly answers is the failure
    # mode the scope gate is supposed to trade against, not an instance of it.
    post_acute = bool(POST_ACUTE_CONTEXT.search(query))

    for category, patterns, reason, guidance in OUT_OF_SCOPE_RULES:
        if category == "emergency" and post_acute:
            continue
        matched = [pattern for pattern in patterns if re.search(pattern, query, re.I)]
        if matched:
            return ScopeVerdict(
                in_scope=False,
                category=category,
                reason=reason,
                matched=matched,
                guidance=guidance,
            )
    return ScopeVerdict(in_scope=True, category="adult_ckd", reason="Adult CKD question within scope.")


def refusal_text(verdict: ScopeVerdict) -> str:
    return (
        f"I can't answer this one. {verdict.reason}\n\n{verdict.guidance}\n\n"
        "In scope: adult, non-pregnant CKD assessment and management as covered by "
        "KDIGO 2024 and NICE NG203."
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queries = sys.argv[1:]
    if not queries:
        queries = [
            "When should an adult with CKD be referred to specialist kidney care?",
            "How should I manage CKD in a woman who is 20 weeks pregnant?",
            "My patient is expecting and has stage 3 kidney disease - what about her medication?",
            "What ACR threshold triggers specialist referral for a 6-year-old?",
            "At what eGFR should I start my patient on haemodialysis?",
            "Potassium is 7.2 with ECG changes - what do I give right now?",
            "What blood pressure target applies in CKD?",
        ]
    for query in queries:
        verdict = classify(query)
        flag = "IN SCOPE " if verdict.in_scope else "DECLINED "
        print(f"{flag} [{verdict.category}] {query}")
        if not verdict.in_scope:
            print(f"          -> {verdict.reason}")
    print()
    print(json.dumps(classify(queries[0]).as_dict(), indent=2))


if __name__ == "__main__":
    main()
