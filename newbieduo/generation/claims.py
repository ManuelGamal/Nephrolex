"""Structured clinical claims, and disagreement between guidelines.

The point of carrying two guidelines is not to cite twice as much text - it is to
notice when KDIGO 2024 and NICE NG203 say *different things*, which they do, because
they were written three years apart from different evidence bases. The clearest
example in this corpus:

    KDIGO 2024, Recommendation 3.4.1 : target systolic blood pressure < 120 mmHg
    NICE NG203,  1.6.1               : aim for clinic systolic below 140 mmHg
    NICE NG203,  1.6.2               : ACR >= 70 mg/mmol -> below 130 mmHg

A retrieval system that returns both and says nothing leaves the clinician to spot
the contradiction. This module turns each recommendation into a comparable claim -
(quantity, comparator, value, unit, population) - so the divergence can be stated
explicitly and attributed to each source.

Deliberately rule-based rather than LLM-based: a claim that a guideline says
"< 120 mmHg" must be extracted from the text, never generated, or the safety story
collapses. Every field traces to a span in the source chunk.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


from newbieduo.paths import ROOT
# Quantities worth comparing across guidelines, with the units they are stated in.
QUANTITIES = {
    "systolic_blood_pressure": {
        "patterns": [r"systolic (?:blood pressure|bp)", r"\bSBP\b", r"clinic systolic"],
        "units": {"mmhg"},
        "label": "Systolic blood pressure target",
    },
    # eGFR and ACR numbers are role-dependent: the same "eGFR < 30" is a staging
    # boundary in one sentence, a referral trigger in another and a drug-eligibility
    # cut-off in a third. Comparing them across guidelines without knowing the role
    # manufactures contradictions - an early version reported "KDIGO < 15 vs NICE
    # < 60" as a disagreement when one was the G5 boundary and the other the
    # threshold for confirmatory testing. They are extracted for display but not
    # compared.
    "egfr": {
        "patterns": [r"\beGFR\b", r"\bGFR\b", r"glomerular filtration rate"],
        "units": {"ml/min", "ml/min/1.73 m2", "ml/min per 1.73 m2"},
        "label": "eGFR threshold",
        "comparable": False,
    },
    "acr": {
        "patterns": [r"\bACR\b", r"albumin[- ]creatinine ratio", r"albuminuria"],
        "units": {"mg/mmol", "mg/g"},
        "label": "Albuminuria (ACR) threshold",
        "comparable": False,
    },
    "kidney_failure_risk": {
        "patterns": [r"kidney failure risk", r"\bKFRE\b", r"risk of (?:needing )?(?:renal|kidney) replacement"],
        "units": {"%"},
        "label": "Kidney failure risk threshold",
        "comparable": False,
    },
    "potassium": {
        "patterns": [r"\bpotassium\b", r"hyperkal"],
        "units": {"mmol/l"},
        "label": "Serum potassium threshold",
    },
    "protein_intake": {
        "patterns": [r"protein intake", r"dietary protein"],
        "units": {"g/kg"},
        "label": "Dietary protein intake",
    },
    "ferritin": {
        "patterns": [r"\bferritin\b"],
        "units": {"micrograms/litre", "ng/ml"},
        "label": "Serum ferritin ceiling",
    },
    "haemoglobin": {
        "patterns": [r"\bhaemoglobin\b", r"\bhemoglobin\b", r"\bHb\b"],
        "units": {"g/l", "g/dl"},
        "label": "Haemoglobin target",
    },
}

COMPARATORS = [
    (r"below|less than|under|lower than|<=|<", "<"),
    (r"above|more than|greater than|over|at least|>=|>", ">"),
    (r"between", "between"),
]

# Units must tolerate the spacing PDFs actually produce: KDIGO prints "< 120 mm Hg"
# with a space, which a naive "mmHg" pattern silently misses - and a missed unit
# means a missed claim, which means a real disagreement reported as agreement.
UNIT_PATTERN = (
    r"mm\s?Hg|ml\s?/\s?min(?:\s*per\s*1\.73\s*m\s?2|\s?/\s?1\.73\s*m\s?2)?|"
    r"mg\s?/\s?mmol|mg\s?/\s?g|mmol\s?/\s?l|"
    r"micrograms?\s?/\s?litre|ng\s?/\s?ml|g\s?/\s?kg|g\s?/\s?l|g\s?/\s?dl|%"
)

# Population qualifiers change what a threshold applies to; a comparison that
# ignores them would call two compatible statements a conflict.
POPULATIONS = [
    (r"\bchildren\b|\byoung people\b|\badolescen|\bpaediatric\b|\bpediatric\b", "children"),
    (r"\bpregnan|\bgestation", "pregnancy"),
    (r"\bdiabet", "diabetes"),
    (r"\bdialysis\b|\bkidney replacement\b|\btransplant\b", "kidney replacement"),
    (r"\badults?\b", "adults"),
]


# Units are lower-cased for matching; restore clinical casing for anything shown to
# a reader. "< 120 mmhg" on screen undercuts the credibility of the number next to it.
UNIT_DISPLAY = {
    "mmhg": "mmHg",
    "ml/min/1.73m2": "ml/min/1.73m²",
    "mg/mmol": "mg/mmol",
    "mg/g": "mg/g",
    "mmol/l": "mmol/L",
    "micrograms/litre": "micrograms/litre",
    "ng/ml": "ng/mL",
    "g/kg": "g/kg",
    "g/l": "g/L",
    "g/dl": "g/dL",
}


@dataclass
class Claim:
    quantity: str
    label: str
    comparator: str
    value: float
    value_high: float | None
    unit: str
    population: str
    document_id: str
    publisher: str
    citation: str
    chunk_id: str
    span: str

    def magnitude(self) -> str:
        unit = UNIT_DISPLAY.get(self.unit, self.unit)
        if self.comparator == "between" and self.value_high is not None:
            return f"{self.value:g}-{self.value_high:g} {unit}"
        if self.comparator == "=":
            return f"{self.value:g} {unit}"
        return f"{self.comparator} {self.value:g} {unit}"

    def statement(self) -> str:
        return f"{self.label} {self.magnitude()} ({self.population})"


def _detect_population(text: str) -> str:
    for pattern, name in POPULATIONS:
        if re.search(pattern, text, re.I):
            return name
    return "unspecified"


def _normalise_unit(unit: str) -> str:
    unit = re.sub(r"\s+", "", unit.lower())
    if unit.startswith("ml/min"):
        return "ml/min/1.73m2"
    if unit.startswith("micrograms"):
        return "micrograms/litre"
    return unit


def extract_claims(chunk: dict) -> list[Claim]:
    """Pull comparable numeric claims out of one retrieved chunk."""
    text = chunk.get("raw_text", "")
    meta = chunk.get("metadata", chunk)
    claims: list[Claim] = []

    for quantity, spec in QUANTITIES.items():
        if not any(re.search(pattern, text, re.I) for pattern in spec["patterns"]):
            continue

        for match in re.finditer(
            rf"(below|less than|under|lower than|above|more than|greater than|over|at least|between|<=|>=|<|>)?"
            rf"\s*(\d+(?:\.\d+)?)\s*(?:(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*)?({UNIT_PATTERN})",
            text,
            re.I,
        ):
            raw_comparator, value, value_high, unit = match.groups()
            unit_norm = _normalise_unit(unit)
            if unit_norm not in {_normalise_unit(u) for u in spec["units"]}:
                continue

            comparator = "="
            if raw_comparator:
                for pattern, symbol in COMPARATORS:
                    if re.fullmatch(pattern, raw_comparator.strip(), re.I):
                        comparator = symbol
                        break
            if value_high:
                comparator = "between"

            window = text[max(0, match.start() - 120) : match.end() + 40]
            claims.append(
                Claim(
                    quantity=quantity,
                    label=spec["label"],
                    comparator=comparator,
                    value=float(value),
                    value_high=float(value_high) if value_high else None,
                    unit=unit_norm,
                    population=_detect_population(window),
                    document_id=meta.get("document_id", ""),
                    publisher=meta.get("publisher", ""),
                    citation=meta.get("citation", ""),
                    chunk_id=chunk.get("id", ""),
                    span=" ".join(window.split()),
                )
            )
    return claims


def _bounds(claim: Claim) -> tuple[float, float]:
    """Interval a claim permits, for overlap testing."""
    if claim.comparator == "<":
        return (0.0, claim.value)
    if claim.comparator == ">":
        return (claim.value, float("inf"))
    if claim.comparator == "between" and claim.value_high is not None:
        return (claim.value, claim.value_high)
    return (claim.value, claim.value)


def _headline(claims: list[Claim]) -> Claim:
    """The claim that best represents a guideline's position on a quantity.

    A recommendation often states a bound and then a bracketing range ("below 140
    mmHg (target range 120 to 139)"). The bound is the recommendation; the range
    restates it. Prefer the bound.
    """
    bounded = [claim for claim in claims if claim.comparator in {"<", ">"}]
    return (bounded or claims)[0]


def compare_across_guidelines(claims: list[Claim]) -> list[dict]:
    """Group claims by quantity and population, then classify agreement.

    Only claims about the same quantity, in the same unit, for the same population
    are compared - otherwise "systolic < 120 for adults" and "systolic < 130 for
    children" would be reported as a contradiction when they are not.

    Three outcomes, because "compatible" and "the same" are not the same thing.
    KDIGO targets systolic < 120 and NICE targets < 140; a value of 119 satisfies
    both, so they are not strictly incompatible - but they are plainly different
    recommendations, and a clinician needs to be told that. Reporting "agree" there
    would be true logic and useless advice.
    """
    groups: dict[tuple[str, str, str], list[Claim]] = {}
    for claim in claims:
        if not QUANTITIES.get(claim.quantity, {}).get("comparable", True):
            continue
        groups.setdefault((claim.quantity, claim.unit, claim.population), []).append(claim)

    findings: list[dict] = []
    for (quantity, unit, population), group in sorted(groups.items()):
        documents = {claim.document_id for claim in group}
        if len(documents) < 2:
            continue

        by_doc: dict[str, list[Claim]] = {}
        for claim in group:
            by_doc.setdefault(claim.document_id, []).append(claim)

        lows, highs = zip(*(_bounds(claim) for claim in group))
        compatible = max(lows) <= min(highs)

        headlines = {doc: _headline(doc_claims) for doc, doc_claims in by_doc.items()}
        stated = {(claim.comparator, claim.value) for claim in headlines.values()}

        if not compatible:
            status = "incompatible"
        elif len(stated) > 1:
            status = "differs"
        else:
            status = "agree"

        findings.append(
            {
                "quantity": quantity,
                "label": group[0].label,
                "unit": unit,
                "population": population,
                "status": status,
                "headline": {doc: claim.magnitude() for doc, claim in headlines.items()},
                "by_guideline": {
                    doc: [
                        {
                            "statement": claim.statement(),
                            "citation": claim.citation,
                            "chunk_id": claim.chunk_id,
                            "span": claim.span,
                        }
                        for claim in doc_claims
                    ]
                    for doc, doc_claims in by_doc.items()
                },
            }
        )
    return findings


def analyse(results: list[dict]) -> dict:
    """Full claim analysis for a set of retrieved chunks."""
    claims = [claim for chunk in results for claim in extract_claims(chunk)]
    findings = compare_across_guidelines(claims)
    divergences = [f for f in findings if f["status"] in {"differs", "incompatible"}]
    return {
        "claim_count": len(claims),
        "claims": [asdict(claim) for claim in claims],
        "comparisons": findings,
        "divergences": divergences,
        "summary": [
            f"{f['label']}, {f['population']}: "
            + " vs ".join(f"{doc.split('_')[0].upper()} {text}" for doc, text in f["headline"].items())
            for f in divergences
        ],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    from newbieduo.retrieval.retrieve import retrieve

    parser = argparse.ArgumentParser(description="Extract and compare guideline claims.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    args = parser.parse_args()

    results = retrieve(args.query, args.top_k, args.dense_model, args.reranker_model)
    print(json.dumps(analyse(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
