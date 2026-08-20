"""A CDS Hooks service: guideline evidence delivered into the clinician's workflow.

What this is
------------
CDS Hooks is HL7's standard for an electronic health record to call an outside service
at a defined moment - a chart being opened, a drug being prescribed - and receive back
*cards*: small boxes of guidance the record shows in the corner of the screen. It is
vendor-neutral and FHIR-based, so any record system that speaks it can call this service
without bespoke work.

It matters because the system otherwise is a website, and a clinician will not leave
their record software to visit one. This turns retrieval into something that arrives
where the decision is being made.

Why silence is the feature
--------------------------
Deployed decision support fails in a well-documented way: drug-safety alerts are
overridden in 49-96% of cases, drug-interaction alerts as high as 95%, and every study
names the same cause - too many irrelevant boxes, so clinicians stop reading them.
Reducing unnecessary triggers has been measured to raise adoption from 6.5% to 29.3% at
one centre and 14.7% to 42.6% at another.

This system already refuses to answer when the evidence will not support one: scope
gates, a premise check, a verifier that discards unsupported numbers. Here that becomes
the product rather than a safeguard. **A trigger that fires but produces no grounded
answer emits no card at all.** Returning nothing is a correct outcome, and the code below
treats it as one.

What a card carries
-------------------
Verbatim guideline text and its citation - never generated prose. The extractive quote
is what `answer.py` already resolved to a page and a bounding box, so a clinician can
open the source and see the sentence under its own box. Generated phrasing is deliberately
not used here: it is an enhancement on a screen the user chose to look at, not something
to push into a chart unprompted.

Endpoints (wired in scripts/demo_server.py):
    GET  /cds-services              discovery - what this service offers
    POST /cds-services/ckd-guidance the hook itself - returns {"cards": [...]}
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Callable

# LOINC codes for the two measurements CKD guidance turns on. Several are in use for
# each; a record system may send any of them.
EGFR_LOINC = {"33914-3", "48642-3", "48643-1", "62238-1", "77147-7", "98979-8"}
ACR_LOINC = {"9318-7", "14959-1", "13705-9", "32294-1", "76401-9"}

SERVICE_ID = "ckd-guidance"


@dataclass(frozen=True)
class Facts:
    """What we managed to read about the patient. Any field may be missing."""

    egfr: float | None = None
    acr: float | None = None

    def known(self) -> bool:
        return self.egfr is not None or self.acr is not None


@dataclass(frozen=True)
class Trigger:
    """A patient state, and the guideline question it raises.

    The condition is deliberately a plain predicate over two numbers rather than a
    learned or scored rule. A clinician can read it, disagree with it, and point at the
    line. `question` is then answered by the ordinary retrieval path - this file adds no
    clinical knowledge of its own, it only decides when it is worth asking.
    """

    name: str
    applies: Callable[[Facts], bool]
    question: str
    summary: str  # what the clinician reads first; the spec caps this at 140 characters
    indicator: str  # "info" | "warning" | "critical", per the CDS Hooks spec
    rationale: str


TRIGGERS: tuple[Trigger, ...] = (
    Trigger(
        "referral_threshold",
        lambda f: (f.egfr is not None and f.egfr < 30) or (f.acr is not None and f.acr >= 70),
        "When should an adult with CKD be referred to specialist kidney care?",
        "Referral to specialist kidney care may be indicated",
        "warning",
        "eGFR below 30 or ACR at or above 70 mg/mmol",
    ),
    Trigger(
        "bp_target",
        lambda f: f.acr is not None and f.acr >= 3,
        "What blood pressure target applies to adults with CKD?",
        "Blood pressure target in CKD with albuminuria",
        "info",
        "albuminuria present",
    ),
    Trigger(
        "monitoring_frequency",
        lambda f: f.egfr is not None and f.egfr < 60,
        "How often should eGFR be monitored in people with CKD?",
        "Monitoring frequency for reduced eGFR",
        "info",
        "eGFR below 60",
    ),
    Trigger(
        "sglt2_eligibility",
        lambda f: f.egfr is not None and 20 <= f.egfr < 60,
        "Who with CKD should be offered an SGLT2 inhibitor?",
        "SGLT2 inhibitor may be indicated",
        "info",
        "eGFR between 20 and 60",
    ),
    Trigger(
        "staging",
        lambda f: f.egfr is not None and f.egfr < 90,
        "What GFR category does this eGFR fall into?",
        "GFR category for this result",
        "info",
        "eGFR below 90",
    ),
)


def discovery() -> dict:
    """The document a record system fetches to learn what this service offers.

    `prefetch` asks the record to send the patient's eGFR and ACR with the hook, so the
    service needs no credentials of its own and holds no patient data between calls.
    """
    return {
        "services": [
            {
                "hook": "patient-view",
                "id": SERVICE_ID,
                "title": "CKD guideline evidence (KDIGO 2024 / NICE NG203)",
                "description": (
                    "Surfaces the guideline text that applies to this patient's kidney "
                    "function, quoted verbatim with a citation to page and position. "
                    "Returns no card when the guidelines do not clearly answer."
                ),
                "prefetch": {
                    "egfr": (
                        "Observation?patient={{context.patientId}}"
                        "&code=http://loinc.org|33914-3&_sort=-date&_count=1"
                    ),
                    "acr": (
                        "Observation?patient={{context.patientId}}"
                        "&code=http://loinc.org|9318-7&_sort=-date&_count=1"
                    ),
                },
            }
        ]
    }


def _observation_value(resource: dict) -> tuple[str | None, float | None]:
    """(LOINC code, numeric value) from a FHIR Observation, if both are present."""
    codings = ((resource.get("code") or {}).get("coding") or [])
    code = next((c.get("code") for c in codings if c.get("code")), None)
    quantity = resource.get("valueQuantity") or {}
    value = quantity.get("value")
    if value is None:
        return code, None
    try:
        return code, float(value)
    except (TypeError, ValueError):
        return code, None


def read_facts(prefetch: dict | None) -> Facts:
    """Pull eGFR and ACR out of whatever the record sent.

    Tolerant on purpose. A record may send a Bundle, a bare Observation, or nothing at
    all, and the codes for these two measurements are not standardised to one apiece. A
    value we cannot read leaves its field None, which simply means fewer triggers fire -
    never a card built on a number we guessed at.
    """
    egfr = acr = None
    for entry in (prefetch or {}).values():
        if not isinstance(entry, dict):
            continue
        resources = []
        if entry.get("resourceType") == "Bundle":
            resources = [e.get("resource") or {} for e in (entry.get("entry") or [])]
        elif entry.get("resourceType") == "Observation":
            resources = [entry]
        for resource in resources:
            code, value = _observation_value(resource)
            if value is None:
                continue
            if code in EGFR_LOINC and egfr is None:
                egfr = value
            elif code in ACR_LOINC and acr is None:
                acr = value
    return Facts(egfr=egfr, acr=acr)


def build_cards(facts: Facts, answerer: Callable[[str], dict], limit: int = 3) -> list[dict]:
    """Cards for a patient, or an empty list.

    An empty list is a real answer and the commonest one worth defending: a trigger fires
    on the patient's numbers, retrieval runs, and if it declines or produces nothing
    citable then no card is shown. The clinician sees silence rather than a box they will
    learn to dismiss.
    """
    cards: list[dict] = []
    for trigger in TRIGGERS:
        if len(cards) >= limit:
            break
        if not trigger.applies(facts):
            continue

        answer = answer_for(trigger.question, answerer)
        if answer.get("status") != "answered":
            continue
        quotes = answer.get("quotes") or []
        if not quotes:
            continue

        top = quotes[0]
        text = " ".join((top.get("text") or "").split())
        citation = top.get("citation")
        if not text or not citation:
            continue

        detail = f"> {text}\n\n**{citation}**"
        others = [q for q in quotes[1:3] if q.get("citation")]
        if others:
            detail += "\n\nAlso relevant: " + "; ".join(q["citation"] for q in others)

        cards.append({
            "uuid": str(uuid.uuid4()),
            "summary": trigger.summary,
            "indicator": trigger.indicator,
            "detail": detail,
            "source": {
                "label": top.get("publisher") or "CKD guideline",
                "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
            },
            # Not part of the spec's required fields; carried so a reviewer can see why
            # this card appeared at all and check the rule that produced it.
            "extension": {
                "trigger": trigger.name,
                "because": trigger.rationale,
                "question": trigger.question,
                "chunk_id": top.get("chunk_id"),
                "page": top.get("page_start"),
            },
        })
    return cards


# The guideline does not change between patients, and every trigger asks a fixed
# question. Only *which* triggers fire depends on the patient, so each question is
# retrieved once and reused - a hook then costs a dictionary lookup rather than up to
# five retrievals. This matters because the hook fires when a chart opens: a service
# that takes ten seconds to answer is one a record system will time out on, and the
# clinician would experience the delay as the record being slow.
_ANSWERS: dict[str, dict] = {}
_LOCK = threading.Lock()


def answer_for(question: str, answerer: Callable[[str], dict]) -> dict:
    """Retrieve once per distinct question, then serve from memory."""
    with _LOCK:
        cached = _ANSWERS.get(question)
    if cached is not None:
        return cached
    result = answerer(question)
    with _LOCK:
        _ANSWERS[question] = result
    return result


def warm(answerer: Callable[[str], dict]) -> int:
    """Pre-answer every trigger question. Called at startup so the first hook is fast."""
    for trigger in TRIGGERS:
        answer_for(trigger.question, answerer)
    return len(_ANSWERS)


def handle(request: dict, answerer: Callable[[str], dict],
           warm_first: bool = False) -> dict:
    """One hook invocation. Returns the CDS Hooks response body."""
    if warm_first and not _ANSWERS:
        warm(answerer)
    facts = read_facts(request.get("prefetch"))
    if not facts.known():
        # No kidney measurements means nothing here applies. Saying so with an empty
        # card list is the whole point; an "I have no data" card would be noise.
        return {"cards": []}
    return {"cards": build_cards(facts, answerer)}
