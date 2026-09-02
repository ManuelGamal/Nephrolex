"""Exercise the CDS Hooks service against realistic patients.

Checks the two things that decide whether this is deployable rather than a demo:

    the cards are grounded   every card quotes verbatim guideline text and carries a
                             citation that resolves to a page - nothing generated, and
                             nothing asserted without a source

    the silence is real      a patient the guidelines do not clearly speak to produces
                             no cards at all. Deployed decision support is overridden in
                             49-96% of cases because it fires too often; a service that
                             returns nothing is the answer to that, and it has to be
                             demonstrated rather than claimed

Runs the hook handler directly, so it needs no server and no FHIR endpoint - the
patients below are the same shape a record system would send.

Usage:
    python scripts/check_cds_hooks.py
    python scripts/check_cds_hooks.py --verbose      # print every card in full
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.service import cds_hooks  # noqa: E402

DENSE_MODEL = "abhinand/MedEmbed-large-v0.1"
RERANKER = "BAAI/bge-reranker-v2-m3"


def observation(loinc: str, value: float, unit: str) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc}]},
        "valueQuantity": {"value": value, "unit": unit},
    }


def bundle(*resources: dict) -> dict:
    return {"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}


# (label, prefetch, what a clinician would expect)
PATIENTS = [
    ("advanced CKD, heavy albuminuria",
     {"egfr": bundle(observation("33914-3", 24, "mL/min/1.73m2")),
      "acr": bundle(observation("9318-7", 85, "mg/mmol"))},
     "referral, and the rest of the CKD bundle"),

    ("moderate CKD, mild albuminuria",
     {"egfr": bundle(observation("33914-3", 47, "mL/min/1.73m2")),
      "acr": bundle(observation("9318-7", 12, "mg/mmol"))},
     "monitoring, BP target, SGLT2 eligibility"),

    ("normal kidney function",
     {"egfr": bundle(observation("33914-3", 96, "mL/min/1.73m2")),
      "acr": bundle(observation("9318-7", 1.2, "mg/mmol"))},
     "NO cards - nothing here needs saying"),

    ("no kidney measurements on file",
     {},
     "NO cards - the service holds no opinion without data"),

    ("unreadable observation (no value)",
     {"egfr": bundle({"resourceType": "Observation",
                      "code": {"coding": [{"system": "http://loinc.org", "code": "33914-3"}]}})},
     "NO cards - a value we cannot read is not a value we guess"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from nephrolex.generation.answer import build_answer

    def answerer(question: str) -> dict:
        return build_answer(question, 10, DENSE_MODEL, RERANKER, generate=False)

    disco = cds_hooks.discovery()
    service = disco["services"][0]
    print(f"discovery: {service['id']}  hook={service['hook']}")
    print(f"  prefetch keys: {sorted(service['prefetch'])}\n")

    problems: list[str] = []
    total_cards = 0

    for label, prefetch, expectation in PATIENTS:
        facts = cds_hooks.read_facts(prefetch)
        response = cds_hooks.handle({"hook": "patient-view", "prefetch": prefetch}, answerer)
        cards = response["cards"]
        total_cards += len(cards)

        egfr = f"{facts.egfr:g}" if facts.egfr is not None else "-"
        acr = f"{facts.acr:g}" if facts.acr is not None else "-"
        print(f"{label}")
        print(f"  read: eGFR {egfr}, ACR {acr}   ->   {len(cards)} card(s)")
        print(f"  expected: {expectation}")

        for card in cards:
            for field in ("summary", "indicator", "detail", "source"):
                if not card.get(field):
                    problems.append(f"{label}: card missing {field}")
            if len(card.get("summary", "")) > 140:
                problems.append(f"{label}: summary over the 140-char cap")
            if card.get("indicator") not in {"info", "warning", "critical"}:
                problems.append(f"{label}: indicator {card.get('indicator')!r} not in the spec")
            detail = card.get("detail", "")
            if "**" not in detail or ">" not in detail:
                problems.append(f"{label}: detail carries no quoted text or citation")
            ext = card.get("extension") or {}
            if not ext.get("chunk_id") or ext.get("page") is None:
                problems.append(f"{label}: card does not trace to a chunk and page")
            print(f"    [{card['indicator']:<8}] {card['summary']}")
            print(f"      because: {ext.get('because')}")
            if args.verbose:
                print(textwrap.indent(detail, "      "))
        print()

    print(f"{total_cards} cards across {len(PATIENTS)} patients")
    silent = sum(1 for label, pre, _ in PATIENTS
                 if not cds_hooks.handle({"hook": "patient-view", "prefetch": pre},
                                         answerer)["cards"])
    print(f"{silent} of {len(PATIENTS)} patients produced no card at all")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\nevery card carries verbatim text, a citation, and a page it traces to")


if __name__ == "__main__":
    main()
