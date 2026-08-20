"""Adversarial test of the generation layer's verifier.

The strict grounding prompt in generate.py is a *request* to the model. This file
tests the part that is a *guarantee*: the mechanical check that runs afterwards and
does not depend on the model's cooperation. It does so without calling any model,
by feeding the verifier the answers a model plausibly could produce - including the
ones that are fluent, confident and wrong.

The property under test is one-directional and clinical: a threshold the evidence
does not contain must never reach the user. A false rejection costs an LLM-phrased
answer and falls back to extraction. A false acceptance is a fabricated clinical
number presented with a citation, which is the worst output this system can produce.

Evidence fixtures are pulled from the live corpus by citation anchor rather than
pasted, so the test fails loudly if the underlying chunks change rather than
silently testing text the system no longer serves.

Usage:
    python scripts/check_generation.py
    python scripts/check_generation.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS = ROOT / "data" / "chunks_v2" / "retrieval_corpus.jsonl"

# (anchor substring in citation, substring that must appear in the chunk text)
FIXTURES = [
    ("KDIGO 2024, Recommendation 3.4.1", "120 mm Hg"),
    ("NICE NG203, 1.6.1", "140 mmHg"),
    ("NICE NG203, 1.6.2", "70 mg/mmol"),
    ("KDIGO 2024, Table 2", "30 - 44"),
    ("KDIGO 2024, Table 2", "15 - 29"),
]


def load_fixtures() -> list[dict]:
    """The three chunks that state the blood-pressure targets, straight from the corpus."""
    with CORPUS.open("r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    evidence = []
    for anchor, required in FIXTURES:
        matches = [
            c for c in corpus
            if (c["metadata"].get("citation") or "").startswith(anchor)
            and required.replace(" ", "") in " ".join(c["raw_text"].split()).replace(" ", "")
        ]
        if not matches:
            raise SystemExit(
                f"fixture missing: no chunk cited '{anchor}' containing '{required}'.\n"
                "The corpus changed - update FIXTURES rather than weakening the test."
            )
        evidence.append({
            "citation": matches[0]["metadata"]["citation"],
            "text": " ".join(matches[0]["raw_text"].split()),
        })
    return evidence


# Cases where the question itself supplies a number. Format:
#   (label, question, model output, must_pass, what it probes)
# These exist because the verifier rejected a *correct* answer in the demo: asked
# about a filtration rate of 38, the model classified it as G3b citing the row
# "G3b ... 30 - 44", and was discarded for "states '38 ml/min' which is not in the
# cited evidence". It was not in the evidence - it was in the question.
QUESTION_CASES: list[tuple[str, str, str, bool, str]] = [
    (
        "patient value, in stated band",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        "An eGFR of 38 ml/min falls in GFR category G3b, which spans 30 - 44 [4].",
        True,
        "the clinician supplied 38 and the cited row brackets it - a classification",
    ),
    (
        "patient value, outside stated band",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        "An eGFR of 38 ml/min falls in category G4, which spans 15 - 29 [4].",
        False,
        "number came from the question but the cited range does not contain it",
    ),
    (
        "leading question smuggling a target",
        "Should I aim for a systolic blood pressure of 132 mmHg?",
        "Yes - aim for a systolic blood pressure of 132 mmHg [2].",
        False,
        "132 is inside NICE's 120-139 range, but recommending it is not what NICE says; "
        "in-range alone must not license a recommendation",
    ),
    # The three phrasings a recommending-verb denylist accepted. "consider" and "offer"
    # are the commonest verbs in NICE guidance, so this was not a corner case.
    (
        "smuggled target, 'consider'",
        "Should I consider a systolic blood pressure of 132 mmHg?",
        "Consider a systolic blood pressure of 132 mmHg [2].",
        False,
        "denylist of recommending verbs failed open on the NICE house verb",
    ),
    (
        "smuggled target, 'offer'",
        "Should I offer a systolic blood pressure goal of 132 mmHg?",
        "Offer a systolic blood pressure goal of 132 mmHg [2].",
        False,
        "same failure, second NICE house verb",
    ),
    (
        "smuggled target, bare assertion",
        "Is a systolic blood pressure of 132 mmHg appropriate?",
        "A systolic blood pressure of 132 mmHg is appropriate [2].",
        False,
        "no verb at all - a denylist cannot see this, an allowlist rejects it",
    ),
    (
        "live phrasing the model actually produced",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        "A filtration rate of 38 ml/min per 1.73 m2 puts the patient in GFR category G3b [4].",
        True,
        "the wording gemini-flash-lite returned live; the allowlist must admit it",
    ),
]

# (label, model output, must_pass, what the case is probing)
#
# Evidence is [1] KDIGO <120 mm Hg, [2] NICE <140 mmHg (ACR under 70),
#             [3] NICE <130 mmHg (ACR 70 mg/mmol or more).
CASES: list[tuple[str, str, bool, str]] = [
    (
        "faithful answer",
        "KDIGO suggests a target systolic blood pressure of < 120 mm Hg when tolerated [1]. "
        "NICE instead advises a clinic systolic below 140 mmHg for an ACR under 70 mg/mmol [2].",
        True,
        "correct numbers, correct citations - the baseline that must not be rejected",
    ),
    (
        "spacing variant",
        "The KDIGO target is <120mmHg using standardized office measurement [1].",
        True,
        "'120mmHg' must match evidence written '120 mm Hg' - else every real answer fails",
    ),
    (
        "hallucinated threshold",
        "Adults with CKD should be treated to a target below 130 mm Hg systolic [1].",
        False,
        "a number the cited evidence does not contain - the core failure to catch",
    ),
    (
        "silent unit conversion",
        "Treatment intensifies above an ACR of 700 mg/g [3].",
        False,
        "70 mg/mmol converted to mg/g; arithmetically right, not in the evidence",
    ),
    (
        "split-the-difference",
        "A reasonable compromise target is below 135 mmHg [1][2].",
        False,
        "averaging two real thresholds into a third that no guideline states",
    ),
    (
        "borrowed from uncited source",
        "Aim for a clinic systolic below 130 mmHg [1].",
        False,
        "130 mmHg exists in evidence [3] but the sentence cites [1] - citation must support",
    ),
    (
        "citation out of range",
        "The systolic target is < 120 mm Hg [9].",
        False,
        "index beyond the evidence list, the classic invented-reference failure",
    ),
    (
        "no citation at all",
        "Adults with CKD and high blood pressure should have their systolic pressure lowered.",
        False,
        "guidance with no evidence pointer",
    ),
    (
        "faithful band",
        "For an ACR under 70 mg/mmol the target range is 120 to 139 mmHg [2].",
        True,
        "a range copied correctly from the evidence must survive the range check",
    ),
    (
        "wrong band",
        "That eGFR places the patient in category G3b, which spans 30 to 49 [1].",
        False,
        "unitless band, invented; G3b is 30-44 and the unit rule cannot see it",
    ),
    (
        "recommendation numbers are not a range",
        "This is covered by recommendations 1.6.1 to 1.6.2, which set a target of "
        "< 120 mm Hg [1].",
        True,
        "dotted identifiers must not be parsed as the numeric range 1-1",
    ),
    (
        "range endpoint carrying a unit",
        "GFR category G4 corresponds to a GFR of 15 - 29 ml/min per 1.73 m2 [5].",
        True,
        "the live false positive: '29 ml/min' appears verbatim nowhere because the "
        "source writes '15 - 29' with the unit in a header, but the range is correct",
    ),
    (
        "range endpoint, wrong band",
        "GFR category G4 corresponds to a GFR of 15 - 29 ml/min per 1.73 m2 [4].",
        False,
        "same sentence cited to the G3b row - the range check must still catch it",
    ),
    (
        "diastolic hallucination",
        "NICE advises a clinic diastolic blood pressure below 85 mmHg [2].",
        False,
        "plausible-looking diastolic value; evidence says 90 mmHg",
    ),
]

# Cases the verifier is known NOT to catch. Listed as expectations so the gap is
# recorded and measured rather than discovered by a judge.
#
# The unitless-band gap that this file originally documented here is now closed by
# NUMERIC_RANGE in generate.py and tested above as "wrong band".
KNOWN_GAPS: list[tuple[str, str, str]] = [
    (
        "single unitless value",
        "Refer when the eGFR falls below 25 [1].",
        "One bare number with no unit and no range is still unguarded, and cannot be "
        "guarded without false-alarming on 'stage 3' or 'within 3 months'. The residual "
        "exposure is a fabricated unitless single threshold; ranges and any value with a "
        "unit are covered.",
    ),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from newbieduo.generation.generate import generate, verify

    evidence = load_fixtures()
    print(f"evidence fixtures ({len(evidence)}), pulled from the live corpus:")
    for index, item in enumerate(evidence, start=1):
        print(f"  [{index}] {item['citation']}")
    print()

    failures: list[str] = []
    print(f"{'case':<28} {'expected':<9} {'got':<9}  detail")
    for label, answer, must_pass, probing in CASES:
        result = verify(answer, evidence)
        good = result.ok == must_pass
        expected = "accept" if must_pass else "reject"
        got = "accept" if result.ok else "reject"
        mark = "ok   " if good else "FAIL "
        detail = "; ".join(result.problems)[:64] if result.problems else probing[:64]
        print(f"{mark}{label:<23} {expected:<9} {got:<9}  {detail}")
        if not good:
            failures.append(f"{label}: expected {expected}, got {got} ({probing})")
        if args.verbose:
            print(f"      {answer[:100]}")

    # ------------------------------------ context must not count as supporting
    # Quoted chunks are shown to the model with their surrounding text, so a short
    # atom can be understood. That widening must not widen what counts as evidence:
    # if a number in the surroundings could support a claim, the guarantee is only as
    # tight as the largest context window rather than as the cited chunk.
    print("\nsurrounding context is readable but not citable")
    with_context = [
        dict(evidence[0], context="The systolic target in the surrounding passage is 105 mmHg."),
        *evidence[1:],
    ]
    leak = verify("The systolic blood pressure target is 105 mmHg [1].", with_context)
    ok = not leak.ok
    print(f"{'ok   ' if ok else 'FAIL '}{'number only in context':<36} "
          f"{'reject':<7} {'reject' if not leak.ok else 'accept'}  "
          f"{'; '.join(leak.problems)[:56]}")
    if not ok:
        failures.append("a value present only in the surrounding context was accepted as supported")

    # ------------------------------------------- numbers supplied by the question
    print("\nvalues the question supplied (classification vs recommendation)")
    for label, question, answer, must_pass, probing in QUESTION_CASES:
        result = verify(answer, evidence, question)
        good = result.ok == must_pass
        expected = "accept" if must_pass else "reject"
        got = "accept" if result.ok else "reject"
        detail = "; ".join(result.problems)[:60] if result.problems else probing[:60]
        print(f"{'ok   ' if good else 'FAIL '}{label:<36} {expected:<7} {got:<7}  {detail}")
        if not good:
            failures.append(f"{label}: expected {expected}, got {got} ({probing})")

    # --------------------------------------------------------- end-to-end path
    # The verifier is only useful if a failed verification actually suppresses the
    # text. Stub the provider so the whole generate() path runs with no API key.
    print("\nend-to-end suppression (stubbed model, no API call)")
    from newbieduo.generation import generate as G

    for label, fake_output, expect_status in [
        ("model returns good answer", CASES[0][1], "verified"),
        ("model hallucinates", CASES[2][1], "failed_verification"),
        ("model refuses", "INSUFFICIENT_EVIDENCE", "model_refused"),
    ]:
        original = G.call_gemini
        G.call_gemini = lambda prompt, model, max_tokens=400, _o=fake_output: _o
        try:
            result = generate("What blood pressure target applies in CKD?", evidence,
                              model="stub", provider="gemini")
        finally:
            G.call_gemini = original

        status_ok = result.get("status") == expect_status
        # The property that matters: nothing unverified is ever placed in `text`.
        leaked = result.get("status") == "failed_verification" and result.get("text")
        if not status_ok or leaked:
            failures.append(f"{label}: status={result.get('status')}, text={result.get('text')!r}")
            print(f"FAIL  {label:<30} status={result.get('status')}")
        else:
            shown = "suppressed" if result.get("text") is None else "shown"
            print(f"ok    {label:<30} {result['status']:<20} text {shown}")

    # ------------------------------------------------------------- known gaps
    print("\nknown gaps (documented, not failures)")
    for label, answer, why in KNOWN_GAPS:
        result = verify(answer, evidence)
        state = "not caught" if result.ok else "caught after all"
        print(f"  {label:<20} {state}")
        print(f"      {why}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    # +1 for the context-leak check, which is run inline rather than from a table.
    print(f"all {len(CASES) + len(QUESTION_CASES) + 1} verifier cases and 3 end-to-end cases passed")
    print("no fabricated clinical value reached `text` in any case")


if __name__ == "__main__":
    main()
