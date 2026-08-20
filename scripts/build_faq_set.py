"""A benchmark of the questions clinicians actually ask most, sourced externally.

Where the questions come from
-----------------------------
Not from the corpus, and not from anything this system returned. They are the topics
that recur in the primary-care CKD literature on what clinicians ask and get wrong:
when to refer, how often to monitor, what blood pressure to aim for, who gets an SGLT2
inhibitor or a statin, what to do about NSAIDs and sick days, and what actually counts
as CKD in the first place.

  - GPs' views on managing advanced CKD in primary care, BJGP 65(636)
    https://bjgp.org/content/65/636/e469
  - Clinician agreement on early-stage CKD monitoring in primary care, PMC4800136
  - Referral and management options for patients with CKD, PMC5060784
  - UK Kidney Association, Management of patients with CKD
  - KDOQI US Commentary on the KDIGO 2024 CKD Guideline, AJKD

Why this exists alongside two other sets
----------------------------------------
The 67-question gold set was written question-first by someone reading the corpus, and
its blind spots are the ones that reading produces. The 46-question held-out set was
written chunk-first, which fixes the answer key but makes the questions a function of
whichever chunks were sampled. Neither is anchored to what a clinician is *likely* to
ask, and that is the thing a decision-support tool is actually judged on in use.

It is not independent evidence. Referral, monitoring, blood pressure, staging and
medication safety are both the commonest clinical questions and the main categories of
the existing gold set, so the overlap is large by construction. Read it as "how does the
system do on the questions that come up most", not as a fresh sample.

How the gold is pinned
----------------------
Every question here has an answer the guidelines state outright - that is the selection
criterion, and it is about the guideline, not about the retriever. A question whose
answer is a matter of interpretation, or where KDIGO and NICE genuinely disagree without
the question naming one of them, is not in this file.

The pinning itself is a regex sweep of the whole corpus for chunks that state that
answer, exactly as `reannotate_gold.py` does. No chunk ID is written by hand, nothing is
chosen by looking at what retrieval returned, and the file is frozen before the first
evaluation runs.

Usage:
    python scripts/build_faq_set.py            # writes eval/faq_clinical.jsonl
    python scripts/build_faq_set.py --show     # print what each pattern pins
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newbieduo.paths import CORPUS  # noqa: E402

OUT = ROOT / "eval" / "faq_clinical.jsonl"

# (id, question, topic, answer-pattern, optional source restriction)
# The pattern matches a chunk that STATES the answer. Kept tight on purpose: a loose
# pattern hands out free credit, which is the failure mode the re-annotation work was
# built to avoid.
FAQ: list[tuple[str, str, str, str, str]] = [
    ("faq_01_definition", "What counts as chronic kidney disease?", "definition",
     r"CKD is defined as abnormalities of kidney structure or function|"
     r"Abnormalities of kidney function or structure present for more than 3 months|"
     r"Criteria for chronic kidney disease \(either of the following", ""),

    ("faq_02_chronicity", "How long do the abnormalities have to persist before it is chronic?",
     "definition", r"explicitly yet arbitrarily define the duration|"
     r"Criteria for chronic kidney disease \(either of the following", ""),

    ("faq_03_who_to_test", "Which adults should be tested for CKD?", "testing",
     r"Monitor for (the development or progression of )?CKD|"
     r"Offer testing for CKD (using|with)|"
     r"people with (any of )?the following risk factors.{0,60}(test|offer)", "nice"),

    ("faq_04_confirm_egfr", "Do I need to repeat an abnormal eGFR before diagnosing CKD?",
     "testing", r"Confirm an eGFR result of less than 60|"
     r"Do not assume chronicity based upon a single abnormal", ""),

    ("faq_05_staging_g3b", "What eGFR range is GFR category G3b?", "staging",
     r"Row \d+:\s*(GFR category:\s*)?G3b\b", ""),

    ("faq_06_kidney_failure", "At what eGFR is someone in kidney failure?", "staging",
     r"Row \d+:\s*(GFR category:\s*)?G5\b|"
     r"eGFR of less than 15 ml/min/1\.73 m2 \(GFR category G5\) is referred to as kidney failure", ""),

    ("faq_07_acr_important", "What ACR counts as clinically important proteinuria?", "testing",
     r"Regard a confirmed ACR of 3 mg/mmol or more as clinically important proteinuria", "nice"),

    ("faq_08_refer", "When should I refer an adult with CKD to specialist kidney care?",
     "referral", r"Refer adults with CKD for specialist assessment|"
     r"Refer adults with CKD to specialist kidney care services", ""),

    ("faq_09_kfre", "At what 5-year risk of kidney failure should I refer?", "referral",
     r"5-year risk of needing renal replacement therapy.{0,40}(greater than|>)\s*5%|"
     r">\s*3%\s*-\s*5%\s*5-year risk", ""),

    ("faq_10_monitor_freq", "How often should eGFR be checked in someone with CKD?",
     "monitoring", r"Use table 2 to guide the minimum frequency of eGFR|"
     r"agree the frequency of monitoring", "nice"),

    ("faq_11_acr_rise", "How much does ACR have to rise before it means something?",
     "monitoring", r"a doubling of the ACR on a subsequent test exceeds laboratory variability", ""),

    ("faq_12_progression", "How is accelerated progression of CKD defined?", "monitoring",
     r"sustained decrease in GFR of 25% or more and a change in GFR category|"
     r"sustained decrease in eGFR of 15 ml/min/1\.73 m2", "nice"),

    ("faq_13_bp_low_acr", "What blood pressure should I aim for in CKD with an ACR under 70?",
     "blood_pressure", r"ACR under 70 mg/mmol, aim for a clinic systolic blood pressure below 140", "nice"),

    ("faq_14_bp_high_acr", "What blood pressure target applies when ACR is 70 or above?",
     "blood_pressure", r"ACR of 70 mg/mmol or more, aim for a clinic systolic blood pressure below 130", "nice"),

    ("faq_15_sglt2", "Who with CKD should be offered an SGLT2 inhibitor?", "treatment",
     r"We recommend treating adults with CKD with an SGLT2i for the following", ""),

    ("faq_16_statin", "Should adults with CKD be on a statin?", "treatment",
     r"we recommend treatment with a statin|Offer atorvastatin 20 mg for the primary", ""),

    ("faq_17_nsaid", "Are NSAIDs safe in someone with CKD?", "medication_safety",
     r"chronic use of NSAIDs may be associated with progression|"
     r"People with CKD may be more susceptible to the nephrotoxic effects", ""),

    ("faq_18_sick_day", "Which medicines should be held during an acute dehydrating illness?",
     "medication_safety", r"Sick day rules have been endorsed|"
     r"temporarily stop the following medications", ""),

    ("faq_19_rasi_potassium", "At what potassium do I stop an ACE inhibitor or ARB?",
     "medication_safety",
     r"Stop renin-angiotensin system antagonists in adults if the serum potassium", "nice"),

    # Both guidelines state this, so the source restriction that was here was wrong -
    # it excluded KDIGO's practice point and the NICE regex did not match either.
    ("faq_20_rasi_check", "What should I check after starting an ACE inhibitor in CKD?",
     "medication_safety",
     r"Changes in BP, serum creatinine, and serum potassium should be checked within|"
     r"Measure serum potassium concentrations and estimate the GFR before starting", ""),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Print what each pattern pins.")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    with CORPUS.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    cases, unpinned = [], []
    for case_id, query, topic, pattern, source in FAQ:
        pinned = {}
        for record in records:
            if source and source not in record["id"]:
                continue
            text = " ".join(record["raw_text"].split())
            if re.search(pattern, text, re.I):
                pinned[record["id"]] = 2
        if not pinned:
            unpinned.append((case_id, query))
            continue
        cases.append({
            "id": case_id, "query": query, "kind": "faq_clinical",
            "category": topic, "slice": "faq", "should_abstain": False,
            "relevance": pinned,
            "note": "externally sourced clinical FAQ; gold pinned by corpus sweep",
        })
        if args.show:
            print(f"\n{case_id}  ({len(pinned)} pinned)\n  Q: {query}")
            for chunk_id in list(pinned)[:3]:
                text = next(" ".join(r["raw_text"].split())
                            for r in records if r["id"] == chunk_id)
                print(f"    {textwrap.shorten(text, 104)}")

    print(f"\n{len(cases)} questions pinned, {len(unpinned)} could not be pinned")
    for case_id, query in unpinned:
        print(f"  UNPINNED {case_id}: {query}")
    if unpinned:
        print("  (an unpinned question is dropped, not loosened - a pattern widened until "
              "it matches\n   something is how free credit gets in)")

    counts = sorted(len(c["relevance"]) for c in cases)
    print(f"pins per question: min {counts[0]}, median {counts[len(counts)//2]}, max {counts[-1]}")

    with Path(args.out).open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
