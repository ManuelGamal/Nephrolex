"""Re-annotate the gold set by concept rule, blind to what the retriever returns.

Why
---
The gold set pins *the* answer to each question - usually one to three chunks. Anything
else that answers the question scores zero, so a correct retrieval is marked wrong.
Three of the seven total misses on the answerable set were this, not retrieval failure:
`definition_paraphrase_2` returns KDIGO section 1.1.3, "we explicitly yet arbitrarily
define the duration of a minimum of 3 months", and scores 0.000.

The annotation is also internally inconsistent. The corpus prints the CKD staging table
twice (p22 Table 2 and the p11 reprint). `gfr_g3b`, `gfr_g4` and `gfr_g5` pin both
printings; `stag_g1_direct`, `stag_g2_direct` and their paraphrases pin only one. Same
concept, same corpus, different answer key - so per-question scores are not comparable.

How this stays honest
---------------------
The obvious fix - look at what the system returned and pin the good ones - fits the
answer key to the system and inflates every number measured afterwards. Instead:

1. Rules are written against the *clinical concept a question asks about*, read from
   the question text. No rule mentions a chunk ID.
2. Candidates come from a regex sweep of the entire corpus. A chunk the retriever never
   returns is found by exactly the same sweep as one it ranks first, so this can lower
   a score as easily as raise it.
3. Scope is a concept, not a question. Every question asking a concept is re-annotated
   together, including ones already scoring 1.000.
4. Nothing is removed. Existing grades win where they disagree, so this can only add.

Each addition carries its rule name, so any number in the writeup can be traced to the
rule that produced it and challenged on the guideline text.

Usage:
    python scripts/reannotate_gold.py                 # show the proposal, change nothing
    python scripts/reannotate_gold.py --apply         # rewrite the gold set (backs up)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nephrolex.paths import CORPUS, GOLD_SET, REPORTS  # noqa: E402


@dataclass(frozen=True)
class Rule:
    """One clinical concept, the questions that ask it, and what answers it."""

    name: str
    rationale: str
    cases: tuple[str, ...]
    # A chunk matching `primary` states the answer outright -> grade 2.
    primary: str
    # A chunk matching `context` is useful but does not answer on its own -> grade 1.
    context: str = ""
    # Restricts to one guideline when the question names one ("How does NICE define...").
    source: str = ""
    # Excludes chunks that match the regex but discuss a different thing.
    exclude: str = ""
    per_case_primary: dict[str, str] = field(default_factory=dict)
    # Matched against every *question* in the gold set. Any question it matches that is
    # not in `cases` is reported as a possible coverage gap. This exists because the
    # first version of this file re-annotated the G1 and G2 questions and left the G3b,
    # G4, G5 and staging_paraphrase questions - which ask the same thing - on the old
    # annotation, replacing one inconsistency with another. Uniformity claimed is not
    # uniformity checked.
    asks: str = ""


RULES: list[Rule] = [
    # ---------------------------------------------------------------- staging
    # "What range is category X?" is answered by any chunk stating that row of the
    # staging table, in either printing. The existing annotation already does this for
    # G3b, G4 and G5; these rules make it uniform for G1 and G2.
    # `primary` matches a table row stating the asked category, in either printing:
    #   p22  "Row 1: GFR category: G1; ... ; Terms: Normal or high"
    #   p11  "Row 4: G1; Normal or high; >=90"
    # `context` is deliberately narrow. An earlier draft used "GFR categories in CKD",
    # which is the header carried by *every* row of Table 2, so a question about G1
    # pinned the G4 row as partially relevant. A sibling row does not help answer
    # "is 95 normal?"; only chunks presenting the whole scheme do.
    # One rule for every question that asks which band a GFR value or label falls in,
    # parameterised by the category each one asks about. Splitting this into a G1 rule
    # and a G2 rule was itself a defect: it gave the G1 and G2 questions the whole-scheme
    # chunks as partial credit while gfr_g3b, gfr_g4, gfr_g5 and staging_paraphrase ask
    # the same concept and did not get them - the identical inconsistency this rule was
    # written to remove, pointing the other way. A completeness check over every question
    # asking the concept is what caught it.
    Rule(
        name="gfr_category_row",
        rationale="A question asking which band a GFR value or label falls in is answered by "
                  "any chunk stating that row, in either printing of the staging table. "
                  "Chunks presenting the whole scheme are partial credit.",
        cases=("stag_g1_direct", "stag_g1_paraphrase", "stag_g2_direct", "stag_g2_paraphrase",
               "gfr_g3b", "gfr_g4", "gfr_g5", "staging_paraphrase"),
        primary=r"Row \d+:\s*(GFR category:\s*)?G1\b",  # overridden per case below
        per_case_primary={
            "stag_g1_direct": r"Row \d+:\s*(GFR category:\s*)?G1\b",
            "stag_g1_paraphrase": r"Row \d+:\s*(GFR category:\s*)?G1\b",
            "stag_g2_direct": r"Row \d+:\s*(GFR category:\s*)?G2\b",
            "stag_g2_paraphrase": r"Row \d+:\s*(GFR category:\s*)?G2\b",
            # 38 ml/min falls in 30-44, which is G3b.
            "staging_paraphrase": r"Row \d+:\s*(GFR category:\s*)?G3b\b",
            "gfr_g3b": r"Row \d+:\s*(GFR category:\s*)?G3b\b",
            "gfr_g4": r"Row \d+:\s*(GFR category:\s*)?G4\b",
            "gfr_g5": r"Row \d+:\s*(GFR category:\s*)?G5\b",
        },
        context=r"G1 to G5|Figure 13: Albuminuria categories",
    ),
    # ------------------------------------------------------------- definition
    # The 3-month duration criterion. Questions naming a guideline stay inside it.
    # No `context` regex on these. An earlier draft used "chronicity|abnormalities of
    # kidney structure or function" as partial credit and pulled in potassium
    # physiology, a risk-model validation passage and a table of tests for identifying
    # cause - 13 additions to a question with 2. Partial credit that loose inflates
    # recall for free, which is exactly the failure this whole exercise exists to
    # avoid, so every addition below has to match a targeted pattern.
    Rule(
        name="ckd_definition_statement",
        rationale="The formal definition: abnormalities of kidney structure or function "
                  "present for a minimum of 3 months, with implications for health.",
        cases=("kdigo_definition", "def_paraphrase", "definition_paraphrase_2"),
        source="kdigo",
        primary=r"CKD is defined as abnormalities of kidney structure or function|"
                r"Criteria for chronic kidney disease \(either of the following",
        asks=r"formal definition of CKD|define chronic kidney disease|count as having",
    ),
    Rule(
        name="ckd_chronicity_duration",
        rationale="'How long do the problems have to have been going on' is answered by the "
                  "passage that sets the duration, and by the practice point on proving it.",
        cases=("definition_paraphrase_2", "def_paraphrase"),
        source="kdigo",
        primary=r"explicitly yet arbitrarily define the duration",
        context=r"Proof of chronicity \(duration of a minimum of 3 months\)",
    ),
    Rule(
        name="ckd_definition_nice",
        rationale="NICE's own definition of CKD, for the question that names NICE.",
        cases=("def_nice_direct",),
        source="nice",
        primary=r"Abnormalities of kidney function or structure present for more than 3 months",
    ),
    Rule(
        name="ckd_chronicity_not_single_test",
        rationale="'rather than just one bad blood test' is answered directly by the practice "
                  "point telling clinicians not to assume chronicity from a single abnormal value.",
        cases=("def_paraphrase",),
        primary=r"Do not assume chronicity based upon a single abnormal",
    ),
    # ------------------------------------------------------- blood pressure
    # The question does not name a guideline, so both guidelines' targets answer it.
    # This is the KDIGO (<120) / NICE (<140, <130) divergence the system exists to show;
    # scoring only KDIGO penalises the system for surfacing the other one.
    Rule(
        name="systolic_target",
        rationale="An unqualified 'how low should systolic go in CKD?' is answered by any "
                  "guideline's systolic target recommendation, KDIGO or NICE.",
        cases=("bp_target_paraphrase",),
        primary=r"(target systolic blood pressure \(SBP\) of\s*<|"
                r"aim for a clinic systolic blood pressure below)",
        # Deliberately narrow. bp_kdigo names KDIGO, bp_target_low_acr and
        # bp_target_high_acr each name an ACR threshold, and bp_conflict asks about the
        # disagreement itself - all four are scoped questions whose answer key should
        # stay scoped. Only the unqualified phrasing takes both guidelines' targets.
        asks=r"how far down|how low should",
        # NICE 1.6.3 states a systolic target for children and young people. The gold
        # set scopes to adults and the system has a paediatric scope gate, so pinning
        # it would credit an answer the system is built to decline.
        exclude=r"children and young people",
    ),
]


def load_corpus() -> list[dict]:
    with CORPUS.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def matches(rule: Rule, case_id: str, chunk_id: str, text: str) -> int:
    """Grade this rule assigns to this chunk: 2, 1 or 0."""
    if rule.source and rule.source not in chunk_id:
        return 0
    if rule.exclude and re.search(rule.exclude, text, re.I):
        return 0
    primary = rule.per_case_primary.get(case_id, rule.primary)
    if re.search(primary, text, re.I):
        return 2
    if rule.context and re.search(rule.context, text, re.I):
        return 1
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Rewrite the gold set.")
    parser.add_argument("--chars", type=int, default=118)
    parser.add_argument("--out", default=str(REPORTS / "gold_reannotation.json"))
    args = parser.parse_args()

    corpus = load_corpus()
    texts = {r["id"]: " ".join(r["raw_text"].split()) for r in corpus}

    with GOLD_SET.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    by_id = {c["id"]: c for c in cases}

    proposal: list[dict] = []

    for rule in RULES:
        print("=" * 78)
        print(f"{rule.name}")
        print(textwrap.fill(rule.rationale, 76, initial_indent="  ", subsequent_indent="  "))
        for case_id in rule.cases:
            case = by_id.get(case_id)
            if case is None:
                raise SystemExit(f"rule {rule.name} names unknown case {case_id!r}")
            existing = case["relevance"]
            found = {
                chunk_id: grade
                for chunk_id, text in texts.items()
                if (grade := matches(rule, case_id, chunk_id, text))
            }
            added = {k: v for k, v in found.items() if k not in existing}
            print(f"\n  {case_id}: pinned {len(existing)} -> {len(existing) + len(added)}"
                  f"  (+{len(added)})")
            for chunk_id, grade in sorted(added.items(), key=lambda kv: -kv[1]):
                print(f"    +grade {grade}  {chunk_id[:64]}")
                print(f"              {textwrap.shorten(texts[chunk_id], args.chars)}")
                proposal.append({"case": case_id, "chunk": chunk_id, "grade": grade,
                                 "rule": rule.name, "date": str(date.today())})
            for chunk_id, grade in found.items():
                if chunk_id in existing and existing[chunk_id] != grade:
                    print(f"    kept existing grade {existing[chunk_id]} over rule's {grade}"
                          f"  {chunk_id[:52]}")
        print()

    touched = {p["case"] for p in proposal}
    print("=" * 78)
    print(f"{len(proposal)} additions across {len(touched)} of {len(cases)} cases")

    # Coverage: does any question ask a concept a rule covers without being assigned it?
    gaps = 0
    for rule in RULES:
        if not rule.asks:
            continue
        for case in cases:
            if case["should_abstain"] or case["id"] in rule.cases:
                continue
            if re.search(rule.asks, case["query"], re.I):
                covered = [r.name for r in RULES if case["id"] in r.cases]
                gaps += 1
                print(f"\n  COVERAGE GAP  {rule.name} may apply to {case['id']} "
                      f"({len(case['relevance'])} pinned)")
                print(f"                {case['query'][:70]}")
                # A question can legitimately match one rule's phrasing while belonging
                # to a narrower one - "How does NICE define CKD?" reads like the general
                # definition rule but is answered only from NICE.
                print(f"                already covered by: "
                      f"{', '.join(covered) if covered else 'NOTHING - decide'}")
    print(f"\ncoverage check: {gaps} question(s) match a rule they are not assigned to")
    if gaps:
        print("Each is either a concept the rule should cover - assign it - or a question "
              "the rule only looks like it covers. Decide explicitly; silence here is "
              "how the last inconsistency got in.")

    Path(args.out).write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    if not args.apply:
        print("\nnothing written to the gold set. Re-run with --apply to commit.")
        return

    backup = GOLD_SET.with_suffix(".jsonl.bak")
    shutil.copy2(GOLD_SET, backup)
    for entry in proposal:
        relevance = by_id[entry["case"]]["relevance"]
        relevance.setdefault(entry["chunk"], entry["grade"])
    with GOLD_SET.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"\napplied. previous gold set backed up to {backup}")
    print("Every number measured before this point is on the old key and must be re-run.")


if __name__ == "__main__":
    main()
