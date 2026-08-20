"""Answer-level correctness: does the answer actually say what the guideline says?

The evaluation chain was measured at both ends and nowhere in the middle. nDCG says
the right chunk was retrieved; citation accuracy says the citation resolves to real
text on the cited page. Neither asks whether the answer a clinician reads is correct.

This closes that gap without a human judge and without an LLM grader, both of which
would be unverifiable here. The gold set already pins the chunk a clinician would
cite, so the key values in that chunk - thresholds, bands, categories - are known.
A correct answer to a question about a threshold must state that threshold.

Three outcomes per case:

    answered_correct   every key value from the gold chunk appears in the answer
    answered_partial   some appear, some do not
    answered_missing   the answer states none of them

Cases whose gold chunk contains no extractable value are skipped rather than counted
as passes: scoring a question the method cannot judge would inflate the result.

Runs against the extractive answer by default, which needs no API key. With --generate
it measures the LLM answer instead, which is the number that matters once generation
is live.

Known limitation, measured rather than hidden: KDIGO prints reference numbers inline,
so a run like "127-142" can be read as a numeric range and become an expected value.
That is 1 of 107 expectations (0.9%). The obvious filter - drop ranges whose ends are
both three digits - was tested and rejected: it also removes NICE's real blood
pressure targets, "120 to 139" and "120 to 129", trading one false expectation for
four lost real ones. The metric therefore understates correctness very slightly, in
the safe direction.

Usage:
    python scripts/check_answers.py
    python scripts/check_answers.py --generate --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.paths import CORPUS, GOLD_SET, REPORTS  # noqa: E402

# The value classes a CKD answer turns on. Bare integers are excluded for the same
# reason the answer verifier excludes them: "stage 3" and "3 months" are prose.
KEY_VALUE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg\s?/\s?mmol|mg\s?/\s?g|ml\s?/\s?min(?:\s?/\s?1\.73\s?m2?)?|"
    r"mmol\s?/\s?l|mm\s?Hg|g\s?/\s?kg|%)"
    r"|\bG[1-5][ab]?\b|\bA[1-3]\b"
    r"|(?<![\d.])\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?(?!\.?\d)",
    re.I,
)


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def key_values(text: str) -> list[str]:
    """Distinct clinically load-bearing values, normalised for comparison."""
    seen: dict[str, str] = {}
    for match in KEY_VALUE.finditer(text):
        seen.setdefault(normalise(match.group(0)), match.group(0).strip())
    return list(seen.values())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Measure the LLM answer.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dense-model", default="abhinand/MedEmbed-large-v0.1")
    parser.add_argument("--dense-file", default="data/indexes/dense/abhinand__MedEmbed-large-v0_1.npy")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--out", default=str(REPORTS / "answer_correctness.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from newbieduo.generation.answer import build_answer
    from newbieduo.retrieval import retrieve as retrieve_module

    retrieve_module.set_dense_path(args.dense_file)

    corpus = {record["id"]: record for record in read_jsonl(CORPUS)}
    cases = [c for c in read_jsonl(GOLD_SET) if not c["should_abstain"]]
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict] = []
    counts = {"correct": 0, "partial": 0, "missing": 0, "skipped": 0, "not_answered": 0}

    print(f"{len(cases)} answerable questions | target: "
          f"{'generated answer' if args.generate else 'extractive answer'}\n")

    for case in cases:
        # Primary evidence only: a grade-2 chunk is the one a clinician would cite.
        primary = [cid for cid, grade in (case.get("relevance") or {}).items() if grade == 2]
        expected: list[str] = []
        for chunk_id in primary:
            record = corpus.get(chunk_id)
            if record:
                expected.extend(key_values(" ".join(record["raw_text"].split())))
        expected = list(dict.fromkeys(expected))

        if not expected:
            counts["skipped"] += 1
            continue

        result = build_answer(case["query"], 10, args.dense_model, args.reranker_model,
                              generate=args.generate)
        if result.get("status") != "answered":
            counts["not_answered"] += 1
            rows.append({"id": case["id"], "outcome": "not_answered",
                         "status": result.get("status")})
            continue

        if args.generate and (result.get("generated") or {}).get("text"):
            answer = result["generated"]["text"]
            source = "generated"
        else:
            answer = " ".join(q.get("text", "") for q in (result.get("quotes") or []))
            source = "extractive"

        found = [v for v in expected if normalise(v) in normalise(answer)]
        if len(found) == len(expected):
            outcome = "correct"
        elif found:
            outcome = "partial"
        else:
            outcome = "missing"
        counts[outcome] += 1

        rows.append({
            "id": case["id"], "kind": case.get("kind"), "category": case.get("category"),
            "outcome": outcome, "source": source,
            "expected": expected, "found": found,
            "absent": [v for v in expected if v not in found],
            # The answer itself, so a failure can be judged rather than only counted.
            # Without it "missing" is ambiguous between a wrong answer and a right one
            # phrased differently - "95 is normal or high" is correct but never says
            # "G1", and this metric cannot tell those apart on its own.
            "answer": " ".join(answer.split())[:400],
        })
        if args.verbose or outcome == "missing":
            print(f"  {outcome:<8} {case['id']:<26} expected {expected[:4]}"
                  f"{' found none' if outcome == 'missing' else ''}")

    scored = counts["correct"] + counts["partial"] + counts["missing"]
    print()
    print(f"scored          {scored}")
    for name in ("correct", "partial", "missing"):
        share = counts[name] / scored if scored else 0.0
        print(f"  {name:<13} {counts[name]:>3}  ({share:.0%})")
    print(f"  not answered  {counts['not_answered']:>3}")
    print(f"  skipped       {counts['skipped']:>3}  (gold chunk states no extractable value)")

    # Value-level recall is the stricter reading: of every threshold the gold evidence
    # states, how many reach the reader?
    total_expected = sum(len(r.get("expected", [])) for r in rows if "expected" in r)
    total_found = sum(len(r.get("found", [])) for r in rows if "found" in r)
    if total_expected:
        print(f"\nkey values reaching the reader: {total_found}/{total_expected} "
              f"({total_found / total_expected:.1%})")

    Path(args.out).write_text(json.dumps({
        "target": "generated" if args.generate else "extractive",
        "counts": counts,
        "value_recall": round(total_found / total_expected, 4) if total_expected else None,
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
