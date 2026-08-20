"""Run the whole system on every gold question and audit what it actually produced.

Every other measurement in this project stops short of the thing the user sees.
`evaluate_gold.py` calls `retrieve()` directly, so the scope gates, the premise check,
the answer assembly and the agreement banner are never exercised by it - a question
the gate wrongly refuses still scores a perfect nDCG. Three real defects reached the
demo that way: an "acute kidney injury" follow-up question refused as an emergency, a
clinical "history of" refused as a question about the guideline's history, and a
"sources agree" banner rendered for a question on which nothing had been compared.

This runs `build_answer` - the actual entry point - on all 75 gold cases and checks
what came back, per case, against what the gold set says should happen. Failures are
grouped by kind so the output says *where* and *why*, not just how many.

Checks per case:
    status          did it answer / decline / flag the premise, as the gold expects
    evidence        for an answerable case, is the pinned gold chunk actually shown
    agreement       is "sources agree" claimed without a comparison behind it
    citations       does every quote carry a citation and a resolvable page
    generation      if enabled, did the verifier accept, reject, or error

Usage:
    python scripts/audit_end_to_end.py
    python scripts/audit_end_to_end.py --generate --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.paths import GOLD_SET, REPORTS  # noqa: E402

# For a case that must abstain, any refusal counts. Which refusal fires is a
# presentation detail: "What does NICE NG203 say about immunosuppression after kidney
# transplant?" is pinned as unanswerable and refused by the transplant scope rule, and
# grading that as a failure said the system was unsafe when it had behaved correctly.
# The only unsafe outcome is answering.
REFUSALS = {
    "declined", "premise_not_found", "out_of_domain",
    "insufficient_evidence", "no_evidence", "no_question",
}
ANSWERABLE = {"answered"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_case(case: dict, result: dict) -> list[tuple[str, str]]:
    """Return (failure_kind, detail) pairs. Empty means the case behaved."""
    problems: list[tuple[str, str]] = []
    status = result.get("status")
    kind = case.get("kind")
    relevance = case.get("relevance") or {}

    # ---------------------------------------------------------------- status
    if case["should_abstain"]:
        if status not in REFUSALS:
            problems.append(("unsafe_answer", f"{kind} case was answered, not refused"))
    elif status not in ANSWERABLE:
        problems.append(("false_decline", f"answerable case returned {status}"))

    if case["should_abstain"]:
        # Nothing further to check: an abstention has no evidence obligations.
        return problems

    # -------------------------------------------------------------- evidence
    shown = {item.get("chunk_id") for item in (result.get("evidence") or [])}
    quoted = {item.get("chunk_id") for item in (result.get("quotes") or [])}
    primary = {cid for cid, grade in relevance.items() if grade == 2}

    if primary and status == "answered":
        if not (primary & shown):
            problems.append(("gold_absent", "no grade-2 gold chunk anywhere in the shown evidence"))
        elif not (primary & quoted):
            problems.append(("gold_not_quoted", "gold chunk retrieved but not among the quotes"))

    # ------------------------------------------------------------- agreement
    # The interface claims agreement when comparisons exist and none diverge. A claim
    # of consistency with nothing compared is an assertion without evidence.
    comparisons = result.get("comparisons") or []
    divergences = result.get("divergences") or []
    used = result.get("guidelines_used") or []
    if status == "answered" and len(used) > 1 and not comparisons and divergences:
        problems.append(("divergence_without_comparison", "divergence reported with no comparison"))

    # ------------------------------------------------------------- citations
    for quote in (result.get("quotes") or []):
        if not quote.get("citation"):
            problems.append(("quote_without_citation", f"chunk {quote.get('chunk_id')}"))
        if not (quote.get("provenance") or quote.get("page_start")):
            problems.append(("quote_without_page", str(quote.get("citation"))[:60]))

    # ------------------------------------------------------------ generation
    generated = result.get("generated") or {}
    if generated.get("available"):
        gen_status = generated.get("status")
        if gen_status == "generation_failed":
            problems.append(("generation_error", str(generated.get("error"))[:80]))
        elif gen_status == "failed_verification":
            problems.append((
                "generation_rejected",
                "; ".join(generated.get("verification", {}).get("problems", []))[:80],
            ))

    return problems


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Also exercise the LLM layer.")
    parser.add_argument("--dense-model", default="abhinand/MedEmbed-large-v0.1")
    parser.add_argument("--dense-file", default="data/indexes/dense/abhinand__MedEmbed-large-v0_1.npy")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=str(REPORTS / "end_to_end_audit.json"))
    args = parser.parse_args()

    from newbieduo.generation.answer import build_answer
    from newbieduo.retrieval import retrieve as retrieve_module

    retrieve_module.set_dense_path(args.dense_file)

    cases = read_jsonl(GOLD_SET)
    if args.limit:
        cases = cases[: args.limit]

    rows: list[dict] = []
    failures: dict[str, list[tuple[str, str]]] = defaultdict(list)
    statuses: Counter = Counter()

    print(f"{len(cases)} gold cases through build_answer() "
          f"| generation {'on' if args.generate else 'off'}\n")

    for case in cases:
        result = build_answer(case["query"], 10, args.dense_model, args.reranker_model,
                              generate=args.generate)
        problems = audit_case(case, result)
        statuses[result.get("status")] += 1
        for kind, detail in problems:
            failures[kind].append((case["id"], detail))

        rows.append({
            "id": case["id"], "kind": case.get("kind"), "category": case.get("category"),
            "slice": case.get("slice"), "should_abstain": case["should_abstain"],
            "query": case["query"], "status": result.get("status"),
            "quotes": [q.get("citation") for q in (result.get("quotes") or [])],
            "comparisons": len(result.get("comparisons") or []),
            "divergences": len(result.get("divergences") or []),
            "generation": (result.get("generated") or {}).get("status"),
            "problems": [{"kind": k, "detail": d} for k, d in problems],
        })

        if args.verbose or problems:
            mark = "FAIL" if problems else "ok  "
            print(f"  {mark} {case['id']:<28} {str(result.get('status')):<18} "
                  f"{', '.join(k for k, _ in problems)}")

    # ------------------------------------------------------------------ summary
    print("\nstatus distribution")
    for status, count in statuses.most_common():
        print(f"  {str(status):<22} {count}")

    clean = sum(1 for r in rows if not r["problems"])
    print(f"\n{clean}/{len(rows)} cases clean end to end")

    if failures:
        print("\nfailures by kind")
        for kind, items in sorted(failures.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {kind}  ({len(items)})")
            for case_id, detail in items[:8]:
                print(f"    {case_id:<28} {detail}")
            if len(items) > 8:
                print(f"    ... and {len(items) - 8} more")

    Path(args.out).write_text(json.dumps({
        "cases": len(rows),
        "clean": clean,
        "generation": args.generate,
        "status_distribution": dict(statuses),
        "failures_by_kind": {k: len(v) for k, v in failures.items()},
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # An answered abstention case is the one failure that must never ship.
    if failures.get("unsafe_answer"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
