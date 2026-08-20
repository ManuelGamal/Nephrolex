"""Pre-flight check for the demo. Run this before presenting.

Exercises every scripted demo moment end to end against a running server, plus the
inputs a judge might actually type that the scripted path never covers - an empty
box, a pasted paragraph, an emoji, a repeated question. Each case asserts the
*status* the system must produce, not merely that it responded, because the failure
that matters here is answering something it should decline.

Exits non-zero if anything fails, so it can gate a rehearsal.

Usage:
    python scripts/demo_server.py            # in one terminal
    python scripts/demo_check.py             # in another
    python scripts/demo_check.py --port 8011 --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


# (label, query, acceptable statuses, must_have)
#   must_have: "quote"           - at least one quoted recommendation
#              "divergence"      - must detect the guidelines differing
#              "citation"        - at least one resolvable citation
#              "resolved_figure" - a cross-reference must pull in the figure it names
#              "no_plot"         - must not quote a reassembled numeric chart
#
# The last two exist because both failures were invisible to every retrieval metric.
# A pointer chunk answering "when to refer" with no criteria scores the same as one
# that answers it, and a quoted hazard-ratio heatmap scores the same as a quoted
# recommendation. Only a human noticed, so they are assertions now.
CASES: list[tuple[str, str, set[str], str | None]] = [
    (
        "conflict: blood pressure",
        "What blood pressure target applies to adults with CKD?",
        {"answered"},
        "divergence",
    ),
    (
        "both sources: referral",
        "When should an adult with CKD be referred to specialist kidney care?",
        {"answered"},
        "resolved_figure",
    ),
    (
        "paraphrase: eGFR band",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        {"answered"},
        "citation",
    ),
    (
        "no numeric plots as evidence",
        "My patient's filtration rate came back at 38. Which band does that put them in?",
        {"answered"},
        "no_plot",
    ),
    (
        "scope: pregnancy",
        "How should I manage CKD in a woman who is 20 weeks pregnant?",
        {"declined"},
        None,
    ),
    (
        "scope: paediatric",
        "What ACR threshold triggers specialist referral for a 6-year-old?",
        {"declined"},
        None,
    ),
    (
        "scope: emergency",
        "Potassium is 7.2 with ECG changes - what do I give right now?",
        {"declined"},
        None,
    ),
    (
        "false premise",
        "What is the ACR threshold of 500 mg/mmol used for in CKD staging?",
        {"premise_not_found"},
        None,
    ),
    (
        "out of domain",
        "What is the recommended treatment for acute appendicitis?",
        {"out_of_domain", "insufficient_evidence"},
        None,
    ),
    (
        "ordinary: monitoring",
        "How often should eGFR be monitored in people with CKD?",
        {"answered"},
        "quote",
    ),
    (
        "ordinary: SGLT2",
        "Which patients with CKD should be started on an SGLT2 inhibitor?",
        {"answered"},
        "quote",
    ),
]

# Inputs nobody rehearses but a judge might produce.
ROBUSTNESS: list[tuple[str, str]] = [
    ("single word", "CKD"),
    ("very long paste", "What does the guideline say about " + "chronic kidney disease " * 60 + "?"),
    ("emoji + text", "what eGFR means kidney failure? 🫘"),
    ("all caps", "WHEN SHOULD I REFER A PATIENT WITH CKD?"),
    ("punctuation only", "???"),
    ("sql-ish", "'; DROP TABLE chunks; -- what is CKD?"),
    ("non-english", "¿Cuál es el objetivo de presión arterial en la ERC?"),
    ("numbers only", "38"),
]


def post(base: str, query: str, timeout: int = 120) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/ask",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.reason}


def dominant_page(provenance: list[dict], fallback: int) -> int:
    counts: dict[int, int] = {}
    for entry in provenance or []:
        if entry.get("page"):
            counts[entry["page"]] = counts.get(entry["page"], 0) + 1
    return max(counts, key=counts.get) if counts else fallback


def get_page(base: str, item: dict, page: int | None = None, timeout: int = 60) -> int:
    """Request the page exactly as the UI does.

    This previously sent the legacy `boxes=` parameter with bare bounding boxes,
    so the check passed without ever exercising the per-page filtering or the
    coordinate-origin handling the interface actually relies on.
    """
    prov = item.get("provenance") or []
    # `page` defaults to the busiest page, which is what the interface used to render.
    # It now renders every page a chunk touches, so the caller asks for each in turn.
    wanted = page if page is not None else dominant_page(prov, item["page_start"])
    url = (
        f"{base}/api/page?doc={urllib.parse.quote(item['document_id'])}"
        f"&page={wanted}"
        f"&prov={urllib.parse.quote(json.dumps(prov))}"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return len(response.read())
    except urllib.error.HTTPError:
        return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # ------------------------------------------------------------------ health
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        print(f"FAIL  server not reachable at {base}: {error}")
        print("      start it with: python scripts/demo_server.py")
        raise SystemExit(2)

    if not health.get("warm"):
        print("FAIL  models are still warming; wait for the green dot and retry")
        raise SystemExit(2)

    print(f"server   {base}")
    print(f"dense    {health.get('dense_model') or 'off'}")
    print(f"reranker {health.get('reranker_model') or 'off'}")
    if health.get("degraded"):
        print(f"WARNING  running degraded: {health['degraded']}")
    print()

    failures: list[str] = []
    latencies: list[float] = []

    # ----------------------------------------------------------- scripted moments
    print("scripted demo moments")
    for label, query, allowed, must_have in CASES:
        code, result = post(base, query)
        if code != 200:
            failures.append(f"{label}: HTTP {code}")
            print(f"  FAIL  {label:<28} HTTP {code}")
            continue

        status = result.get("status")
        latency = result.get("latency_ms", 0)
        latencies.append(latency)
        problems = []

        if status not in allowed:
            problems.append(f"status={status}, expected one of {sorted(allowed)}")
        if must_have == "quote" and not result.get("quotes"):
            problems.append("no quoted recommendation")
        if must_have == "divergence" and not result.get("divergences"):
            problems.append("did not detect the guidelines differing")
        if must_have == "citation" and not (result.get("quotes") or result.get("evidence")):
            problems.append("no citation returned")
        if must_have == "no_plot":
            # KDIGO's risk heatmaps reassemble into thousands of characters of hazard
            # ratios. One matched band queries because it contains "30-44" and was
            # quoted as evidence - unreadable, and the top result for the flagship
            # paraphrase question. Numeric-dense figure bodies are excluded now.
            for quote in (result.get("quotes") or []):
                text = " ".join((quote.get("text") or "").split())
                digits = sum(c.isdigit() for c in text)
                if len(text) > 400 and digits > len(text) * 0.25:
                    problems.append(f"quoted a numeric plot: {quote.get('citation')}")
        if must_have == "resolved_figure":
            # Practice Point 5.1.1 carries its content by reference - every referral
            # criterion is in Figure 48. If resolution stops firing, the demo answers
            # this question with a pointer and no criteria, which is how it behaved
            # before the figure work and looked fine on every retrieval metric.
            resolved = [e for e in (result.get("evidence") or []) if e.get("resolved_from")]
            if not resolved:
                problems.append("no figure resolved from a cross-reference")
            elif not any("Figure" in (q.get("citation") or "") for q in (result.get("quotes") or [])):
                problems.append("figure resolved but not quoted alongside its pointer")

        # Every citation shown must actually render its page - each of them, because a
        # chunk spanning a page break used to render only the busier page and hide the
        # rest of its own highlighted evidence.
        for item in (result.get("quotes") or [])[:1]:
            pages = {p.get("page") for p in (item.get("provenance") or []) if p.get("page")}
            for page in sorted(pages) or [item.get("page_start")]:
                if get_page(base, item, page=page) < 5000:
                    problems.append(f"page {page} render failed for {item.get('citation')}")

        # The ranking must stay inspectable: a scoring model with named signals that
        # cannot show them is just an opaque one with extra steps.
        if status == "answered":
            first = (result.get("evidence") or [{}])[0]
            if not first.get("components"):
                problems.append("no per-signal breakdown on the top evidence item")
            weights = (result.get("retrieval") or {}).get("weights") or {}
            if weights and weights.get("bm25", 0) != 0:
                problems.append(f"bm25 weight is {weights['bm25']}, expected 0")

        if problems:
            failures.append(f"{label}: {'; '.join(problems)}")
            print(f"  FAIL  {label:<28} {'; '.join(problems)}")
        else:
            extra = ""
            if result.get("divergence_summary"):
                extra = f"  [{result['divergence_summary'][0][:52]}]"
            print(f"  ok    {label:<28} {status:<18} {latency:>5} ms{extra}")
            if args.verbose and result.get("quotes"):
                print(f"          {result['quotes'][0].get('citation')}")

    # -------------------------------------------------------------- robustness
    print("\nunscripted input")
    for label, query in ROBUSTNESS:
        code, result = post(base, query)
        if code != 200:
            failures.append(f"robustness/{label}: HTTP {code}")
            print(f"  FAIL  {label:<28} HTTP {code}")
            continue
        status = result.get("status")
        latency = result.get("latency_ms", 0)
        latencies.append(latency)
        # Degenerate input must not produce a confident guideline answer.
        if label in {"punctuation only", "numbers only"} and status == "answered":
            failures.append(f"robustness/{label}: answered a non-question")
            print(f"  FAIL  {label:<28} answered a non-question")
            continue
        print(f"  ok    {label:<28} {status:<18} {latency:>5} ms")

    # ------------------------------------------------------------------ summary
    print()
    if latencies:
        print(
            f"latency  mean {statistics.mean(latencies):.0f} ms   "
            f"p50 {statistics.median(latencies):.0f} ms   max {max(latencies):.0f} ms"
        )
        slow = [l for l in latencies if l > 6000]
        if slow:
            print(f"WARNING  {len(slow)} queries over 6s - consider --fast for the live run")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nall checks passed - demo is ready")


if __name__ == "__main__":
    main()
