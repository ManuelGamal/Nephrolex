"""Live generation audit against a running server.

check_generation.py proves the verifier rejects fabricated values. It cannot show the
opposite failure: a verifier so strict that every real generation is discarded and the
whole layer silently degrades to extractive. Only live model output shows that, and
the rate that matters is *false* rejections - a rejection whose cited evidence in fact
contains the number.

Every rejection is printed with its reason and the discarded text so each one can be
judged rather than counted. A high rejection rate is not automatically bad; an
unexamined one is.

Requires a server started with --generate and GEMINI_API_KEY set in *its* environment.
The key is never read here.

Usage:
    python scripts/check_generation_live.py
    python scripts/check_generation_live.py --limit 20 --delay 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "ckd_gold_eval.jsonl"


def ask(base: str, query: str, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        f"{base}/api/ask",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between calls.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "generation_live.json"))
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    health = json.loads(urllib.request.urlopen(f"{base}/api/health", timeout=15).read())
    if not health.get("generate"):
        raise SystemExit(
            "server is running without generation.\n"
            "  restart it with:  python scripts/demo_server.py --generate\n"
            "  and GEMINI_API_KEY set in that shell."
        )

    with GOLD.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    answerable = [c for c in cases if not c["should_abstain"]][: args.limit]

    print(f"{len(answerable)} answerable questions | model resolved by the server\n")
    print(f"{'question':<52} {'status':<14} {'generation':<20} {'ms':>6}")

    rows, rejected, latencies = [], [], []
    counts: dict[str, int] = {}

    for case in answerable:
        query = case["query"]
        try:
            result = ask(base, query)
        except urllib.error.HTTPError as error:
            print(f"  HTTP {error.code} on: {query[:44]}")
            continue
        except Exception as error:  # noqa: BLE001
            print(f"  {type(error).__name__} on: {query[:44]}")
            continue

        generation = result.get("generated") or {}
        gen_status = generation.get("status", "absent")
        counts[gen_status] = counts.get(gen_status, 0) + 1
        latencies.append(result.get("latency_ms", 0))

        rows.append({
            "id": case.get("id"), "query": query,
            "answer_status": result.get("status"), "generation_status": gen_status,
            "latency_ms": result.get("latency_ms"),
            "problems": generation.get("verification", {}).get("problems", []),
            "error": generation.get("error"),
            "text": generation.get("text"),
            "rejected_text": generation.get("rejected_text"),
        })
        if gen_status == "failed_verification":
            rejected.append(rows[-1])

        print(f"  {query[:50]:<50} {result.get('status',''):<14} {gen_status:<20} "
              f"{result.get('latency_ms',0):>6}")
        time.sleep(args.delay)

    print()
    total = sum(counts.values()) or 1
    for status, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<22} {count:>3}  ({count / total:.0%})")
    if latencies:
        print(f"\nlatency  mean {statistics.mean(latencies):.0f} ms  "
              f"p50 {statistics.median(latencies):.0f} ms  max {max(latencies):.0f} ms")

    if rejected:
        print(f"\n{len(rejected)} REJECTION(S) - judge each, do not just count them:")
        for row in rejected:
            print(f"\n  Q: {row['query'][:78]}")
            for problem in row["problems"]:
                print(f"     reason: {problem}")
            if row["rejected_text"]:
                print(f"     discarded: {' '.join(row['rejected_text'].split())[:150]}")
    else:
        print("\nno rejections - the verifier discarded nothing the model produced")

    Path(args.out).write_text(json.dumps({
        "counts": counts, "queries": len(rows),
        "latency_ms": {"mean": round(statistics.mean(latencies)) if latencies else None,
                       "p50": round(statistics.median(latencies)) if latencies else None},
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
