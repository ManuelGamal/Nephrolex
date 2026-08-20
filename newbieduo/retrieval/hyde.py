"""Hypothetical Document Embeddings: rewrite the question into guideline language.

Why this and not the two techniques already rejected
---------------------------------------------------
The paraphrase slice fails because the question and the guideline use different
words. "At what point do I stop handling this in general practice and hand the
patient over?" contains no term the corpus uses, and the system answers it with a
practice point about written transfer summaries - it matched "hand over" to
"transfer".

Two techniques were built and rejected against this same weakness. Contextual
Retrieval enriched the *documents*; ColBERT changed the *matching function*. Neither
touched the query, which is where the mismatch actually is. HyDE does: an LLM writes
the answer it would expect a guideline to give, and that hypothetical - already in
clinical register - is what gets embedded.

    query        "when do I hand the patient over?"
    hypothetical "Refer adults with CKD for specialist assessment if they have a
                  5-year risk of needing renal replacement therapy above 5%..."

The hypothetical does not need to be correct. Nothing from it reaches the answer; it
is a search key, and it is thrown away after retrieval. A wrong hypothetical costs
ranking, never grounding.

Caching
-------
Evaluation must be reproducible and must not cost an API call per run, so
hypotheticals are cached to disk keyed by the query. `build` pre-populates the cache
for every gold question; retrieval then reads it with no network dependency. A cache
miss at query time generates one if a key is present, and otherwise degrades to the
plain query rather than failing.

Usage:
    python scripts/hyde.py build             # pre-generate for the gold set
    python scripts/hyde.py show "..."        # inspect one hypothetical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from newbieduo.paths import DATA, GOLD_SET  # noqa: E402

CACHE = DATA / "hyde_cache.jsonl"

# Short on purpose. A long hypothetical drifts into invented specifics and pulls the
# embedding toward whatever it happened to invent; the register is what matters, not
# the content.
PROMPT = """Write the passage a clinical guideline would contain if it answered this question.

Rules:
- Use the vocabulary and register of a clinical practice guideline for chronic kidney disease.
- Two sentences at most. No preamble, no citation, no hedging.
- It does not need to be factually correct. It is a search key, not an answer.

QUESTION
{question}

PASSAGE"""


def _key(query: str) -> str:
    return hashlib.sha1(" ".join(query.lower().split()).encode("utf-8")).hexdigest()[:16]


_CACHE: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if CACHE.exists():
            with CACHE.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        _CACHE[row["key"]] = row["hypothetical"]
    return _CACHE


def _append(query: str, hypothetical: str) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": _key(query), "query": query,
                            "hypothetical": hypothetical}, ensure_ascii=False) + "\n")
    _load()[_key(query)] = hypothetical


def _generate(query: str) -> str:
    """Call the model for one hypothetical. Raises, so callers can report why."""
    from newbieduo.generation.generate import DEFAULT_MODEL, available, call_anthropic, call_gemini

    provider = available()
    if not provider:
        raise RuntimeError("no API key (GEMINI_API_KEY or ANTHROPIC_API_KEY)")
    prompt = PROMPT.format(question=query.strip())
    model = DEFAULT_MODEL[provider]
    text = call_gemini(prompt, model) if provider == "gemini" else call_anthropic(prompt, model)
    return " ".join(text.split())


def hypothetical(query: str, allow_generate: bool = True) -> str | None:
    """The cached hypothetical for this query, generating one only if permitted.

    Errors are swallowed here on purpose: this sits in the retrieval path, where a
    failed search-key call must degrade to the plain query rather than take down the
    question. `build` calls `_generate` directly so its failures are reported instead.
    """
    cached = _load().get(_key(query))
    if cached is not None:
        return cached
    if not allow_generate:
        return None
    try:
        text = _generate(query)
    except Exception:  # noqa: BLE001 - a search key is never worth failing a query for
        return None
    if text:
        _append(query, text)
    return text or None


def build(limit: int = 0, verbose: bool = False, rpm: float = 10.0) -> None:
    """Pre-generate hypotheticals for every gold question.

    Paced, because free tiers meter requests per minute far more tightly than tokens.
    The first version of this fired all 75 as fast as they were accepted, was rate
    limited after 9, and reported "failed 66" with no reason attached - the errors
    went through the silent handler that belongs in the retrieval path, not here.
    """
    from newbieduo.generation.generate import available

    if not available():
        raise SystemExit(
            "no API key. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in this shell first."
        )

    with GOLD_SET.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if limit:
        cases = cases[:limit]

    made = skipped = failed = 0
    reasons: dict[str, int] = {}
    interval = 60.0 / max(rpm, 1)
    last = 0.0

    for case in cases:
        if _load().get(_key(case["query"])) is not None:
            skipped += 1
            continue

        wait = interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)

        text = None
        attempts = 3
        for attempt in range(attempts):
            try:
                last = time.time()
                text = _generate(case["query"])
                break
            except urllib.error.HTTPError as error:
                if error.code == 429:
                    if attempt == attempts - 1:
                        reasons["rate limited (retries exhausted)"] = \
                            reasons.get("rate limited (retries exhausted)", 0) + 1
                        break
                    header = error.headers.get("Retry-After") if error.headers else None
                    delay = float(header) if header and header.isdigit() else 20 * (attempt + 1)
                    print(f"  rate limited, waiting {delay:.0f}s")
                    time.sleep(delay)
                    continue
                label = f"HTTP {error.code}"
                reasons[label] = reasons.get(label, 0) + 1
                break
            except Exception as error:  # noqa: BLE001
                label = type(error).__name__
                reasons[label] = reasons.get(label, 0) + 1
                break

        if text:
            _append(case["query"], text)
            made += 1
            if verbose:
                print(f"  {case['id']:<28} {text[:92]}")
        else:
            failed += 1

    print(f"\ngenerated {made}, already cached {skipped}, failed {failed} -> {CACHE}")
    if reasons:
        print("failure reasons: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
        print("re-run to fill the gaps; anything already cached is skipped.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "show", "stats"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rpm", type=float, default=10.0,
                        help="Requests per minute. Free tiers meter by request, not token.")
    args = parser.parse_args()

    if args.command == "build":
        build(args.limit, args.verbose, args.rpm)
    elif args.command == "stats":
        cache = _load()
        print(f"{len(cache)} hypotheticals cached at {CACHE}")
    else:
        text = hypothetical(args.query or "")
        print(text or "(no hypothetical; cache miss and no API key)")


if __name__ == "__main__":
    main()
