"""Add LLM-generated situating context to each chunk (Anthropic's Contextual Retrieval).

A chunk taken out of a 199-page guideline often cannot be understood - or matched -
on its own. The clearest case in this corpus is NICE's definition of CKD, which
reads in full:

    "Abnormalities of kidney function or structure present for more than 3 months,
     with implications for health."

The term being defined lives in a heading, so the chunk never says "chronic kidney
disease". No embedding model can match "how does NICE define CKD" to text that does
not mention CKD. Prefixing a short generated sentence that situates the chunk in its
document fixes that class of failure. Anthropic report a 35% reduction in retrieval
failures for contextual embeddings, 49% with contextual BM25 included.

Runs once at build time and writes into the corpus, so it adds no query latency and
no network dependency during a demo.

Rate limiting
-------------
Chunks are batched into a single request (default 10), because free API tiers are
limited by *requests* per minute far more tightly than by tokens. One call per chunk
means 1,844 requests and certain 429s; batching by 10 means 185, which fits
comfortably. The client also paces itself to a target request rate and honours
Retry-After on 429 rather than guessing.

Key handling: read from the environment, never written to disk or into the corpus.

    $env:GEMINI_API_KEY    = "..."   # Google AI Studio
    $env:ANTHROPIC_API_KEY = "..."   # console.anthropic.com (not part of Claude Pro)

Usage:
    python scripts/contextualize_chunks.py --dry-run --limit 20
    python scripts/contextualize_chunks.py --rpm 10
    python scripts/contextualize_chunks.py --model gemini-2.5-flash-lite --rpm 15
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks_v2" / "all_chunks.jsonl"
CACHE = ROOT / "data" / "chunks_v2" / "chunk_context.jsonl"

MAX_CHUNK_CHARS = 900

# Front matter, contents listings and figure indexes are not clinical guidance and
# should not consume quota or compete during retrieval.
NON_GUIDANCE = re.compile(
    # Runs of page markers: contents pages and figure/table indexes.
    r"S\d{3}\s+[A-Z].{0,60}?S\d{3}\s+"
    # Two or more numbered figure/table captions in one chunk = an index, not content.
    r"|(?:Figure|Table)\s*\d+[.|].{0,400}?(?:Figure|Table)\s*\d+[.|]"
    r"|Work Group membership|Executive Committee|Reference keys|Conversion factors"
    r"|Abbreviations and acronyms|SUPPLEMENT(?:ARY)?\s+(?:TO|MATERIAL)|Patient foreword"
    r"|disclosure|Search strategies|PRISMA|Appendix [A-Z][.\s]",
    re.I,
)

# Section headings that mark a whole region as apparatus rather than guidance.
NON_GUIDANCE_SECTIONS = re.compile(
    r"^(FIGURES|TABLES|SUPPLEMENTARY MATERIAL|Abbreviations|Reference keys|Notice"
    r"|Conversion factors|Work Group|Biographic|Methods for guideline development"
    r"|Appendix|Data supplement|Summary of ?findings)",
    re.I,
)


def is_non_guidance(chunk: dict) -> bool:
    if NON_GUIDANCE.search(chunk.get("raw_text", "")):
        return True
    return any(NON_GUIDANCE_SECTIONS.match(part.strip()) for part in chunk.get("section_path", []))

BATCH_PROMPT = """You are indexing a clinical guideline for retrieval.

For EACH numbered chunk below, write ONE short sentence (max 30 words) that
situates it within its guideline so search can find it. Name the specific clinical
topic, the population it applies to, and what kind of statement it is
(recommendation, threshold, definition, table row).

Rules:
- State only what the chunk and its section heading support. Invent nothing.
- Do not repeat the chunk's wording; supply the context it is missing.
- If a chunk is front matter, a contents listing or an index, say exactly:
  "Front matter, not clinical guidance."

Return ONLY a JSON array, one object per chunk, no other text:
[{{"i": 0, "context": "..."}}, {{"i": 1, "context": "..."}}]

{chunks}"""

CHUNK_TEMPLATE = """
--- CHUNK {index} ---
Document: {document}
Section: {section}
{label_line}Text: {text}
"""


class NoKeyError(RuntimeError):
    pass


class TruncatedError(RuntimeError):
    """The model stopped at the token limit; the reply is unusable."""


class DailyQuotaError(RuntimeError):
    """The per-day allowance is gone. Retrying only burns more of tomorrow's."""


class ModelNotFoundError(RuntimeError):
    """The model id does not exist for this key; retrying cannot help."""


MIN_CONTEXT_WORDS = 6

# Models observed to reject generationConfig.thinkingConfig, remembered per run.
_NO_THINKING_CONFIG: set[str] = set()


class RateLimiter:
    """Pace requests to a target rate, and back off hard when told to."""

    def __init__(self, rpm: float) -> None:
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if not self.min_interval:
            return
        elapsed = time.time() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.time()


def _post(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_gemini(prompt: str, model: str, max_tokens: int) -> str:
    """Call Gemini, degrading gracefully if the model rejects a config field.

    Gemini 2.5 models reason before answering and charge that reasoning to the
    output budget; left on, the budget is spent thinking and the reply is truncated
    mid-sentence, which is what produced cached fragments like "This chunk is".
    Disabling it is therefore important - but the field is not accepted by every
    model or API version, and a rejected field surfaces as a 404/400 that looks
    like a wrong model id. So the request is retried once without it rather than
    failing the run.
    """
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    base_config = {"temperature": 0.0, "maxOutputTokens": max_tokens}

    attempts = [
        {**base_config, "thinkingConfig": {"thinkingBudget": 0}},
        base_config,
    ]
    if model in _NO_THINKING_CONFIG:
        attempts = [base_config]

    last_error: Exception | None = None
    for index, config in enumerate(attempts):
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
        try:
            data = _post(url, payload, {"Content-Type": "application/json", "x-goog-api-key": key})
        except urllib.error.HTTPError as error:
            if error.code in {400, 404} and index == 0 and len(attempts) > 1:
                # Remember, so the whole corpus is not re-probed for every batch.
                _NO_THINKING_CONFIG.add(model)
                last_error = error
                continue
            raise
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise TruncatedError(
                f"{model} hit maxOutputTokens; raise --batch-size down or the token budget"
            )
        parts = candidate.get("content", {}).get("parts") or []
        if not parts:
            raise TruncatedError(f"{model} returned no text (finishReason={candidate.get('finishReason')})")
        return parts[0]["text"]

    raise last_error or RuntimeError("gemini call failed")


def call_anthropic(prompt: str, model: str, max_tokens: int) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = _post(
        "https://api.anthropic.com/v1/messages",
        payload,
        {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return data["content"][0]["text"]


def list_gemini_models() -> list[dict]:
    """Ask the API which models this key can actually call.

    Model identifiers move between releases and differ per key/tier, so guessing one
    produces a 404 that looks like a code bug. Enumerating them is authoritative.
    """
    key = os.environ["GEMINI_API_KEY"]
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    usable = []
    for model in data.get("models", []):
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        usable.append(
            {
                "id": model["name"].removeprefix("models/"),
                "display": model.get("displayName", ""),
                "input_limit": model.get("inputTokenLimit"),
            }
        )
    return usable


def pick_provider(preferred: str | None) -> tuple[str, str]:
    have_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if preferred == "gemini" or (preferred is None and have_gemini):
        if not have_gemini:
            raise NoKeyError("GEMINI_API_KEY is not set")
        return "gemini", "gemini-flash-lite-latest"
    if preferred == "anthropic" or (preferred is None and have_anthropic):
        if not have_anthropic:
            raise NoKeyError("ANTHROPIC_API_KEY is not set")
        return "anthropic", "claude-haiku-4-5-20251001"
    raise NoKeyError("No API key found. Set GEMINI_API_KEY or ANTHROPIC_API_KEY.")


def generate(prompt: str, provider: str, model: str, max_tokens: int, limiter: RateLimiter,
             retries: int = 6) -> str:
    delay = 8.0
    for attempt in range(retries):
        limiter.wait()
        try:
            if provider == "gemini":
                return call_gemini(prompt, model, max_tokens)
            return call_anthropic(prompt, model, max_tokens)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise ModelNotFoundError(
                    f"model '{model}' was not found for this key. "
                    "Run with --list-models to see what is available."
                ) from error
            if error.code == 429:
                body = ""
                try:
                    body = error.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001 - body is best-effort diagnostics
                    pass
                # A per-day quota cannot be waited out inside a run, and each retry
                # spends another request against it. Stop the whole run instead.
                if re.search(r"per\s*day|PerDay|GenerateRequestsPerDay", body, re.I):
                    raise DailyQuotaError(
                        "daily quota exhausted for this model/key"
                    ) from error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            # Honour the server's own guidance when it gives any.
            retry_after = error.headers.get("Retry-After") if error.headers else None
            pause = float(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"      {error.code}; waiting {pause:.0f}s (attempt {attempt + 1}/{retries})")
            time.sleep(pause)
            delay = min(delay * 2, 120)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            print(f"      {type(error).__name__}; waiting {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_batch_prompt(batch: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(batch):
        blocks.append(
            CHUNK_TEMPLATE.format(
                index=index,
                document=chunk.get("document", ""),
                section=" > ".join(chunk.get("section_path", [])) or "(none)",
                label_line=f"Reference: {chunk['label']}\n" if chunk.get("label") else "",
                text=" ".join(chunk.get("raw_text", "").split())[:MAX_CHUNK_CHARS],
            )
        )
    return BATCH_PROMPT.format(chunks="".join(blocks))


def parse_batch(reply: str, size: int, stats: dict | None = None) -> dict[int, str] | None:
    """Extract the JSON array, tolerating fenced code blocks and stray prose."""
    match = re.search(r"\[.*\]", reply, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    out: dict[int, str] = {}
    returned = rejected = 0
    for item in items:
        try:
            index = int(item["i"])
            context = " ".join(str(item["context"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        returned += 1
        if 0 <= index < size and is_usable_context(context):
            out[index] = context
        else:
            rejected += 1
    if stats is not None:
        stats["returned"] = returned
        stats["rejected"] = rejected
        stats["omitted"] = size - returned
    return out or None


def is_usable_context(context: str) -> bool:
    """Reject truncated or contentless output before it reaches the cache.

    A cached bad value is worse than a missing one: the resume logic treats any
    cached chunk as done, so a fragment like "This chunk is" would silently poison
    the corpus and never be retried.
    """
    words = context.split()
    if len(words) < MIN_CONTEXT_WORDS:
        return False
    if not context.rstrip().endswith((".", "!", "?")):
        return False
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate situating context per chunk.")
    parser.add_argument("--provider", choices=["gemini", "anthropic"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rpm", type=float, default=10.0, help="Target requests per minute.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-front-matter", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="Show models this key can call.")
    args = parser.parse_args()

    if args.list_models:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set")
        models = list_gemini_models()
        print(f"{len(models)} models callable with this key:\n")
        for model in sorted(models, key=lambda m: m["id"]):
            print(f"  {model['id']:<44} {model['display']}")
        return

    chunks = [c for c in read_jsonl(CHUNKS) if c.get("indexable")]
    skipped = 0
    if not args.keep_front_matter:
        before = len(chunks)
        chunks = [c for c in chunks if not is_non_guidance(c)]
        skipped = before - len(chunks)

    cache = {row["chunk_id"] for row in read_jsonl(CACHE)} if CACHE.exists() else set()
    todo = [c for c in chunks if c["id"] not in cache]

    # Spend limited quota where context actually changes retrievability. A
    # recommendation chunk already begins "1.6.2 In adults with CKD and an ACR..."
    # and carries its own section; a bare narrative passage or table row does not.
    priority = {"section_passage": 0, "table_row": 1, "practice_point": 2, "recommendation_atom": 3}
    todo.sort(key=lambda c: priority.get(c.get("chunk_type", ""), 4))
    if args.limit:
        todo = todo[: args.limit]

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    print(
        f"{len(chunks)} indexable chunks ({skipped} front-matter skipped) | "
        f"{len(cache)} cached | {len(todo)} to do in {len(batches)} requests"
    )
    if args.rpm:
        print(f"pacing at {args.rpm:g} req/min -> about {len(batches) / args.rpm:.0f} min")

    if args.dry_run:
        if batches:
            print("\n" + "=" * 70)
            print(build_batch_prompt(batches[0])[:2200])
        print(f"\n--dry-run: no API calls made.")
        return

    try:
        provider, default_model = pick_provider(args.provider)
    except NoKeyError as error:
        print(f"\n{error}\n\nSet a key, then re-run. In PowerShell:")
        print('  $env:GEMINI_API_KEY = "your-key-here"')
        raise SystemExit(1)

    model = args.model or default_model
    limiter = RateLimiter(args.rpm)
    print(f"provider={provider} model={model} batch={args.batch_size}\n")

    done = failed = 0
    started = time.time()
    with CACHE.open("a", encoding="utf-8") as sink:
        for number, batch in enumerate(batches, start=1):
            prompt = build_batch_prompt(batch)
            try:
                reply = generate(prompt, provider, model, 90 * len(batch) + 400, limiter)
            except ModelNotFoundError as error:
                print(f"\n  STOPPING: {error}")
                break
            except DailyQuotaError as error:
                print(f"\n  STOPPING: {error}")
                print(f"  {done} chunks contextualised this run and cached.")
                print("  Re-run when the quota resets; cached chunks are skipped.")
                break
            except Exception as error:  # noqa: BLE001 - one bad batch must not end the run
                print(f"  batch {number}/{len(batches)} FAILED: {type(error).__name__}: {error}")
                failed += len(batch)
                continue

            stats: dict = {}
            parsed = parse_batch(reply, len(batch), stats)
            if not parsed:
                print(f"  batch {number}/{len(batches)}: unparseable reply, skipping")
                failed += len(batch)
                continue

            for index, context in parsed.items():
                sink.write(
                    json.dumps({"chunk_id": batch[index]["id"], "context": context}, ensure_ascii=False) + "\n"
                )
            sink.flush()
            done += len(parsed)

            rate = done / max(time.time() - started, 1e-6)
            remaining = (len(todo) - done) / max(rate, 1e-9)
            detail = ""
            if stats.get("omitted") or stats.get("rejected"):
                detail = f" (model omitted {stats.get('omitted', 0)}, rejected {stats.get('rejected', 0)})"
            print(
                f"  batch {number}/{len(batches)}  +{len(parsed)}/{len(batch)}{detail}  "
                f"total {done}/{len(todo)}  ~{remaining / 60:.0f} min left"
            )

    print(f"\ncontextualised {done} chunks ({failed} failed) -> {CACHE}")
    print("Re-run to fill any gaps; completed chunks are cached and skipped.")
    print("Then: python scripts/apply_context.py")


if __name__ == "__main__":
    main()
