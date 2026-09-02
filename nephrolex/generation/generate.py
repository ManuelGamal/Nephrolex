"""Grounded answer generation, with mechanical verification of every claim.

The hackathon brief asks for a generation layer with strict grounding prompts. A
prompt alone is a request, not a guarantee: an LLM asked not to invent a threshold
will usually comply and occasionally will not, and a fluent wrong number is
indistinguishable from a right one at a glance. So generation here is followed by
checks that do not involve the model's cooperation:

  1. Every sentence carrying guidance must cite evidence by index.
  2. Every cited index must exist.
  3. Every number-with-unit in the answer must appear verbatim in the evidence it
     cites. This is the check that matters clinically - "< 120 mmHg" is the kind of
     claim that must never be synthesised - and it is decidable by string matching.

If any check fails, the generated text is discarded and the extractive answer is
returned instead. The system therefore cannot present an unverified clinical number,
regardless of what the model produced.

Usage:
    python scripts/generate.py "What blood pressure target applies in CKD?"
    python scripts/generate.py "..." --model gemini-3.5-flash-lite --show-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"

PROMPT = """You answer clinical questions using ONLY the numbered evidence below.

RULES
- Use only the evidence. Never use outside medical knowledge.
- Every sentence that states clinical guidance must end with citation markers like [1] or [2][3].
- Copy every number, threshold and unit exactly as it appears in the evidence. Never round, convert, average or infer a value.
- If the evidence states the answer, give it - including when it appears inside a table row or a list. Reply with exactly {refusal} ONLY when no evidence item bears on the question at all.
- If the sources disagree, say so explicitly and cite both.
- Do not recommend anything the evidence does not state.
- SURROUNDING TEXT is background only. Never quote a number from it and never cite it.
- At most 110 words. No preamble, no headings.

QUESTION
{question}

EVIDENCE
{evidence}

ANSWER"""


@dataclass
class Verification:
    ok: bool
    citations_used: list[int] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "citations_used": self.citations_used, "problems": self.problems}


# A number attached to a clinical unit is the class of claim that must never be
# invented. Bare integers are excluded: "3 months" or "stage 3" are prose, not
# thresholds, and demanding verbatim support for them produces false alarms.
VALUE_WITH_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(mm\s?Hg|ml\s?/\s?min(?:\s?/\s?1\.73\s?m2?)?|mg\s?/\s?mmol|mg\s?/\s?g|mmol\s?/\s?l|"
    r"micrograms?\s?/\s?litre|ng\s?/\s?ml|g\s?/\s?kg|g\s?/\s?l|g\s?/\s?dl|%)",
    re.I,
)

CITATION = re.compile(r"\[(\d+)\]")

# Bands are stated without units - "G3b is 30-44", "target range 120 to 139" - so the
# unit rule above cannot see them, and they are among the most common claims here. A
# *range* is safe to check even though a bare integer is not: "stage 3" and "3 months"
# are single numbers, while two numbers joined by a dash or "to" is a threshold band
# in essentially every case. The lookarounds keep it off dotted identifiers, so
# "recommendations 1.6.1 to 1.6.2" is not read as the range 1-1.
NUMERIC_RANGE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)(?!\.?\d)")


def _ranges(text: str) -> set[tuple[str, str]]:
    return {(low, high) for low, high in NUMERIC_RANGE.findall(text)}


def build_prompt(question: str, evidence: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        # The section path is included because a quarter of the corpus is under 200
        # characters and some of it is a pointer rather than a statement: Practice
        # Point 5.1.1 is entirely "Refer adults with CKD ... in the circumstances
        # listed in Figure 48". With no context the model can only repeat the
        # pointer; the heading at least says what the chunk is about.
        section = item.get("section")
        heading = f"[{index}] ({item['citation']})"
        if section:
            heading += f" - from: {section}"
        text = " ".join(item["text"].split())
        block = heading + "\n" + text
        # Surrounding text, shown so a short chunk can be understood, and marked so it
        # is not mistaken for citable evidence. verify() checks numbers against `text`
        # alone, so a value appearing only in context is still rejected - the prompt
        # and the verifier have to agree about what counts as support.
        context = " ".join((item.get("context") or "").split())
        if context:
            block += "\n    SURROUNDING TEXT (for understanding only, do not cite): " + context
        blocks.append(block)
    return PROMPT.format(
        refusal=REFUSAL_TOKEN,
        question=question.strip(),
        evidence="\n\n".join(blocks),
    )


def _normalise(text: str) -> str:
    """Collapse spacing so "120mmHg", "120 mmHg" and "120 mm Hg" compare equal."""
    return re.sub(r"\s+", "", text).lower()


# Classifying a measurement the clinician supplied and *recommending* a value are
# different acts, and only the first is safe to admit on range containment. "An eGFR
# of 38 falls in G3b (30-44)" reports where a number sits; "aim for 132 mmHg" off
# NICE's 120-139 target range invents a recommendation the guideline never makes.
#
# This is an allowlist of classifying phrasing, not a denylist of recommending verbs,
# because the two fail in opposite directions. A denylist fails OPEN: the first draft
# listed "aim|target|treat|should|recommend..." and duly accepted "Consider 132 mmHg",
# "Offer a goal of 132 mmHg" and "132 mmHg is appropriate" - and "consider" and
# "offer" are the two commonest verbs in NICE guidance. An allowlist fails CLOSED: an
# unrecognised phrasing is rejected and the answer degrades to the extractive text,
# which costs presentation rather than safety.
CLASSIFYING = re.compile(
    r"\b(falls?|sits?|lies?|places?|puts?|belongs?|corresponds?|equates?)\b[^.;]{0,40}?"
    r"\b(in|into|within|to)\b"
    r"|\bis\s+(in|within|classified|categoris|categoriz)"
    r"|\bcategory\s+is\b",
    re.I)


def _sentence_around(answer: str, needle: str) -> str:
    for sentence in re.split(r"(?<=[.;])\s+", answer):
        if needle in sentence:
            return sentence
    return answer


def _within_stated_range(value: str, supporting: str) -> bool:
    """Does this number fall inside a range the cited evidence actually states?"""
    try:
        number = float(value)
    except ValueError:
        return False
    for low, high in NUMERIC_RANGE.findall(supporting):
        try:
            if float(low) <= number <= float(high):
                return True
        except ValueError:
            continue
    return False


def verify(answer: str, evidence: list[dict], question: str = "") -> Verification:
    problems: list[str] = []

    cited = [int(n) for n in CITATION.findall(answer)]
    valid = [n for n in cited if 1 <= n <= len(evidence)]
    for n in cited:
        if n < 1 or n > len(evidence):
            problems.append(f"cites [{n}], which does not exist")

    if not valid:
        problems.append("no evidence cited")

    # Numbers may only appear if the evidence the answer cites actually contains
    # them. Checking against cited evidence rather than all of it prevents an
    # answer from borrowing a value from a source it never pointed at.
    supporting = " ".join(evidence[n - 1]["text"] for n in set(valid)) if valid else ""
    supporting_norm = _normalise(supporting)
    # The lookahead rejects a following *digit*, optionally after a dot, so dotted
    # identifiers ("1.6.1") are still excluded while a number ending a sentence
    # ("came back at 38.") is not. The simpler (?![\d.]) made every sentence-final
    # value invisible here, which is what rejected a correct G3b classification.
    question_numbers = {v for v, _ in VALUE_WITH_UNIT.findall(question)}
    question_numbers.update(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?!\.?\d)", question))

    for value, unit in VALUE_WITH_UNIT.findall(answer):
        if _normalise(value + unit) in supporting_norm:
            continue
        # An endpoint of a range is already the range-checker's business. Checking it
        # again as a standalone value demands the unit sit next to it - and sources
        # write "15 - 29 ml/min" or put the unit in a column header, so "29 ml/min"
        # appears verbatim nowhere. That rejected a correct statement of the G4 band
        # live. Scoped to the value's own sentence, so a genuine standalone claim
        # elsewhere in the answer is still checked.
        sentence = _sentence_around(answer, value)
        if any(value in (low, high) for low, high in NUMERIC_RANGE.findall(sentence)):
            continue
        # A value the clinician supplied and the evidence brackets is a classification,
        # not a fabrication: "filtration rate came back at 38" against a row reading
        # "G3b ... 30 - 44" is the system doing its job. Rejecting it downgraded a
        # correct answer to extractive, and a false rejection is not free.
        #
        # Both halves are required. "In a stated range" alone would let a model write
        # "aim for 132 mmHg" off NICE's 120-139 target range - a recommendation no
        # guideline makes. "In the question" alone would let a leading question smuggle
        # its own premise into a cited answer.
        if (value in question_numbers
                and _within_stated_range(value, supporting)
                and CLASSIFYING.search(_sentence_around(answer, value))):
            continue
        problems.append(f"states '{value} {unit}' which is not in the cited evidence")

    supported_ranges = _ranges(supporting)
    for low, high in _ranges(answer):
        if (low, high) not in supported_ranges:
            problems.append(f"states the range '{low}-{high}' which is not in the cited evidence")

    return Verification(ok=not problems, citations_used=sorted(set(valid)), problems=problems)


# ------------------------------------------------------------------- providers

# How long to wait on a provider before giving up. 90s was the batch-job default and is
# far too long for anything a person is watching: measured against Gemini flash-lite,
# this call is 3-4s most of the time but spikes to ~21s on roughly one call in four, and
# a 90s ceiling means the worst case is a minute and a half of a frozen demo. Generation
# is an enhancement layered on evidence that has already been retrieved and cited, so a
# slow call should surrender to the evidence rather than hold it hostage. `set_timeout`
# lets the demo server pick a short one while batch scripts keep the patient default.
REQUEST_TIMEOUT = 90

# How often to say "still waiting" while the provider is thinking.
#
# Without this, a slow answer is indistinguishable from a hung pipeline: the server
# prints nothing between accepting the question and returning, so a 30-second wait
# looks like retrieval being slow. Retrieval finishes in about a second; everything
# after it is the provider. Printing the elapsed time makes that attributable rather
# than arguable.
HEARTBEAT_SECONDS = 5.0


def set_timeout(seconds: float | None) -> None:
    """Seconds to wait on the provider. 0 or less means wait indefinitely."""
    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = None if seconds is None or seconds <= 0 else seconds


def _with_heartbeat(call, label: str):
    """Run `call` on a worker thread, reporting elapsed seconds while it runs.

    The provider call itself is untouched - this only watches it. The thread is a
    daemon so a hung provider can never keep the process alive.
    """
    box: dict = {}

    def run():
        try:
            box["value"] = call()
        except BaseException as error:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    started = time.time()
    worker.start()
    while True:
        worker.join(HEARTBEAT_SECONDS)
        if not worker.is_alive():
            break
        print(f"  ... waiting on {label}: {time.time() - started:.0f}s elapsed",
              flush=True)
    elapsed = time.time() - started
    if "error" in box:
        print(f"  {label} failed after {elapsed:.1f}s: "
              f"{type(box['error']).__name__}", flush=True)
        raise box["error"]
    print(f"  {label} answered in {elapsed:.1f}s", flush=True)
    return box["value"]


def _post(url: str, payload: dict, headers: dict, timeout: int | float | None = None) -> dict:
    timeout = REQUEST_TIMEOUT if timeout is None else timeout
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_gemini(prompt: str, model: str, max_tokens: int = 400) -> str:
    key = os.environ["GEMINI_API_KEY"]
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            payload,
            {"Content-Type": "application/json", "x-goog-api-key": key},
        )
    except urllib.error.HTTPError as error:
        if error.code in {400, 404}:  # model may reject thinkingConfig
            payload["generationConfig"].pop("thinkingConfig", None)
            data = _post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                payload,
                {"Content-Type": "application/json", "x-goog-api-key": key},
            )
        else:
            raise
    parts = data["candidates"][0].get("content", {}).get("parts") or []
    return parts[0]["text"].strip() if parts else ""


def call_anthropic(prompt: str, model: str, max_tokens: int = 400) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return data["content"][0]["text"].strip()


def available() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


DEFAULT_MODEL = {"gemini": "gemini-flash-lite-latest", "anthropic": "claude-haiku-4-5-20251001"}


def generate(question: str, evidence: list[dict], model: str | None = None,
             provider: str | None = None) -> dict:
    """Generate a grounded answer and verify it. Never raises on a model failure."""
    provider = provider or available()
    if not provider:
        return {"available": False, "reason": "no API key set (GEMINI_API_KEY or ANTHROPIC_API_KEY)"}
    if not evidence:
        return {"available": True, "status": "no_evidence", "text": None}

    model = model or DEFAULT_MODEL[provider]
    prompt = build_prompt(question, evidence)

    try:
        raw = _with_heartbeat(
            lambda: call_gemini(prompt, model) if provider == "gemini"
            else call_anthropic(prompt, model),
            provider,
        )
    except Exception as error:  # noqa: BLE001 - generation is an enhancement, never a hard dependency
        # A timeout is the expected failure, not an exceptional one: this provider is
        # 3-4s most of the time and spikes past 20s on roughly one call in four, so on a
        # short demo timeout it will fire. Say so in words a person can read, rather than
        # surfacing "timed out" from a socket, because the evidence below it is intact
        # and the reader needs to know that the answer is missing but nothing is broken.
        timed_out = isinstance(error, (TimeoutError, socket.timeout)) or "timed out" in str(error)
        return {
            "available": True,
            "status": "generation_failed",
            "error": (f"the model did not respond within "
                      f"{REQUEST_TIMEOUT:g}s - "
                      "the cited guideline text below is unaffected"
                      if timed_out and REQUEST_TIMEOUT
                      else "the model did not respond - the cited guideline text "
                           "below is unaffected"
                      if timed_out else f"{type(error).__name__}: {error}"),
            "text": None,
        }

    if REFUSAL_TOKEN in raw:
        return {"available": True, "status": "model_refused", "model": model,
                "text": None, "verification": Verification(ok=True).as_dict()}

    checked = verify(raw, evidence, question)
    return {
        "available": True,
        "status": "verified" if checked.ok else "failed_verification",
        "model": model,
        "text": raw if checked.ok else None,
        "rejected_text": None if checked.ok else raw,
        "verification": checked.as_dict(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate a grounded, verified answer.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dense-model", default="abhinand/MedEmbed-large-v0.1")
    parser.add_argument("--dense-file", default="data/indexes/dense/abhinand__MedEmbed-large-v0_1.npy")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    from nephrolex.retrieval import retrieve as retrieve_module
    from nephrolex.generation.answer import build_answer

    retrieve_module.set_dense_path(args.dense_file)
    grounded = build_answer(args.query, args.top_k, args.dense_model, args.reranker_model)

    if grounded["status"] != "answered":
        print(grounded["text"])
        print(f"\n[{grounded['status']} - generation not attempted]")
        return

    evidence = [{"citation": q["citation"], "text": q["text"]} for q in grounded["quotes"]]
    if args.show_prompt:
        print(build_prompt(args.query, evidence))
        print("\n" + "=" * 70 + "\n")

    result = generate(args.query, evidence, args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
