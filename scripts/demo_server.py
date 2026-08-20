"""NewbieDuo demo server: ask a CKD question, see the evidence on the page it came from.

Deliberately built on the standard library. A live demo should not depend on a web
framework installed hours beforehand alongside a CUDA torch build, and every failure
mode here is one this file can handle.

Endpoints:
    GET  /                     the app
    POST /api/ask              {"query": ...} -> grounded answer + evidence + claim comparison
    GET  /api/page?...         PNG of the cited PDF page with the cited span boxed
    GET  /api/health           model warm-up state

Design notes for the demo:
  - Models load once at startup, not per query, so nothing stalls on stage.
  - Page renders are cached; the second click on a citation is instant.
  - If the dense model or reranker fails to load, retrieval degrades to lexical and
    says so, rather than 500ing mid-question.

Usage:
    python scripts/demo_server.py
    python scripts/demo_server.py --port 8000 --no-rerank
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import traceback
import urllib.parse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.generation import answer as answer_module  # noqa: E402
from newbieduo.retrieval import retrieve as retrieve_module  # noqa: E402


RAW_DIR = ROOT / "data" / "raw"
STATIC = Path(__file__).parent / "demo_static"

PDF_BY_DOCUMENT = {
    "kdigo_2024_ckd": "KDIGO-2024-CKD-Guideline.pdf",
    "nice_ng203_ckd": "chronic-kidney-disease-assessment-and-management-pdf-66143713055173.pdf",
}

EXAMPLE_QUESTIONS = [
    {"q": "What blood pressure target applies to adults with CKD?", "tag": "guidelines disagree"},
    {"q": "When should an adult with CKD be referred to specialist kidney care?", "tag": "both sources"},
    {"q": "My patient's filtration rate came back at 38. Which band does that put them in?", "tag": "no shared vocabulary"},
    {"q": "How should I manage CKD in a woman who is 20 weeks pregnant?", "tag": "out of scope"},
    {"q": "What is the ACR threshold of 500 mg/mmol used for in CKD staging?", "tag": "false premise"},
    {"q": "What is the recommended treatment for acute appendicitis?", "tag": "not a kidney question"},
]


class Config:
    dense_model: str | None = None
    dense_file: str | None = None
    reranker_model: str | None = None
    top_k: int = 10
    rerank_depth: int = 30
    generate: bool = False
    warm: bool = False
    degraded: str | None = None
    # "warming" -> "ready" | "degraded". A hook arriving while this is "warming" is
    # answered with 503 rather than being made to wait for the warm-up.
    cds_state: str = "warming"
    cds_warm_seconds: float | None = None


CONFIG = Config()


def cds_answerer(question: str) -> dict:
    """How a CDS trigger question gets answered - one definition, used by both paths.

    Extractive only. A card is pushed at a clinician who did not ask for it, so it
    carries verbatim guideline text and its citation, never generated phrasing. It has
    to be the same function that warms the cache and that serves a hook, or the warm-up
    fills the cache with answers the request will not find.
    """
    return answer_module.build_answer(
        question, CONFIG.top_k, CONFIG.dense_model, CONFIG.reranker_model,
        rerank_depth=CONFIG.rerank_depth, generate=False,
    )


def warm_cds() -> None:
    """Pre-answer every CDS trigger question, before any record system asks.

    Measured before this existed: a cold POST to /cds-services/ckd-guidance took
    **689.6 s**, because the handler warmed on first use and ran five full retrievals
    inline while the caller waited. The same call served warm is **15 ms**. The work is
    identical; the only question was whether a clinician's chart was blocked on it.

    This is worth doing eagerly precisely because the questions are fixed. Only *which*
    triggers fire depends on the patient - the five questions themselves never change,
    so the whole cache can be built before the first patient is ever seen.
    """
    from newbieduo.service import cds_hooks

    started = time.time()
    try:
        cached = cds_hooks.warm(cds_answerer)
        CONFIG.cds_warm_seconds = round(time.time() - started, 1)
        CONFIG.cds_state = "ready"
        print(f"CDS triggers warm in {CONFIG.cds_warm_seconds}s "
              f"({cached} questions cached)")
    except Exception as error:  # noqa: BLE001
        # Leave the endpoint reachable rather than bricked by a transient failure: it
        # falls back to answering on demand, bounded by the triggers that actually fire.
        CONFIG.cds_state = "degraded"
        print(f"WARNING: CDS warm-up failed, hooks will answer on demand - "
              f"{type(error).__name__}: {error}")


def warm_models() -> None:
    """Load everything once so the first question on stage is not the slow one."""
    started = time.time()
    try:
        if CONFIG.dense_file:
            retrieve_module.set_dense_path(CONFIG.dense_file)
        retrieve_module.load_indexes()
        answer_module.build_answer(
            "What blood pressure target applies to adults with CKD?",
            CONFIG.top_k,
            CONFIG.dense_model,
            CONFIG.reranker_model,
            rerank_depth=CONFIG.rerank_depth,
        )
        CONFIG.warm = True
        print(f"models warm in {time.time() - started:.1f}s")
    except Exception as error:  # noqa: BLE001 - degrade rather than refuse to start
        CONFIG.degraded = f"{type(error).__name__}: {error}"
        CONFIG.dense_model = None
        CONFIG.reranker_model = None
        CONFIG.warm = True
        print(f"WARNING: falling back to lexical retrieval - {CONFIG.degraded}")

    # Only after the models are up, so this measures retrieval rather than model load.
    warm_cds()


@lru_cache(maxsize=64)
def render_page(document_id: str, page: int, prov_key: str, dpi: int = 130,
                crop: bool = False) -> bytes:
    """Render one PDF page to PNG, boxing only the spans that live on that page.

    Filtering by page is essential, not defensive. A chunk may span a page break -
    one NICE passage carries one box on page 51 and four on page 52 - and drawing
    all of them on the first page paints page-52 coordinates over page-51 text,
    producing boxes that bound nothing and boxes offset from the words they cite.
    """
    import pymupdf

    pdf_name = PDF_BY_DOCUMENT.get(document_id)
    if not pdf_name:
        raise KeyError(f"unknown document {document_id}")

    provenance = json.loads(prov_key) if prov_key else []
    with pymupdf.open(RAW_DIR / pdf_name) as pdf:
        pdf_page = pdf[page - 1]
        height = pdf_page.rect.height
        drawn: list = []
        for entry in provenance:
            origin = "BOTTOMLEFT"
            if isinstance(entry, dict):
                if entry.get("page") != page:
                    continue
                box = entry.get("bbox")
                origin = entry.get("coord_origin") or origin
            else:
                box = entry  # legacy: a bare bbox, assumed to be on this page
            if not box or len(box) != 4 or any(v is None for v in box):
                continue
            left, top, right, bottom = box
            # Docling mixes origins: element boxes are BOTTOMLEFT, table-cell boxes
            # TOPLEFT. Flipping the wrong one puts the highlight on the opposite half
            # of the page, so the recorded origin is honoured rather than assumed.
            if origin == "TOPLEFT":
                rect = pymupdf.Rect(left, top, right, bottom)
            else:
                rect = pymupdf.Rect(left, height - top, right, height - bottom)

            # A cited table row is drawn boldly; the table it belongs to is outlined
            # faintly around it, so a single row is never shown stripped of the
            # table that gives it meaning.
            if isinstance(entry, dict) and entry.get("role") == "table_context":
                pdf_page.draw_rect(rect, color=(0.55, 0.58, 0.65), width=0.8, dashes="[3 3] 0")
            else:
                pdf_page.draw_rect(rect, color=(0.85, 0.15, 0.15), width=1.6)
            drawn.append(rect)

        # A whole A4 page shrunk into a side panel renders the cited sentence about ten
        # pixels tall - the provenance is present and unreadable, which is the worst of
        # both. Cropping to the boxed span with a margin of surrounding lines shows the
        # sentence at a size a person can actually read, still in place on the page and
        # still inside its own box.
        if crop and drawn:
            span = drawn[0]
            for rect in drawn[1:]:
                span |= rect
            pad_x, pad_y = 26, 30
            window = pymupdf.Rect(
                max(pdf_page.rect.x0, span.x0 - pad_x),
                max(pdf_page.rect.y0, span.y0 - pad_y),
                min(pdf_page.rect.x1, span.x1 + pad_x),
                min(pdf_page.rect.y1, span.y1 + pad_y),
            )
            # Render the crop at a higher dpi: it covers less of the page, so the same
            # panel width buys proportionally more detail.
            return pdf_page.get_pixmap(dpi=dpi * 2, clip=window).tobytes("png")
        return pdf_page.get_pixmap(dpi=dpi).tobytes("png")


def dominant_page(provenance: list[dict], fallback: int) -> int:
    """The page carrying most of a chunk's text, which is what should be shown.

    `page_start` can be the page holding a single trailing line, so a chunk that is
    mostly on page 52 would open on 51 with one box near the bottom edge.
    """
    counts: dict[int, int] = {}
    for entry in provenance or []:
        page = entry.get("page")
        if page:
            counts[page] = counts.get(page, 0) + 1
    return max(counts, key=counts.get) if counts else fallback


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter console during a demo
        if "/api/ask" in (args[0] if args else ""):
            print(f"  {args[0]}")

    # ------------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, content_type: str,
              extra_headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        extra = {"Retry-After": "15"} if code == 503 else None
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", extra)

    # --------------------------------------------------------------------- GET
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return

        if parsed.path == "/cds-services":
            # CDS Hooks discovery. A record system fetches this to learn what the
            # service offers and what patient data to send with the hook.
            from newbieduo.service.cds_hooks import discovery

            self._json(200, discovery())
            return

        if parsed.path == "/api/faq":
            # The clinician-FAQ benchmark, served from the same file the evaluation
            # reads, so what is on screen cannot drift from what was scored.
            path = ROOT / "eval" / "faq_clinical.jsonl"
            try:
                rows = [json.loads(line) for line in
                        path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                self._json(200, [])
                return
            self._json(200, [{"id": r.get("id"), "query": r.get("query"),
                              "category": r.get("category")} for r in rows])
            return

        if parsed.path == "/api/health":
            self._json(200, {
                "warm": CONFIG.warm,
                "dense_model": CONFIG.dense_model,
                "reranker_model": CONFIG.reranker_model,
                "degraded": CONFIG.degraded,
                "cds_state": CONFIG.cds_state,
                "cds_warm_seconds": CONFIG.cds_warm_seconds,
                "generate": CONFIG.generate,
                "examples": EXAMPLE_QUESTIONS,
            })
            return

        if parsed.path == "/api/page":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                document_id = params["doc"][0]
                page = int(params["page"][0])
                prov = params.get("prov", params.get("boxes", ["[]"]))[0]
                crop = params.get("crop", ["0"])[0] in {"1", "true", "yes"}
                png = render_page(document_id, page, prov, crop=crop)
            except Exception as error:  # noqa: BLE001
                self._json(400, {"error": f"{type(error).__name__}: {error}"})
                return
            self._send(200, png, "image/png")
            return

        self._json(404, {"error": "not found"})

    # -------------------------------------------------------------------- POST
    def _cds_hook(self, service_id: str) -> None:
        """One CDS Hooks invocation: patient context in, cards out.

        Returning zero cards is the expected outcome whenever the guidelines do not
        clearly answer for this patient, and is not an error. That is the whole design:
        deployed decision support is ignored because it fires too often, so this service
        stays silent rather than showing a box a clinician will learn to dismiss.
        """
        from newbieduo.service import cds_hooks

        if service_id != cds_hooks.SERVICE_ID:
            self._json(404, {"error": f"unknown CDS service {service_id!r}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        # Still warming: say so immediately instead of holding the connection open.
        # A record system opening a chart cannot wait, and a 503 with Retry-After is
        # the honest answer - the caller shows no card and may come back, which is
        # exactly what it would do for any other unavailable service.
        if CONFIG.cds_state == "warming":
            self._json(503, {"error": "warming up", "retry_after_seconds": 15})
            return

        started = time.time()
        try:
            # Warming is the startup path's job now; a hook never triggers it inline.
            response = cds_hooks.handle(request, cds_answerer, warm_first=False)
        except Exception:  # noqa: BLE001 - a hook failure must not disturb the record
            traceback.print_exc()
            self._json(200, {"cards": []})
            return

        response["extension"] = {"latency_ms": round((time.time() - started) * 1000)}
        self._json(200, response)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path

        if path.startswith("/cds-services/"):
            self._cds_hook(path.rsplit("/", 1)[-1])
            return

        if path != "/api/ask":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            query = (payload.get("query") or "").strip()
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        if not query:
            self._json(400, {"error": "empty query"})
            return

        started = time.time()
        try:
            result = answer_module.build_answer(
                query, CONFIG.top_k, CONFIG.dense_model, CONFIG.reranker_model,
                rerank_depth=CONFIG.rerank_depth, generate=CONFIG.generate,
            )
        except Exception:  # noqa: BLE001 - never take the demo down on one bad query
            traceback.print_exc()
            self._json(500, {"error": "retrieval failed", "query": query})
            return

        result["latency_ms"] = round((time.time() - started) * 1000)
        result["config"] = {
            "dense_model": CONFIG.dense_model,
            "reranker_model": CONFIG.reranker_model,
            "degraded": CONFIG.degraded,
        }
        self._json(200, result)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="NewbieDuo demo server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dense-model",
        default="abhinand/MedEmbed-large-v0.1",
        help="Best measured: MedEmbed-large-v0.1 (recall@20 0.783).",
    )
    parser.add_argument(
        "--dense-file",
        default=str(ROOT / "data" / "indexes" / "dense" / "abhinand__MedEmbed-large-v0_1.npy"),
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    # Measured on the gold set at depth 30: v2-m3 nDCG 0.6873, MiniLM 0.6529.
    # MiniLM is ~10x faster, which matters when the machine is throttled.
    parser.add_argument("--fast", action="store_true",
                        help="Swap to ms-marco-MiniLM-L-6-v2: ~10x faster reranking, -0.034 nDCG.")
    parser.add_argument("--rerank-depth", type=int, default=30,
                        help="30 measured better AND faster than 50 (0.6873 vs 0.6819).")
    parser.add_argument("--generate-timeout", type=float, default=12.0,
                        help="Seconds to wait on the LLM before falling back to "
                             "evidence-only. Keep it short: the answer is an "
                             "enhancement, the cited evidence is the product.")
    parser.add_argument("--generate", action="store_true",
                        help="Add an LLM-generated answer, verified against the evidence.")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    CONFIG.top_k = args.top_k
    CONFIG.dense_model = None if args.no_dense else args.dense_model
    CONFIG.dense_file = None if args.no_dense else args.dense_file
    CONFIG.rerank_depth = args.rerank_depth
    CONFIG.generate = args.generate
    if args.generate:
        from newbieduo.generation.generate import set_timeout

        set_timeout(args.generate_timeout)
    CONFIG.reranker_model = None if args.no_rerank else args.reranker_model
    if args.fast and not args.no_rerank:
        CONFIG.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    print("NewbieDuo demo")
    print(f"  dense    : {CONFIG.dense_model or 'off (lexical only)'}")
    print(f"  reranker : {CONFIG.reranker_model or 'off'}")
    print("  warming models...")
    threading.Thread(target=warm_models, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  open http://{args.host}:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()


if __name__ == "__main__":
    main()
