# Nephrolex: Clinical Guideline RAG for Chronic Kidney Disease

Evidence retrieval over KDIGO 2024 and NICE NG203, served inside the EHR through
HL7 CDS Hooks. Every answer quotes the guideline text it rests on, with page-level
and bounding-box citations; questions outside adult CKD are refused.

| Result | Value |
|---|---|
| nDCG@10, 67-question clinician-style gold set | **0.71** (95% CI 0.63–0.78) |
| nDCG@10, held-out set never tuned on | **0.70** |
| nDCG@10, clinician-phrased FAQ set | **0.70** |
| A gold chunk in the top 10 | **93%** of questions |
| Top-1 result is a gold chunk | **63%** of questions |

Measured with three answer keys and paired bootstrap intervals. A design change
was kept only when its interval cleared the noise floor, and every stage of the
retrieval stack is ablated in [`reports/ABLATION.md`](reports/ABLATION.md).

Runs in one container inside your own network; nothing reaches a cloud at runtime.

**Start here:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — the four layers, and the
measurement behind each design decision. [`reports/EVALUATION.md`](reports/EVALUATION.md)
for the gold set and metrics; [`reports/ABLATION.md`](reports/ABLATION.md) for how the
retrieval stack was built up stage by stage.

## Run it

One container, inside your own network. Nothing reaches a cloud at runtime.

```bash
cp .env.example .env                              # optional: a key enables generated phrasing
docker compose --profile build run --rm build     # PDFs -> corpus -> indexes, once
docker compose up -d serve                        # http://127.0.0.1:8000
```

The service reports readiness honestly: until the models are loaded and the five
CDS trigger answers are cached, `GET /api/health` returns `cds_state: "warming"`
and hook requests are answered `503` with `Retry-After` rather than being held
open. Expect roughly 30 seconds.

| endpoint | what it is |
|---|---|
| `/` | the clinician-facing UI |
| `/api/health` | warm state, loaded models, `cds_state` |
| `/cds-services` | HL7 CDS Hooks discovery |
| `/cds-services/ckd-guidance` | the hook itself; returns cards, or `{"cards": []}` |

## Build the corpus

**The guidelines are not in this repository.** KDIGO 2024 and NICE NG203 are
copyrighted and are licensed to you, not to us, so this project ships the engine
that reads them and not the text itself. See [`NOTICE`](NOTICE).

Put your own licensed copies in `data/raw/` and run:

```bash
python scripts/build_corpus.py     # or: docker compose --profile build run --rm build
```

That parses them layout-aware with Docling, chunks them into typed citable units
with per-row bounding boxes, and builds the lexical and dense indexes. Re-run it
on each guideline revision - the corpus staying current, and being re-measured
against the gold set, is the thing that has to be maintained.

`--skip-dense` builds the lexical index only: much faster, and about 0.08 nDCG
worse. `--ocr` handles scanned or low-text PDFs.

## Architecture

1. Docling parses PDFs with layout, OCR, and accurate table extraction.
2. PyMuPDF and pdfplumber cross-check page text extraction.
3. The parser records file hashes, PDF metadata, outlines, page provenance, parser agreement, extracted tables, and flagged pages.
4. The chunker creates CKD evidence atoms:
   - recommendation atoms
   - KDIGO practice points
   - NICE numbered recommendations
   - practice points
   - table rows
   - threshold facts
   - parent sections
   - non-citable router summaries
5. Every chunk keeps raw citable text separate from contextual retrieval text.
6. Every chunk receives CKD topics, exact terms, threshold terms, and deterministic HyPE-style seed questions.
7. Indexing builds:
   - a TF-IDF lexical index
   - MedEmbed-large-v0.1 dense vectors
   - a doc2query index of generated clinician-voice questions, kept separate from
     the chunk text so a generated sentence can never reach a citation
   - the HyPE question file (retrieval boosts on it, so evaluation excludes it)
8. Retrieval fuses four signals convexly - TF-IDF 0.21, dense 0.29, metadata 0.21,
   doc2query 0.29 - over a 120-candidate pool, applies chunk-type and numeric-band
   priors multiplicatively, and reranks the top 30 with a `bge-reranker-v2-m3`
   cross-encoder. Deterministic CKD query expansion, parent-context attachment and
   reference resolution sit around it. BM25 was measured and removed; RRF remains
   available behind `--fusion rrf` for comparison rather than as the default.
9. Evaluation reports nDCG@10, recall@10/@20 and MRR@10 against a 67-case gold set
   with three independent answer keys, a held-out chunk-first set, and a clinical
   FAQ set - with paired bootstrap intervals, so a change is kept only when it
   clears the noise floor.

Never store API keys in this repository. Set `HF_TOKEN` in the shell environment.

## Commands

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
```

Set Hugging Face access in the current terminal:

```powershell
$env:HF_TOKEN = "paste_new_read_only_token_here"
$env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN
$env:DOCLING_INFERENCE_COMPILE_TORCH_MODELS = "false"
$env:TORCH_COMPILE_DISABLE = "1"
$env:TORCHDYNAMO_DISABLE = "1"
```

Parse with Docling layout/table extraction plus parser cross-checks:

```powershell
python .\scripts\parse_guidelines.py --run-docling --docling-bin ".\.venv\Scripts\docling.exe"
```

Use OCR only for scanned PDFs or pages flagged as low-text/disagreement:

```powershell
python .\scripts\parse_guidelines.py --run-docling --docling-ocr --docling-bin ".\.venv\Scripts\docling.exe"
```

Build CKD evidence atoms:

```powershell
python .\scripts\chunk_docling.py
```

Build lexical indexes only:

```powershell
python .\scripts\build_indexes.py --skip-dense
```

Build full dense indexes:

```powershell
python .\scripts\build_indexes.py --dense-model "abhinand/MedEmbed-large-v0.1" --batch-size 4
```

Test retrieval:

```powershell
python .\scripts\retrieve.py "When should an adult with CKD be referred to specialist kidney care?" --top-k 8 --dense-model "abhinand/MedEmbed-large-v0.1"
```

Create a strict evidence pack for the generation layer:

```powershell
python .\scripts\make_evidence_pack.py "When should an adult with CKD be referred to specialist kidney care?" --top-k 8 --dense-model "abhinand/MedEmbed-large-v0.1" --output .\reports\sample_evidence_pack.json
```

Add reranking after the reranker model is downloaded:

```powershell
python .\scripts\retrieve.py "When should the Kidney Failure Risk Equation be used?" --top-k 8 --dense-model "abhinand/MedEmbed-large-v0.1" --reranker-model "BAAI/bge-reranker-v2-m3"
```

Evaluate retrieval:

```powershell
python .\scripts\evaluate_gold.py --top-k 20 --dense-model "abhinand/MedEmbed-large-v0.1" --reranker-model "BAAI/bge-reranker-v2-m3"
```

Fast smoke test without model downloads. This runs lexical-only and scores well
below the headline - it checks that the pipeline executes, not how well it retrieves:

```powershell
python .\scripts\parse_guidelines.py
python .\scripts\chunk_docling.py
python .\scripts\build_indexes.py --skip-dense
python .\scripts\evaluate_gold.py --top-k 8
```
