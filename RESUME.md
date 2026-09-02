# Nephrolex — where things stand

Everything below is on disk. Nothing needs to be left running.

## Run the demo

```powershell
cd "C:\Users\manue\Documents\Codex\2026-08-16\we\NewbieDuo"
.\.venv\Scripts\python.exe scripts\demo_server.py
```

Open <http://127.0.0.1:8000>. Models warm in ~30s; the dot in the header turns
green when ready. Six example questions are built in, one per demo moment.

With LLM generation (needs a key; falls back to extractive silently without one):

```powershell
$env:GEMINI_API_KEY = "..."
.\.venv\Scripts\python.exe scripts\demo_server.py --generate
```

Escape hatches, in order of severity:
`--fast` (MiniLM reranker, ~10x faster, -0.034 nDCG) ·
`--no-rerank` (~77 ms/query, 0.60 nDCG) ·
`--no-dense` (~34 ms/query, lexical only).

**Nothing else GPU-heavy may run during the demo.** A background evaluation
sharing the GPU took one query from 112 ms to 23 s.

## Pre-flight check — run before presenting

```powershell
.\.venv\Scripts\python.exe scripts\demo_check.py
```

Ten scripted moments end to end (including rendering the cited PDF page) plus
eight unscripted inputs. Asserts the *status* each must produce, not merely that
it responded. Exits non-zero on failure.

Measured latency, all checks passing:

| Config | mean | p50 | max |
|---|---:|---:|---:|
| default, on AC power | 914 ms | 998 ms | 2922 ms |
| default, on battery (throttled) | 2726 ms | 2924 ms | 7628 ms |
| `--fast`, on battery | 167 ms | 184 ms | 447 ms |

## Measured state

Gold set: 59 questions, 109 pinned chunk-level judgements, dev/test split.

| | Original code | Now |
|---|---|---|
| nDCG@10 | 0.2760 | **0.6898** |
| recall@20 | 0.4900 | **0.8046** |
| MRR@10 | 0.2549 | **0.6911** |
| P@1 / P@5 | – | 0.628 / 0.255 |
| zero-evidence questions | 26 / 51 | **3 / 51** |
| test slice nDCG | 0.134 | 0.611 |

Precision@k must be quoted against its ceiling: queries average 2.1 gold chunks,
so P@5 cannot exceed 0.42. We reach **60% of achievable at k=5, 63% at k=1**, and
**94% of questions surface gold evidence in the top 10**.

Safety gate: **0 unsafe answers, 0 false declines** across all 59 cases.

Best configuration (the demo default): `MedEmbed-large-v0.1` dense +
`bge-reranker-v2-m3` at rerank depth 30, over the 1,760-chunk corpus.

Full table: `reports/ABLATION.md`. Every row is a real run whose config is
recorded in the matching `reports/gold_eval_*.json`.

## Hyperparameters (the ones the hackathon brief names)

| Brief | Ours |
|---|---|
| chunk size / overlap | target 900 chars, min 90; median 380. **No overlap** — semantic boundaries, so recommendations are never split |
| k-value | demo top-k 10, eval 20; candidate pool 120; rerank depth 30 |
| embedding model | MedEmbed-large-v0.1, chosen by measuring 7 models |
| retrieval confidence threshold | **none** — abstention is categorical (scope / genre / domain / premise). A score threshold provably cannot separate: an epidemiology question scores 0.838 cosine, above the median answerable one |
| Precision@k | P@1 0.6667 · P@3 0.3399 · P@5 0.2588 · P@10 0.1549 |

Fusion weights: bm25 **0.0** / tfidf 0.30 / dense 0.40 / metadata 0.30. BM25 still
builds the candidate pool but was measured to *hurt* the score (+0.030 nDCG when
dropped from scoring). Original query weighted 0.7 against its expansion 0.3. Length
damping above 600 chars. Chunk-type priors 1.15 / 1.13 / 1.10 / 0.88.

See `ARCHITECTURE.md` for the layer diagram and the measurement behind each decision.

## The four layers the brief requires

1. **Ingestion** — Docling layout-aware parse, evidence-level chunking, page +
   bounding-box provenance. `chunk_docling.py`, `build_indexes.py`
2. **Retrieval** — hybrid BM25 + TF-IDF + dense + metadata, cross-encoder rerank.
   `retrieve.py`
3. **Generation** — strict grounding prompt, then *mechanical verification*: every
   number-with-unit must appear verbatim in the evidence it cites, or the answer is
   discarded and the extractive one shown. `generate.py`
4. **Safety** — scope gate (pre-retrieval), domain gate, confidence gate, false-premise
   gate. `scope.py`, `answer.py`

## Vector store

Real Qdrant, ingested and benchmarked. It is **not** the default, and the benchmark
is why:

| | median | results |
|---|---:|---|
| flat exhaustive index (numpy) | **0.377 ms** | exact |
| Qdrant server (Rust HNSW) | 17.005 ms | 100% identical |

At 1,760 vectors an ANN index has nothing to avoid — scanning everything costs
0.377 ms — so the whole difference is transport. Qdrant becomes the right choice
past roughly 10^5 vectors or when persistence, concurrent writers or replication
matter. `--local` runs it embedded if Docker is unavailable.

Because the measurement settled the question, the code was retired from the active
tree to `scratch/attic/` (`vector_store.py`, `ingest_qdrant_postgres.py`,
`retrieve_qdrant_postgres.py`, `db/`, `docker-compose.yml`) — the *finding* is kept
in `reports/vector_store_benchmark.json`, the unused infrastructure is not. To rerun
it, move those back:

```powershell
docker compose up -d
.\.venv\Scripts\python.exe scratch\attic\scripts\vector_store.py ingest
.\.venv\Scripts\python.exe scratch\attic\scripts\vector_store.py benchmark
```

## Not done

1. **Contextual Retrieval** — pipeline built and tested, needs the generation run
   (~145 requests, ~15 min). All three remaining zero-evidence questions are
   paraphrases, which is exactly what it targets.
   ```powershell
   $env:GEMINI_API_KEY = "..."
   .\.venv\Scripts\python.exe scripts\contextualize_chunks.py --rpm 10
   ```
2. **Generation not yet tested against a live model** — prompt, verifier and the
   no-key path are verified; an actual round-trip is not.
3. **Retrieved-chunk display** — the brief asks for it explicitly ("display
   retrieved chunks before generation"); the UI currently shows the top 3 quotes,
   not the ranked candidate list with scores.
4. **Citation-accuracy and faithfulness metrics** — both named in the brief, both
   easy given the machinery already here.
5. **Gold set expansion** — only 3 unanswerable cases, too few to calibrate
   abstention properly.

## Known weak spots

- `paraphrase` questions are the weakest slice (nDCG ~0.30 vs 0.79 for direct).
- KDIGO Recommendation 1.3.1 is absent from Docling's parse (1 of 170).
