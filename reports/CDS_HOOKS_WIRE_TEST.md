# CDS Hooks: what was actually exercised over HTTP

Run 2026-08-20 against `scripts/demo_server.py --port 8071`, extractive mode.

`scripts/check_cds_hooks.py` calls `cds_hooks.handle()` **in process**. It proves the
card-building logic and the silence property; it does not touch HTTP, so until this run
the wire path had never been exercised. Everything below is `curl` against a running
server, with the response bodies kept in `reports/` alongside this note.

## 1. Discovery

```
GET http://127.0.0.1:8071/cds-services        ->  HTTP 200
```

```json
{"services": [{
  "hook": "patient-view",
  "id": "ckd-guidance",
  "title": "CKD guideline evidence (KDIGO 2024 / NICE NG203)",
  "description": "Surfaces the guideline text that applies to this patient's kidney function, quoted verbatim with a citation to page and position. Returns no card when the guidelines do not clearly answer.",
  "prefetch": {
    "egfr": "Observation?patient={{context.patientId}}&code=http://loinc.org|33914-3&_sort=-date&_count=1",
    "acr":  "Observation?patient={{context.patientId}}&code=http://loinc.org|9318-7&_sort=-date&_count=1"}}]}
```

## 2. Hook invocation, spec-shaped request

Request carried `hookInstance`, `fhirServer`, `context.userId`, `context.patientId`, and
`prefetch` as FHIR `Bundle` searchsets of LOINC-coded `Observation` resources.

| patient | prefetch | HTTP | time | cards |
|---|---|---|---|---|
| 84213 | eGFR 24, ACR 85 | 200 | **689.6 s** (cold) | 3 |
| 84213 | same, repeated | 200 | **0.015 s** | 3 |
| 90471 | eGFR 96, ACR 1.2 | 200 | **0.003 s** | **0** |

The healthy patient returns `{"cards": []}` over the wire - the silence property holds
through HTTP, not just in process.

Each card came back with `uuid`, `summary`, `indicator`, `detail` (verbatim guideline
text plus citation), `source`, and a non-spec `extension` carrying the trigger name, the
plain-language reason, the question asked, and the `chunk_id` and `page` it resolves to.

## 3. The cold start was a real problem, and is fixed

**689 seconds on the first call**, originally. The handler was invoked with
`warm_first=True`, so the first request loaded the models and retrieved all five trigger
questions before answering. No record system would wait: a card is meant to appear while
the chart is still opening.

It was a startup problem rather than a per-request one - the same call served warm was
15 ms - and it is worth fixing eagerly precisely because **the five trigger questions are
fixed**. Only *which* of them fire depends on the patient, so the entire cache can be
built before the first patient is ever seen.

Two changes in `scripts/demo_server.py`:

- `warm_cds()` runs at startup, chained after model warm-up so it measures retrieval
  rather than model loading. Both it and the request path now call the same
  `cds_answerer`, because a warm-up that fills the cache through a different function
  fills it with answers the request will not find.
- A hook arriving while `cds_state == "warming"` is answered **503 with `Retry-After`**
  instead of holding the connection. The caller shows no card and may come back, which
  is what it would do for any unavailable service - far better than a socket held open
  for minutes.

Re-measured after the fix, full stack (`degraded: null`, MedEmbed + bge-reranker-v2-m3):

| call | before | after |
|---|---|---|
| hook during warm-up | *(would block)* | **HTTP 503 in 0.003 s**, `Retry-After: 15` |
| **first** hook after startup | **689.642 s** | **0.004 s** |
| healthy patient, first call | n/a | 0.003 s, `{"cards": []}` |
| startup to `cds_state: ready` | n/a | ~30 s, of which 5.5 s is trigger warming |

`GET /api/health` now reports `cds_state` and `cds_warm_seconds`, so "is it ready" is a
question with an answer rather than something to infer from a stopwatch.

### One measurement had to be thrown away

An earlier run of this fix reported the triggers warming in 1.0 s. That run was invalid:
`/api/health` also carried `degraded: "RemoteProtocolError: Server disconnected without
sending a response"`, meaning model loading had failed and the server had silently
fallen back to lexical-only retrieval. The 1.0 s measured the wrong stack. Re-run with
the models actually loaded, warming takes 5.5 s. This is the third time in this project
that a health/provenance field has caught a run measuring something other than what it
claimed, and the reason those fields exist.

## 4. What this does *not* demonstrate

Stated plainly, because the difference matters:

- **No EHR has called this.** No vendor sandbox, no production FHIR server, no real
  patient data. "A conformant service any EHR can call" is the defensible claim;
  "deployed in an EHR" is not, and is not claimed.
- **No JWT authentication.** The spec has record systems authenticate to services with a
  signed bearer token. Not implemented; every request is served unauthenticated.
- **No FHIR fetch fallback.** The service relies entirely on `prefetch`. The spec allows
  a service to call back to `fhirServer` with the supplied token when prefetch is absent
  or incomplete; here, an absent value simply means fewer triggers fire.
- **No `hook-instance` deduplication or feedback endpoint**, both of which a production
  deployment would want.
