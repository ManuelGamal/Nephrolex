# The 20 questions clinicians ask most

## Result

**100% success@5** - on every one of the 20 commonest CKD questions, a correct and
citable guideline chunk appears in the top five results.

| metric | value |
|---|---|
| success@1 | 14/20 = **70%** |
| success@3 | 18/20 = **90%** |
| success@5 | 20/20 = **100%** |
| success@10 | 20/20 = 100% |
| nDCG@10 | 0.7013 |
| recall@10 | 0.8083 |
| recall@20 | 0.8500 |
| MRR@10 | 0.8058 |

Reproduce with `python scripts/evaluate_faq.py`, which prints both figures every run.

### This number moved, and how

The first run of this benchmark scored 90% success@5 and nDCG 0.5456, failing on
`faq_04` and `faq_11`. Diagnosing them found that the cross-encoder was filling the top
four positions with near-identical rows of NICE's monitoring table and pushing the
correct answer out of the top twenty - it scores raw chunk text and cannot see the
generated questions that surfaced the candidate.

Showing the reranker those expansions (`rerank_with_doc2query`) fixes both. That change
had already been built, measured, and *reverted* once, because it costs the
67-question gold set recall@20 (0.8483 -> 0.8184) and takes its end-to-end audit from
69/75 to 67/75. This benchmark was the third piece of evidence and reopened the
decision; the full trade-off is recorded in `retrieve.py` beside the parameter and in
`ABLATION.md`. Both configurations are published.

The questions were frozen before any of this and were not touched.

## Why success@k is the primary number here, and nDCG is not

The gold is pinned by sweeping the corpus for *every* chunk that states the answer -
both printings of a KDIGO recommendation, and both guidelines where they agree. That is
the correct answer key, but it means nDCG@10 measures "found every printing" rather than
"answered the question". `faq_06` asks at what eGFR someone is in kidney failure; it
pins three chunks and is scored on all three - while a
clinician reading that one chunk has a complete, cited answer.

For a point-of-care lookup the question is whether a correct chunk appeared high enough
to be read. Both metrics are reported together, always, because reporting only the
kinder one is the failure this project's evaluation work exists to prevent.

## Where the questions come from

Externally sourced, from the primary-care literature on what clinicians ask about CKD
and where they diverge from guidance - not from the corpus, and not from anything this
system returned:

- GPs' views on managing advanced CKD in primary care, BJGP 65(636)
- Clinician agreement on early-stage CKD monitoring in primary care, PMC4800136
- Referral and management options for patients with CKD, PMC5060784
- UK Kidney Association, Management of patients with CKD
- KDOQI US Commentary on the KDIGO 2024 CKD Guideline, AJKD

Selection criterion: the guidelines state one unambiguous answer. That is a judgement
about the guideline, not about the retriever. A question whose answer is a matter of
interpretation, or where KDIGO and NICE disagree without the question naming one, is not
in the file. Gold was pinned by regex sweep of the whole corpus (`build_faq_set.py`),
no chunk ID written by hand, and **the file was frozen before the first evaluation ran**.

One question was initially unpinnable (`faq_20`, what to check after starting an ACEi).
The pattern was wrong - it excluded KDIGO and its NICE regex did not match. Corrected
against the guideline text, not against retrieval output. A pattern widened until it
matches something is how free credit gets in.

## What it is not

**Not independent of the main gold set.** Referral, monitoring, blood pressure, staging
and medication safety are simultaneously the commonest clinical questions and the main
categories of the existing 67-question set, so the overlap is large by construction.
This measures "how does the system do on what comes up most", not "how does it do on
fresh material". The 46-question chunk-first held-out set remains the independent one.

**Not the same task as the other benchmarks.** nDCG 0.7013 here is not directly
comparable to 0.7080 on the gold set: this answer key is strictly more demanding,
pinning every printing where the gold set often pins one.

## The two failures, and their fix

| case | question | outcome |
|---|---|---|
| `faq_04_confirm_egfr` | Do I need to repeat an abnormal eGFR before diagnosing CKD? | now rank 4 |
| `faq_11_acr_rise` | How much does ACR have to rise before it means something? | now rank 1 |

Neither was fixed by changing the question or the gold. `faq_11` pins KDIGO Practice
Point 2.1.5, which the doc2query signal ranked first on its own all along; the reranker
was burying it.

One caveat that belongs with this: `faq_04` and the gold set's `egfr_confirm` are the
same clinical question in different words, and the change that fixes the first breaks
the second. It trades phrasing sensitivity rather than removing it.

## By clinical topic

| topic | n | nDCG@10 | recall@20 |
|---|---|---|---|
| staging | 2 | 0.8519 | 1.0000 |
| medication safety | 4 | 0.8420 | 1.0000 |
| blood pressure | 2 | 0.7500 | 1.0000 |
| monitoring | 3 | 0.7421 | 0.8333 |
| referral | 2 | 0.7385 | 1.0000 |
| testing | 3 | 0.5778 | 0.6111 |
| treatment | 2 | 0.5761 | 0.8334 |
| definition | 2 | 0.4325 | 0.5000 |

Staging and medication safety are strongest. Referral and blood pressure - the two
topics the primary-care literature flags as causing clinicians the most difficulty -
both sit above 0.73 with full recall@20, which is the claim worth making: the tool is
good where the documented need is.
